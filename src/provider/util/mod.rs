//! Utilities for provider module.
pub(in crate::provider) mod fb_msm;
pub mod msm {
  use halo2curves::{msm::best_multiexp, CurveAffine};
  use group::Group;
  use ff::Field;
  // this argument swap is useful until Rust gets named arguments
  // and saves significant complexity in macro code
  pub fn cpu_best_msm<C: CurveAffine>(bases: &[C], scalars: &[C::Scalar]) -> C::Curve {
    let (bases, scalars): (Vec<C>, Vec<C::Scalar>) = bases
      .iter()
      .zip(scalars)
      .filter(|&(_, ref s)| *s != &C::Scalar::ZERO)
      .unzip();
    best_multiexp(&scalars, &bases)
  }

  pub fn web_gpu_best_msm<C: CurveAffine>(bases: &[C], scalars: &[C::Scalar]) -> C::Curve {
    let (bases_one, scalars_one, bases_rest, scalars_rest) =
    bases
        .iter()
        .zip(scalars)
        .filter(|&(_, s)| *s != C::Scalar::ZERO)
        .fold(   
            (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            |(mut b1, mut s1, mut b0, mut s0), (b, s)| {
                if *s == C::Scalar::ONE {
                    b1.push(*b);
                    s1.push(*s);
                } else {
                    b0.push(*b);
                    s0.push(*s);
                }
                (b1, s1, b0, s0)
            },
        );
    let boolean_sum = scalars_one
      .iter()
      .zip(bases_one.iter())
      .fold(C::Curve::identity(), |mut acc, (_, base)| {
        acc += *base;
        acc
      });
    let rest_sum = best_multiexp(&scalars_rest, &bases_rest);

    boolean_sum + rest_sum
  }
}

pub mod field {
  use crate::errors::NovaError;
  use ff::{BatchInverter, Field};

  #[inline]
  pub fn batch_invert<F: Field>(mut v: Vec<F>) -> Result<Vec<F>, NovaError> {
    // we only allocate the scratch space if every element of v is nonzero
    let mut scratch_space = v
      .iter()
      .map(|x| {
        if !x.is_zero_vartime() {
          Ok(*x)
        } else {
          Err(NovaError::InternalError)
        }
      })
      .collect::<Result<Vec<_>, _>>()?;
    let _ = BatchInverter::invert_with_external_scratch(&mut v, &mut scratch_space[..]);
    Ok(v)
  }
}

pub mod iterators {
  use ff::Field;
  use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
  use rayon_scan::ScanParallelIterator;
  use std::{
    borrow::Borrow,
    iter::DoubleEndedIterator,
    ops::{AddAssign, MulAssign},
  };

  pub trait DoubleEndedIteratorExt: DoubleEndedIterator {
    /// This function employs Horner's scheme and core traits to create a combination of an iterator input with the powers
    /// of a provided coefficient.
    fn rlc<T, F>(&mut self, coefficient: &F) -> T
    where
      T: Clone + for<'a> MulAssign<&'a F> + for<'r> AddAssign<&'r T>,
      Self::Item: Borrow<T>,
    {
      let mut iter = self.rev();
      let Some(fst) = iter.next() else {
        panic!("input iterator should not be empty")
      };

      iter.fold(fst.borrow().clone(), |mut acc, item| {
        acc *= coefficient;
        acc += item.borrow();
        acc
      })
    }
  }

  impl<I: DoubleEndedIterator> DoubleEndedIteratorExt for I {}

  pub trait IndexedParallelIteratorExt: IndexedParallelIterator {
    /// This function core traits to create a combination of an iterator input with the powers
    /// of a provided coefficient.
    fn rlc<T, F>(self, coefficient: &F) -> T
    where
      F: Field,
      Self::Item: Borrow<T>,
      T: Clone + for<'a> MulAssign<&'a F> + for<'r> AddAssign<&'r T> + Send + Sync,
    {
      debug_assert!(self.len() > 0);
      // generate an iterator of powers of the right length
      let v = {
        let mut v = vec![*coefficient; self.len()];
        v[0] = F::ONE;
        v
      };
      // the collect is due to Scan being unindexed
      let powers: Vec<_> = v.into_par_iter().scan(|a, b| *a * *b, F::ONE).collect();

      self
        .zip_eq(powers.into_par_iter())
        .map(|(pt, val)| {
          let mut pt = pt.borrow().clone();
          pt *= &val;
          pt
        })
        .reduce_with(|mut a, b| {
          a += &b;
          a
        })
        .unwrap()
    }
  }

  impl<I: IndexedParallelIterator> IndexedParallelIteratorExt for I {}
}

#[cfg(test)]
pub mod test_utils {
  //! Contains utilities for testing and benchmarking.
  use crate::{
    spartan::polys::multilinear::MultilinearPolynomial,
    traits::{commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Engine},
  };
  use ff::Field;
  use rand::rngs::StdRng;
  use rand_core::{CryptoRng, RngCore};
  use std::sync::Arc;

  /// Returns a random polynomial, a point and calculate its evaluation.
  pub(crate) fn random_poly_with_eval<E: Engine, R: RngCore + CryptoRng>(
    num_vars: usize,
    mut rng: &mut R,
  ) -> (
    MultilinearPolynomial<<E as Engine>::Scalar>,
    Vec<<E as Engine>::Scalar>,
    <E as Engine>::Scalar,
  ) {
    // Generate random polynomial and point.
    let poly = MultilinearPolynomial::random(num_vars, &mut rng);
    let point = (0..num_vars)
      .map(|_| <E as Engine>::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    // Calculation evaluation of point over polynomial.
    let eval = poly.evaluate(&point);

    (poly, point, eval)
  }

  /// Methods used to test the prove and verify flow of [`MultilinearPolynomial`] Commitment Schemes
  /// (PCS).
  ///
  /// Generates a random polynomial and point from a seed to test a proving/verifying flow of one
  /// of our [`EvaluationEngine`].
  pub(crate) fn prove_verify_from_num_vars<E: Engine, EE: EvaluationEngineTrait<E>>(
    num_vars: usize,
  ) {
    use rand_core::SeedableRng;

    let mut rng = StdRng::seed_from_u64(num_vars as u64);

    let (poly, point, eval) = random_poly_with_eval::<E, StdRng>(num_vars, &mut rng);

    // Mock commitment key.
    let ck = E::CE::setup(b"test", 1 << num_vars);
    let ck = Arc::new(ck);
    // Commits to the provided vector using the provided generators.
    let commitment = E::CE::commit(&ck, poly.evaluations(), &E::Scalar::ZERO);

    prove_verify_with::<E, EE>(ck, &commitment, &poly, &point, &eval, true)
  }

  fn prove_verify_with<E: Engine, EE: EvaluationEngineTrait<E>>(
    ck: Arc<<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey>,
    commitment: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    poly: &MultilinearPolynomial<<E as Engine>::Scalar>,
    point: &[<E as Engine>::Scalar],
    eval: &<E as Engine>::Scalar,
    evaluate_bad_proof: bool,
  ) {
    use crate::traits::TranscriptEngineTrait;
    use std::ops::Add;

    // Generate Prover and verifier key for given commitment key.
    let ock = ck.clone();
    let (prover_key, verifier_key) = EE::setup(ck);

    // Generate proof.
    let mut prover_transcript = E::TE::new(b"TestEval");
    let proof = EE::prove(
      &*ock,
      &prover_key,
      &mut prover_transcript,
      commitment,
      poly.evaluations(),
      point,
      eval,
    )
    .unwrap();
    let pcp = prover_transcript.squeeze(b"c").unwrap();

    // Verify proof.
    let mut verifier_transcript = E::TE::new(b"TestEval");
    EE::verify(
      &verifier_key,
      &mut verifier_transcript,
      commitment,
      point,
      eval,
      &proof,
    )
    .unwrap();
    let pcv = verifier_transcript.squeeze(b"c").unwrap();

    // Check if the prover transcript and verifier transcript are kept in the same state.
    assert_eq!(pcp, pcv);

    if evaluate_bad_proof {
      // Generate another point to verify proof. Also produce eval.
      let altered_verifier_point = point
        .iter()
        .map(|s| s.add(<E as Engine>::Scalar::ONE))
        .collect::<Vec<_>>();
      let altered_verifier_eval =
        MultilinearPolynomial::evaluate_with(poly.evaluations(), &altered_verifier_point);

      // Verify proof, should fail.
      let mut verifier_transcript = E::TE::new(b"TestEval");
      assert!(EE::verify(
        &verifier_key,
        &mut verifier_transcript,
        commitment,
        &altered_verifier_point,
        &altered_verifier_eval,
        &proof,
      )
      .is_err());
    }
  }
}

#[cfg(test)]
pub mod test_msm {
  use halo2curves::{bn256::Bn256, msm::best_multiexp, CurveAffine};
  use msm_webgpu::run_webgpu_msm;

  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    nebula::rs::{PublicParams, RecursiveSNARK, StepCircuit},
    provider::{
      hyperkzg::EvaluationEngine as EvaluationEngineKZG,
      ipa_pc::EvaluationEngine as EvaluationEngineIPA, Bn256EngineKZG, GrumpkinEngine,
    },
    spartan::snark::RelaxedR1CSSNARK,
    traits::{snark::RelaxedR1CSSNARKTrait, Engine},
  };
  use ff::Field;
  use halo2curves::bn256::{Fr, G1Affine};
  use rand::thread_rng;
  use wasm_bindgen::prelude::*;
  use wasm_bindgen_test::*;
  use web_sys::console;

  wasm_bindgen_test_configure!(run_in_browser);

  #[wasm_bindgen]
  extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
  }

  #[derive(Clone, Copy, Debug)]
  pub struct CubicFCircuit {}

  impl CubicFCircuit {
    /// Create a new circuit
    pub fn new() -> Self {
      Self {}
    }
  }

  impl Default for CubicFCircuit {
    fn default() -> Self {
      Self::new()
    }
  }

  impl StepCircuit<halo2curves::bn256::Fr> for CubicFCircuit {
    fn arity(&self) -> usize {
      1
    }
    fn synthesize<CS: ConstraintSystem<halo2curves::bn256::Fr>>(
      &self,
      cs: &mut CS,
      z_in: &[AllocatedNum<halo2curves::bn256::Fr>],
    ) -> Result<Vec<AllocatedNum<halo2curves::bn256::Fr>>, SynthesisError> {
      let five = AllocatedNum::alloc(cs.namespace(|| "five"), || {
        Ok(halo2curves::bn256::Fr::from(5u64))
      })?;
      let z_i = z_in[0].clone();
      let z_i_sq = z_i.mul(cs.namespace(|| "z_i_sq"), &z_i)?;
      let z_i_cube = z_i_sq.mul(cs.namespace(|| "z_i_cube"), &z_i)?;
      let result = z_i_cube.add(cs.namespace(|| "add z_i"), &z_i)?;
      let result = result.add(cs.namespace(|| "add five"), &five)?;

      Ok(vec![result])
    }
    fn non_deterministic_advice(&self) -> Vec<halo2curves::bn256::Fr> {
      vec![]
    }
  }

  #[wasm_bindgen_test]
  async fn test_webgpu_msm_bn254() {
    type E1 = Bn256EngineKZG;
    type E2 = GrumpkinEngine;
    type EE1 = EvaluationEngineKZG<Bn256, E1>;
    type EE2 = EvaluationEngineIPA<E2>;
    type S1 = RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
    type S2 = RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK

    let num_steps = 5;

    let f_circuit = CubicFCircuit::new();

    // produce public parameters
    let start = now();
    console::log_1(&format!("Producing public parameters...").into());
    let rs_pp = PublicParams::<E1>::setup(&f_circuit, &*S1::ck_floor(), &*S2::ck_floor());
    console::log_1(&format!("PublicParams::setup, took {:?} ", now() - start).into());
    console::log_1(
      &format!(
        "Number of constraints per step (primary circuit): {}",
        rs_pp.num_constraints().0
      )
      .into(),
    );
    console::log_1(
      &format!(
        "Number of constraints per step (secondary circuit): {}",
        rs_pp.num_constraints().1
      )
      .into(),
    );
    console::log_1(
      &format!(
        "Number of variables per step (primary circuit): {}",
        rs_pp.num_variables().0
      )
      .into(),
    );
    console::log_1(
      &format!(
        "Number of variables per step (secondary circuit): {}",
        rs_pp.num_variables().1
      )
      .into(),
    );

    // produce a recursive SNARK
    console::log_1(&format!("Generating a RecursiveSNARK...").into());

    let mut IC_i = <E1 as Engine>::Scalar::ZERO;
    let z0 = vec![<E1 as Engine>::Scalar::from(3u64)];
    let mut rs: RecursiveSNARK<E1> = RecursiveSNARK::<E1>::new(&rs_pp, &f_circuit, &z0).unwrap();

    for i in 0..num_steps {
      let start = now();
      rs.prove_step(&rs_pp, &f_circuit, IC_i).unwrap();

      IC_i = rs.increment_commitment(&rs_pp, &f_circuit);
      console::log_1(&format!("RecursiveSNARK::prove {} : took {:?} ", i, now() - start).into());
    }

    // verify the recursive SNARK
    console::log_1(&format!("Verifying a RecursiveSNARK...").into());
    let res = rs.verify(&rs_pp, num_steps, &z0, IC_i);
    console::log_1(&format!("RecursiveSNARK::verify: {:?}", res.is_ok()).into());
    res.unwrap();
  }
}
