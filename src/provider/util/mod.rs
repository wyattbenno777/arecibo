//! Utilities for provider module.
pub(in crate::provider) mod fb_msm;
pub mod web_gpu_msm;
pub mod msm {
  use halo2curves::{msm::best_multiexp, CurveAffine};

  use crate::provider::util::web_gpu_msm::{curve_to_js_point, jsvalue_to_primefield};

  // this argument swap is useful until Rust gets named arguments
  // and saves significant complexity in macro code
  pub fn cpu_best_msm<C: CurveAffine>(bases: &[C], scalars: &[C::Scalar]) -> C::Curve {
    best_multiexp(scalars, bases)
  }

  pub fn web_gpu_best_msm<C: CurveAffine>(bases: &[C], scalars: &[C::Scalar]) -> C::Curve {
    use crate::provider::util::web_gpu_msm::{primefield_to_bigint_js, run_gpu_msm};
    use std::sync::mpsc;
    use wasm_bindgen_futures::spawn_local;

    // Mock data for testing with BigInt
    let js_scalars = scalars
      .iter()
      .map(|s| primefield_to_bigint_js(*s))
      .collect(); // vec![JsValue::bigint_from_str("1")]; // Use JsValue to represent BigInt
    let js_bases = bases.iter().map(|b| curve_to_js_point(*b)).collect(); // vec![U32ArrayPoint::new(

    // Create a channel to communicate between the async task and the caller
    let (tx, rx) = mpsc::channel();

    // Spawn the async task
    spawn_local(async move {
      if let Ok(result) = run_gpu_msm(js_bases, js_scalars).await {
        let x = jsvalue_to_primefield::<C::Base>(result.x()).unwrap();
        let y = jsvalue_to_primefield::<C::Base>(result.y()).unwrap();

        let curve_result = C::from_xy(x, y).unwrap().into();

        // Send the result back to the caller
        tx.send(curve_result).unwrap();
      }
    });

    // Block until the result is received
    rx.recv().unwrap()
  }

  pub async fn web_gpu_best_msm_async<C: CurveAffine>(
    bases: &[C],
    scalars: &[C::Scalar],
  ) -> C::Curve {
    use crate::provider::util::web_gpu_msm::{primefield_to_bigint_js, run_gpu_msm};

    // Mock data for testing with BigInt
    let js_scalars = scalars
      .iter()
      .map(|s| primefield_to_bigint_js(*s))
      .collect(); // vec![JsValue::bigint_from_str("1")]; // Use JsValue to represent BigInt
    let js_bases = bases.iter().map(|b| curve_to_js_point(*b)).collect(); // vec![U32ArrayPoint::new(

    let result = run_gpu_msm(js_bases, js_scalars).await;
    assert!(result.is_ok());
    let msm_result = result.unwrap();
    let x = jsvalue_to_primefield::<C::Base>(msm_result.x()).unwrap();
    let y = jsvalue_to_primefield::<C::Base>(msm_result.y()).unwrap();

    C::from_xy(x, y).unwrap().into()
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
mod tests {
  use crate::{
    provider::{
      util::msm::{cpu_best_msm, web_gpu_best_msm_async},
      Bn256EngineKZG,
    },
    traits::Engine,
  };

  use group::{Curve, Group};
  use halo2curves::bn256::{Fr, G1Affine};
  use rand::Rng;
  use wasm_bindgen::prelude::wasm_bindgen;
  use wasm_bindgen_test::*;
  use web_sys::console;

  wasm_bindgen_test_configure!(run_in_browser);

  type E = Bn256EngineKZG;

  #[wasm_bindgen]
  extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
  }

  #[wasm_bindgen_test]
  async fn test_run_gpu_msm() {
    use rand::thread_rng;
    // Create a vector of random Fr elements
    let mut rng = thread_rng();
    let scalars: Vec<Fr> = (0..4)
      .map(|_| {
        let random_u64: u64 = rng.gen();
        Fr::from(random_u64)
      })
      .collect();
    let bases: Vec<G1Affine> = (0..4)
      .map(|_| <E as Engine>::GE::random(&mut rng).to_affine())
      .collect();

    let start = now();
    let cpu_msm_result = cpu_best_msm(&bases, &scalars);
    console::log_1(&format!("cpu_msm_result: {:?}", cpu_msm_result).into());
    let end = now();
    console::log_1(&format!("cpu_msm_time: {:?}", end - start).into());

    let start = now();
    let gpu_msm_result = web_gpu_best_msm_async(&bases, &scalars).await;
    let end = now();
    console::log_1(&format!("gpu_msm_time: {:?}", end - start).into());

    console::log_1(&format!("cpu_msm_result: {:?}", cpu_msm_result).into());
    console::log_1(&format!("gpu_msm_result: {:?}", gpu_msm_result).into());
    assert_eq!(cpu_msm_result, gpu_msm_result);
  }
}
