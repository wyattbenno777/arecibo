//! This module implements the HyperNova folding scheme.
use crate::constants::NUM_CHALLENGE_BITS;
use crate::gadgets::scalar_as_base;
use crate::spartan::math::Math;
use crate::spartan::polys::eq::EqPolynomial;
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::spartan::sumcheck::SumcheckProof;
use crate::traits::ROTrait;
use crate::NovaError;
use crate::{
  constants::DEFAULT_ABSORBS,
  cyclefold::util::absorb_primary_r1cs,
  r1cs::{LR1CSInstance, R1CSInstance, R1CSShape, R1CSWitness},
  traits::{CurveCycleEquipped, Dual, Engine, ROConstants, TranscriptEngineTrait},
};
use ff::Field;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

/// A SNARK that holds the proof of a step of an incremental computation
pub struct NIFS<E: CurveCycleEquipped> {
  pub(crate) sc: SumcheckProof<E>,
  pub(crate) sigmas: Vec<E::Scalar>,
  pub(crate) thetas: Vec<E::Scalar>,
}

impl<E> NIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prove a step of an incremental computation
  pub fn prove(
    S: &R1CSShape<E>,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    (U1, W1): (&LR1CSInstance<E>, &R1CSWitness<E>),
    (U2, W2): (&R1CSInstance<E>, &R1CSWitness<E>),
  ) -> Result<(Self, (LR1CSInstance<E>, R1CSWitness<E>)), NovaError> {
    // rho, gamma, beta
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(*pp_digest);
    absorb_primary_r1cs::<E, Dual<E>>(U2, &mut ro);
    let rho = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(rho);
    let gamma = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(gamma);
    let s = S.num_vars.log_2() + 1;
    let beta = {
      ro.squeeze_vec(NUM_CHALLENGE_BITS, s)
        .iter()
        .map(|b| scalar_as_base::<Dual<E>>(*b))
        .collect::<Vec<_>>()
    };

    // Compute L_j = eq(rx, y) • H_j(y)
    let z1 = [W1.W.as_slice(), [U1.u].as_slice(), U1.X.as_slice()].concat();
    let mut poly_ABC = {
      let (mut evals_A, mut evals_B, mut evals_C) = S.multiply_vec(&z1)?;
      evals_A.resize(S.num_vars * 2, E::Scalar::ZERO);
      evals_B.resize(S.num_vars * 2, E::Scalar::ZERO);
      evals_C.resize(S.num_vars * 2, E::Scalar::ZERO);
      let evals_ABC = (0..evals_A.len())
        .into_par_iter()
        .map(|i| (evals_A[i] + gamma * evals_B[i] + gamma * gamma * evals_C[i]))
        .collect::<Vec<E::Scalar>>();
      MultilinearPolynomial::new(evals_ABC)
    };
    let evals_rx = EqPolynomial::evals_from_points(&U1.rx);
    let mut evals_rx = MultilinearPolynomial::new(evals_rx);
    let L_comb_func = |abc: E::Scalar, e: E::Scalar| -> E::Scalar { abc * e };

    // Q = eq(beta, x) • G(x)
    let z2 = [
      W2.W.as_slice(),
      [E::Scalar::ONE].as_slice(),
      U2.X.as_slice(),
    ]
    .concat();
    let eq_beta = EqPolynomial::new(beta.clone());
    let gamma_cubed = gamma * gamma * gamma;
    let mut poly_beta = MultilinearPolynomial::new(eq_beta.evals());
    let (mut poly_Az, mut poly_Bz, mut poly_Cz) = {
      let (mut poly_Az, mut poly_Bz, mut poly_Cz) = S.multiply_vec(&z2)?;
      poly_Az.resize(S.num_vars * 2, E::Scalar::ZERO);
      poly_Bz.resize(S.num_vars * 2, E::Scalar::ZERO);
      poly_Cz.resize(S.num_vars * 2, E::Scalar::ZERO);
      (
        MultilinearPolynomial::new(poly_Az),
        MultilinearPolynomial::new(poly_Bz),
        MultilinearPolynomial::new(poly_Cz),
      )
    };
    let Q_comb_func = |a: E::Scalar, b: E::Scalar, c: E::Scalar, eq: E::Scalar| -> E::Scalar {
      (a * b - c) * eq * gamma_cubed
    };

    // sumcheck
    let comb_func =
      |L_abc: E::Scalar,
       L_e: E::Scalar,
       Q_a: E::Scalar,
       Q_b: E::Scalar,
       Q_c: E::Scalar,
       Q_eq: E::Scalar|
       -> E::Scalar { L_comb_func(L_abc, L_e) + Q_comb_func(Q_a, Q_b, Q_c, Q_eq) };
    let claim = U1.vs[0] + gamma * U1.vs[1] + gamma * gamma * U1.vs[2];
    let (sc, rx_p, _) = SumcheckProof::<E>::prove_cubic_hypernova(
      claim,
      s,
      &mut poly_ABC,
      &mut evals_rx,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      &mut poly_beta,
      comb_func,
      &mut E::TE::new(b"transcript"), // TODO: change to use poseidonRO
    )?;

    // Send over sigmas and thetas
    let sigmas = {
      let (mut poly_Az, mut poly_Bz, mut poly_Cz) = S.multiply_vec(&z1)?;
      poly_Az.resize(S.num_vars * 2, E::Scalar::ZERO);
      poly_Bz.resize(S.num_vars * 2, E::Scalar::ZERO);
      poly_Cz.resize(S.num_vars * 2, E::Scalar::ZERO);
      vec![
        MultilinearPolynomial::new(poly_Az).evaluate(&rx_p),
        MultilinearPolynomial::new(poly_Bz).evaluate(&rx_p),
        MultilinearPolynomial::new(poly_Cz).evaluate(&rx_p),
      ]
    };
    let thetas = {
      let (mut poly_Az, mut poly_Bz, mut poly_Cz) = S.multiply_vec(&z2)?;
      poly_Az.resize(S.num_vars * 2, E::Scalar::ZERO);
      poly_Bz.resize(S.num_vars * 2, E::Scalar::ZERO);
      poly_Cz.resize(S.num_vars * 2, E::Scalar::ZERO);
      vec![
        MultilinearPolynomial::new(poly_Az).evaluate(&rx_p),
        MultilinearPolynomial::new(poly_Bz).evaluate(&rx_p),
        MultilinearPolynomial::new(poly_Cz).evaluate(&rx_p),
      ]
    };

    // Output the folded instance, witness pair
    let U = U1.fold(U2, rho, &rx_p, &sigmas, &thetas)?;
    let W = W1.fold(W2, rho)?;
    Ok((Self { sc, sigmas, thetas }, (U, W)))
  }

  /// Verify a fold
  pub fn verify(
    &self,
    s: usize,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &LR1CSInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<LR1CSInstance<E>, NovaError> {
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(*pp_digest);
    absorb_primary_r1cs::<E, Dual<E>>(U2, &mut ro);
    let rho = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(rho);
    let gamma = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(gamma);
    let beta = {
      ro.squeeze_vec(NUM_CHALLENGE_BITS, s)
        .iter()
        .map(|b| scalar_as_base::<Dual<E>>(*b))
        .collect::<Vec<_>>()
    };
    let claim = U1.vs[0] + gamma * U1.vs[1] + gamma * gamma * U1.vs[2];
    let (new_claim, rx_p) = self
      .sc
      .verify(claim, s, 3, &mut E::TE::new(b"transcript"))?;
    let e1 = EqPolynomial::new(U1.rx.to_vec()).evaluate(&rx_p);
    let cl = (self.sigmas[0] + gamma * self.sigmas[1] + gamma * gamma * self.sigmas[2]) * e1;
    let e2 = EqPolynomial::new(beta).evaluate(&rx_p);
    let gamma_cubed = gamma * gamma * gamma;
    let cr = (self.thetas[0] * self.thetas[1] - self.thetas[2]) * e2 * gamma_cubed;
    if cl + cr != new_claim {
      assert_eq!(cl + cr, new_claim);
      return Err(NovaError::InvalidSumcheckProof);
    }
    let U = U1.fold(U2, rho, &rx_p, &self.sigmas, &self.thetas)?;
    Ok(U)
  }
}

#[cfg(test)]
mod tests {
  use crate::frontend::r1cs::NovaShape;
  use crate::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use crate::hypernova::nifs::NIFS;
  use crate::r1cs::{LR1CSInstance, R1CSInstance, R1CSShape, R1CSWitness};
  use crate::spartan::math::Math;
  use crate::traits::{CurveCycleEquipped, Dual, ROConstants};
  use crate::CommitmentKey;
  use crate::{
    frontend::{r1cs::NovaWitness, solver::SatisfyingAssignment, test_shape_cs::TestShapeCS},
    provider::{Bn256EngineKZG, PallasEngine, Secp256k1Engine},
    traits::{snark::default_ck_hint, Engine},
  };
  use ff::{Field, PrimeField};

  #[test]
  fn test_tiny_r1cs_bellpepper() {
    test_tiny_r1cs_bellpepper_with::<PallasEngine>();
    test_tiny_r1cs_bellpepper_with::<Bn256EngineKZG>();
    test_tiny_r1cs_bellpepper_with::<Secp256k1Engine>();
  }

  fn test_tiny_r1cs_bellpepper_with<E: CurveCycleEquipped>() {
    // First create the shape
    let mut cs: TestShapeCS<E> = TestShapeCS::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, None);
    let (shape, ck) = cs.padded_r1cs_shape(&*default_ck_hint());
    let ro_consts = ROConstants::<Dual<E>>::default();

    // Now get the instance and assignment for one instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(5)));
    let (U1, W1) = cs.padded_r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the first instance is satisfiable
    shape.is_sat(&ck, &U1, &W1).unwrap();

    // Now get the instance and assignment for second instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(135)));
    let (U2, W2) = cs.padded_r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the second instance is satisfiable
    shape.is_sat(&ck, &U2, &W2).unwrap();

    // Now get the instance and assignment for second instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(100)));
    let (U3, W3) = cs.padded_r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the second instance is satisfiable
    shape.is_sat(&ck, &U3, &W3).unwrap();

    // Now get the instance and assignment for second instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(100)));
    let (U4, W4) = cs.padded_r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the second instance is satisfiable
    shape.is_sat(&ck, &U4, &W4).unwrap();

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &shape,
      &U1,
      &W1,
      &U2,
      &W2,
      &U3,
      &W3,
      &U4,
      &W4,
    );
  }

  fn execute_sequence<E: CurveCycleEquipped>(
    ck: &CommitmentKey<E>,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &<E as Engine>::Scalar,
    S: &R1CSShape<E>,
    U1: &R1CSInstance<E>,
    W1: &R1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    U3: &R1CSInstance<E>,
    W3: &R1CSWitness<E>,
    U4: &R1CSInstance<E>,
    W4: &R1CSWitness<E>,
  ) {
    let s = S.num_vars.log_2() + 1;
    // produce a default running instance
    let mut r_W = R1CSWitness::default(S);
    let mut r_U = LR1CSInstance::default(S);
    S.is_sat_linearized(ck, &r_U, &r_W).unwrap();

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let (nifs, (_U, W)) = NIFS::prove(S, ro_consts, pp_digest, (&r_U, &r_W), (U1, W1)).unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let U = nifs.verify(s, ro_consts, pp_digest, &r_U, U1).unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;
    S.is_sat_linearized(ck, &r_U, &r_W).unwrap();

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let (nifs, (_U, W)) = NIFS::prove(S, ro_consts, pp_digest, (&r_U, &r_W), (U2, W2)).unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let U = nifs.verify(s, ro_consts, pp_digest, &r_U, U2).unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // check if the running instance is satisfiable
    S.is_sat_linearized(ck, &r_U, &r_W).unwrap();

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let (nifs, (_U, W)) = NIFS::prove(S, ro_consts, pp_digest, (&r_U, &r_W), (U3, W3)).unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let U = nifs.verify(s, ro_consts, pp_digest, &r_U, U3).unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // check if the running instance is satisfiable
    S.is_sat_linearized(ck, &r_U, &r_W).unwrap();

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let (nifs, (_U, W)) = NIFS::prove(S, ro_consts, pp_digest, (&r_U, &r_W), (U4, W4)).unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let U = nifs.verify(s, ro_consts, pp_digest, &r_U, U4).unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // check if the running instance is satisfiable
    S.is_sat_linearized(ck, &r_U, &r_W).unwrap();
  }

  fn synthesize_tiny_r1cs_bellpepper<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    x_val: Option<Scalar>,
  ) -> Result<(), SynthesisError> {
    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = AllocatedNum::alloc_infallible(cs.namespace(|| "x"), || x_val.unwrap());
    let _ = x.inputize(cs.namespace(|| "x is input"));

    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + Scalar::from(5u64))
    })?;
    let _ = y.inputize(cs.namespace(|| "y is output"));

    cs.enforce(
      || "y = x^3 + x + 5",
      |lc| {
        lc + x_cu.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    Ok(())
  }
}
