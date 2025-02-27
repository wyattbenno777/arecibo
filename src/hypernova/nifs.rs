//! This module implements the HyperNova folding scheme.
use super::{ro_sumcheck::ROSumcheckProof, utils::absorb_split_instance};
use crate::{
  constants::{DEFAULT_ABSORBS, NUM_CHALLENGE_BITS},
  cyclefold::{circuit::CycleFoldCircuit, util::absorb_cyclefold_r1cs},
  frontend::{r1cs::NovaWitness, solver::SatisfyingAssignment, ConstraintSystem},
  gadgets::scalar_as_base,
  r1cs::{
    split::{LR1CSInstance, SplitR1CSInstance, SplitR1CSWitness},
    R1CSInstance, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness,
  },
  spartan::{
    math::Math,
    polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  },
  traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROConstants, ROTrait},
  Commitment, CommitmentKey, NovaError,
};
use ff::{Field, PrimeFieldBits};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
/// A SNARK that holds the proof of a step of an incremental computation
pub struct NIFS<E: CurveCycleEquipped> {
  pub(crate) partial_nifs: PartialNIFS<E>,
  pub(crate) cyclefold_nifs: CycleFoldNIFS<E>,
  pub(crate) cyclefold_nifs_1: CycleFoldNIFS<E>,
  pub(crate) cyclefold_nifs_2: CycleFoldNIFS<E>,
}

impl<E> NIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prove a step of an incremental computation. Implements CycleFold.
  ///
  /// # Note:
  ///
  /// This protocol is modified to handle two splits in the witness.
  pub fn prove(
    (S, S_cyclefold): (&R1CSShape<E>, &R1CSShape<Dual<E>>),
    ck_cyclefold: &CommitmentKey<Dual<E>>,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    (U1, W1): (&LR1CSInstance<E>, &SplitR1CSWitness<E>),
    (U2, W2): (&SplitR1CSInstance<E>, &SplitR1CSWitness<E>),
    (U1_cyclefold, W1_cyclefold): (&RelaxedR1CSInstance<Dual<E>>, &RelaxedR1CSWitness<Dual<E>>),
  ) -> Result<
    (
      Self,
      (LR1CSInstance<E>, SplitR1CSWitness<E>),
      (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>),
    ),
    NovaError,
  > {
    let (partial_nifs, (U, W), rho) =
      PartialNIFS::<E>::prove(S, ro_consts, pp_digest, (U1, W1), (U2, W2))?;

    // CycleFold for main part of the witness
    let (cyclefold_nifs, (U_cyclefold_temp, W_cyclefold_temp)) = CycleFoldNIFS::<E>::prove(
      S_cyclefold,
      ck_cyclefold,
      ro_consts,
      U1.comm_W,
      U2.aux.comm_W,
      rho,
      (U1_cyclefold, W1_cyclefold),
    )?;

    // CycleFold for first split of the witness
    let (cyclefold_nifs_1, (U_cyclefold_temp_1, W_cyclefold_temp_1)) = CycleFoldNIFS::<E>::prove(
      S_cyclefold,
      ck_cyclefold,
      ro_consts,
      U1.pre_committed.0,
      U2.pre_committed.0,
      rho,
      (&U_cyclefold_temp, &W_cyclefold_temp),
    )?;

    // CycleFold for second split of the witness
    let (cyclefold_nifs_2, (U_cyclefold, W_cyclefold)) = CycleFoldNIFS::<E>::prove(
      S_cyclefold,
      ck_cyclefold,
      ro_consts,
      U1.pre_committed.1,
      U2.pre_committed.1,
      rho,
      (&U_cyclefold_temp_1, &W_cyclefold_temp_1),
    )?;

    Ok((
      Self {
        partial_nifs,
        cyclefold_nifs,
        cyclefold_nifs_1,
        cyclefold_nifs_2,
      },
      (U, W),
      (U_cyclefold, W_cyclefold),
    ))
  }

  /// Verify a fold
  pub fn verify(
    &self,
    num_rounds: usize,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &LR1CSInstance<E>,
    U2: &SplitR1CSInstance<E>,
    U1_cyclefold: &RelaxedR1CSInstance<Dual<E>>,
  ) -> Result<(LR1CSInstance<E>, RelaxedR1CSInstance<Dual<E>>), NovaError> {
    // Verify partial folding proof
    let U = self
      .partial_nifs
      .verify(num_rounds, ro_consts, pp_digest, U1, U2)?;

    // Output folded instance.
    let U_cyclefold_temp = self.cyclefold_nifs.verify(ro_consts, U1_cyclefold)?;
    let U_cyclefold_temp_1 = self.cyclefold_nifs_1.verify(ro_consts, &U_cyclefold_temp)?;
    let U_cyclefold = self
      .cyclefold_nifs_2
      .verify(ro_consts, &U_cyclefold_temp_1)?;
    Ok((U, U_cyclefold))
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
/// The HyperNova NIFS w/o the CycleFold instances
pub struct PartialNIFS<E: CurveCycleEquipped> {
  pub(crate) sc: ROSumcheckProof<E>,
  pub(crate) sigmas: Vec<E::Scalar>,
  pub(crate) thetas: Vec<E::Scalar>,
}

impl<E> PartialNIFS<E>
where
  E: CurveCycleEquipped,
{
  /// HyperNova partial folding prover
  pub fn prove(
    S: &R1CSShape<E>,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    (U1, W1): (&LR1CSInstance<E>, &SplitR1CSWitness<E>),
    (U2, W2): (&SplitR1CSInstance<E>, &SplitR1CSWitness<E>),
  ) -> Result<(Self, (LR1CSInstance<E>, SplitR1CSWitness<E>), E::Scalar), NovaError> {
    // squeeze rho, gamma, beta
    let num_rounds = S.num_cons.next_power_of_two().log_2();
    let ((rho, gamma, beta), ro) = Self::challenges(ro_consts, pp_digest, U2, num_rounds);

    // Helper function for resizing polynomials
    let pad_poly = |mut vec: Vec<E::Scalar>| {
      vec.resize(S.num_cons.next_power_of_two(), E::Scalar::ZERO);
      vec
    };

    // --- Compute L_j's ---
    // where L_j = eq(rx, y) • H_j(y)
    let z1 = [W1.W().as_slice(), [U1.u].as_slice(), U1.X.as_slice()].concat();
    let mut poly_ABC = {
      let (evals_A, evals_B, evals_C) = S.multiply_vec(&z1)?;
      let evals_ABC = pad_poly(evals_A)
        .into_iter()
        .zip(pad_poly(evals_B))
        .zip(pad_poly(evals_C))
        .map(|((a, b), c)| a + gamma * b + gamma * gamma * c)
        .collect::<Vec<_>>();
      MultilinearPolynomial::new(evals_ABC)
    };
    let eq_eval_rx = EqPolynomial::evals_from_points(&U1.rx);
    let mut eq_rx = MultilinearPolynomial::new(eq_eval_rx);
    let L_comb_func = |abc: E::Scalar, eq: E::Scalar| -> E::Scalar { abc * eq };

    // Q = eq(beta, x) • G(x)
    let z2 = [
      W2.W().as_slice(),
      [E::Scalar::ONE].as_slice(),
      U2.aux.X.as_slice(),
    ]
    .concat();
    let eq_beta = EqPolynomial::new(beta.clone());
    let gamma_cubed = gamma * gamma * gamma;
    let mut poly_beta = MultilinearPolynomial::new(eq_beta.evals());
    let (mut poly_Az, mut poly_Bz, mut poly_Cz) = {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(&z2)?;
      (
        MultilinearPolynomial::new(pad_poly(poly_Az)),
        MultilinearPolynomial::new(pad_poly(poly_Bz)),
        MultilinearPolynomial::new(pad_poly(poly_Cz)),
      )
    };
    let Q_comb_func = |a: E::Scalar, b: E::Scalar, c: E::Scalar, eq: E::Scalar| -> E::Scalar {
      (a * b - c) * eq * gamma_cubed
    };

    // --- HyperNova's sum-check ---
    // The sum-check instance uses poseidon as the transcript, enabling the
    // implementation of the verifier circuit
    let comb_func =
      |L_abc: E::Scalar,
       L_eq: E::Scalar,
       Q_a: E::Scalar,
       Q_b: E::Scalar,
       Q_c: E::Scalar,
       Q_eq: E::Scalar|
       -> E::Scalar { L_comb_func(L_abc, L_eq) + Q_comb_func(Q_a, Q_b, Q_c, Q_eq) };
    let claim = U1.vs[0] + gamma * U1.vs[1] + gamma * gamma * U1.vs[2];
    let (sc, rx_p, _) = ROSumcheckProof::<E>::prove_cubic_hypernova(
      claim,
      num_rounds,
      &mut poly_ABC,
      &mut eq_rx,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      &mut poly_beta,
      comb_func,
      ro,
      ro_consts,
    )?;

    // Compute sigmas and thetas
    let claimed_vals = |z: &[E::Scalar]| -> Result<Vec<E::Scalar>, NovaError> {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(z)?;
      Ok(vec![
        MultilinearPolynomial::new(pad_poly(poly_Az)).evaluate(&rx_p),
        MultilinearPolynomial::new(pad_poly(poly_Bz)).evaluate(&rx_p),
        MultilinearPolynomial::new(pad_poly(poly_Cz)).evaluate(&rx_p),
      ])
    };
    let sigmas = claimed_vals(&z1)?;
    let thetas = claimed_vals(&z2)?;

    // Output the folded instance, witness pair
    let U = U1.fold(U2, rho, &rx_p, &sigmas, &thetas)?;
    let W = W1.fold(W2, rho)?;
    Ok((Self { sc, sigmas, thetas }, (U, W), rho))
  }

  /// HyperNova partial folding verifier
  pub fn verify(
    &self,
    num_rounds: usize,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &LR1CSInstance<E>,
    U2: &SplitR1CSInstance<E>,
  ) -> Result<LR1CSInstance<E>, NovaError> {
    // squeeze rho, gamma, beta
    let ((rho, gamma, beta), ro) = Self::challenges(ro_consts, pp_digest, U2, num_rounds);

    // Verify sum-check proof
    let claim = U1.vs[0] + gamma * U1.vs[1] + gamma * gamma * U1.vs[2];
    let (sub_claim, rx_p) = self.sc.verify(claim, num_rounds, 3, ro, ro_consts)?;

    // sub_claim = (σ_0 + γ•σ_1 + γ^2•σ_2) * eq(rx, rx_p) + (θ_0•θ_1 - θ_2) * eq(β, rx_p)
    let e1 = EqPolynomial::new(U1.rx.to_vec()).evaluate(&rx_p);
    let cl = (self.sigmas[0] + gamma * self.sigmas[1] + gamma * gamma * self.sigmas[2]) * e1;
    let e2 = EqPolynomial::new(beta).evaluate(&rx_p);
    let gamma_cubed = gamma * gamma * gamma;
    let cr = (self.thetas[0] * self.thetas[1] - self.thetas[2]) * e2 * gamma_cubed;
    if cl + cr != sub_claim {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // Output folded instance.
    let U = U1.fold(U2, rho, &rx_p, &self.sigmas, &self.thetas)?;
    Ok(U)
  }

  fn challenges(
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U2: &SplitR1CSInstance<E>,
    num_rounds: usize,
  ) -> (
    (E::Scalar, E::Scalar, Vec<E::Scalar>),
    <Dual<E> as Engine>::RO,
  ) {
    // squeeze rho, gamma, beta
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(*pp_digest);
    absorb_split_instance::<E>(U2, &mut ro);
    let rho = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(rho);
    let gamma = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(gamma);
    let beta = {
      ro.squeeze_vec(NUM_CHALLENGE_BITS, num_rounds)
        .iter()
        .map(|b| scalar_as_base::<Dual<E>>(*b))
        .collect::<Vec<_>>()
    };
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    for b in beta.iter() {
      ro.absorb(*b);
    }
    ((rho, gamma, beta), ro)
  }
}

/// CycleFold NIFS
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CycleFoldNIFS<E>
where
  E: CurveCycleEquipped,
{
  pub(crate) comm_T: Commitment<Dual<E>>,
  pub(crate) l_u: R1CSInstance<Dual<E>>,
}

impl<E> CycleFoldNIFS<E>
where
  E: CurveCycleEquipped,
{
  fn prove(
    S_cyclefold: &R1CSShape<Dual<E>>,
    ck_cyclefold: &CommitmentKey<Dual<E>>,
    ro_consts: &ROConstants<Dual<E>>,
    a: Commitment<E>,
    b: Commitment<E>,
    rho: E::Scalar,
    (U1_cyclefold, W1_cyclefold): (&RelaxedR1CSInstance<Dual<E>>, &RelaxedR1CSWitness<Dual<E>>),
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>),
    ),
    NovaError,
  > {
    // ECC gadgets for scalar multiplication require the scalar to decomposed into bits
    let rho_bits = rho
      .to_le_bits()
      .iter()
      .map(|b| Some(*b))
      .take(NUM_CHALLENGE_BITS)
      .collect::<Option<Vec<_>>>()
      .map(|v| v.try_into().unwrap());

    // Get the committed R1CS instance and witness from first CycleFold instance computing: comm_E1 + r · comm_T
    let (l_u_cyclefold, l_w_cyclefold) = {
      let mut cs_cyclefold = SatisfyingAssignment::<Dual<E>>::new();
      let circuit_cyclefold: CycleFoldCircuit<E> =
        CycleFoldCircuit::new(Some(a), Some(b), rho_bits);
      let _ = circuit_cyclefold.synthesize(&mut cs_cyclefold);
      cs_cyclefold
        .r1cs_instance_and_witness(S_cyclefold, ck_cyclefold)
        .map_err(|_| NovaError::UnSat)?
    };

    // Fold fresh EC instance witness pair into the running instance witness pair
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    U1_cyclefold.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(&l_u_cyclefold, &mut ro);
    let (T, comm_T) = S_cyclefold.commit_T(
      ck_cyclefold,
      U1_cyclefold,
      W1_cyclefold,
      &l_u_cyclefold,
      &l_w_cyclefold,
      &E::Base::ZERO,
    )?;
    comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_cyclefold = U1_cyclefold.fold(&l_u_cyclefold, &comm_T, &r);
    let W_cyclefold = W1_cyclefold.fold(&l_w_cyclefold, &T, &E::Base::ZERO, &r)?;

    // output nifs, & folded instance, witness pair
    Ok((
      Self {
        comm_T,
        l_u: l_u_cyclefold,
      },
      (U_cyclefold, W_cyclefold),
    ))
  }
  fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    U1_cyclefold: &RelaxedR1CSInstance<Dual<E>>,
  ) -> Result<RelaxedR1CSInstance<Dual<E>>, NovaError> {
    // Fold fresh EC instance witness pair into the running instance witness pair
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    U1_cyclefold.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(&self.l_u, &mut ro);
    self.comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_cyclefold = U1_cyclefold.fold(&self.l_u, &self.comm_T, &r);

    // output the folded instance
    Ok(U_cyclefold)
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    cyclefold::circuit::CycleFoldCircuit,
    frontend::{
      num::AllocatedNum,
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
      ConstraintSystem, SynthesisError,
    },
    hypernova::nifs::NIFS,
    provider::{Bn256EngineKZG, PallasEngine, Secp256k1Engine},
    r1cs::{
      split::{LR1CSInstance, SplitR1CSInstance, SplitR1CSWitness},
      R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness,
    },
    spartan::math::Math,
    traits::{snark::default_ck_hint, CurveCycleEquipped, Dual, Engine, ROConstants},
    CommitmentKey, R1CSWithArity,
  };
  use ff::{Field, PrimeField};
  use std::sync::Arc;

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
    let (shape, ck) = cs.r1cs_shape_and_key(&*default_ck_hint());
    let ro_consts = ROConstants::<Dual<E>>::default();

    let instance_witness = |x: E::Scalar| -> (SplitR1CSInstance<E>, SplitR1CSWitness<E>) {
      let mut cs = SatisfyingAssignment::<E>::new();
      let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(x));
      let (u, w) = cs.split_r1cs_instance_and_witness(&shape, &ck).unwrap();
      shape.is_sat_split(&ck, &u, &w).unwrap();
      (u, w)
    };

    // Now get the instance and assignment for one instance
    let (U1, W1) = instance_witness(E::Scalar::from(100));
    let (U2, W2) = instance_witness(E::Scalar::from(135));
    let (U3, W3) = instance_witness(E::Scalar::from(5));
    let (U4, W4) = instance_witness(E::Scalar::from(101));

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
    U1: &SplitR1CSInstance<E>,
    W1: &SplitR1CSWitness<E>,
    U2: &SplitR1CSInstance<E>,
    W2: &SplitR1CSWitness<E>,
    U3: &SplitR1CSInstance<E>,
    W3: &SplitR1CSWitness<E>,
    U4: &SplitR1CSInstance<E>,
    W4: &SplitR1CSWitness<E>,
  ) {
    // Get the structure for the CycleFold circuit and corresponding commitment key
    let mut cs: ShapeCS<Dual<E>> = ShapeCS::new();
    let circuit_cyclefold: CycleFoldCircuit<E> = CycleFoldCircuit::default();
    let _ = circuit_cyclefold.synthesize(&mut cs);
    let (r1cs_shape_cyclefold, ck_cyclefold) = cs.r1cs_shape_and_key(&*default_ck_hint());
    let ck_cyclefold = Arc::new(ck_cyclefold);
    let S_cyclefold = R1CSWithArity::new(r1cs_shape_cyclefold, 0);
    // Get the running CycleFold instance and witness pair
    let S_cyclefold = &S_cyclefold.r1cs_shape;
    let r_U_cyclefold = RelaxedR1CSInstance::default(&*ck_cyclefold, S_cyclefold);
    let r_W_cyclefold = RelaxedR1CSWitness::default(S_cyclefold);
    let s = S.num_cons.next_power_of_two().log_2();
    // produce a default running instance
    let mut r_W = SplitR1CSWitness::default(S);
    let mut r_U = LR1CSInstance::default(S);
    S.is_sat_linearized(ck, &r_U, &r_W).unwrap();

    let mut run_nifs = |u: &SplitR1CSInstance<E>, w: &SplitR1CSWitness<E>| {
      // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
      let (nifs, (_U, W), (_U_cyclefold, _W_cyclefold)) = NIFS::prove(
        (S, S_cyclefold),
        ck_cyclefold.as_ref(),
        ro_consts,
        pp_digest,
        (&r_U, &r_W),
        (u, w),
        (&r_U_cyclefold, &r_W_cyclefold),
      )
      .unwrap();

      // verify the step SNARK with U1 as the first incoming instance
      let (U, _U_cyclefold) = nifs
        .verify(s, ro_consts, pp_digest, &r_U, u, &r_U_cyclefold)
        .unwrap();

      assert_eq!(U, _U);

      // update the running witness and instance
      r_W = W;
      r_U = U;
      S.is_sat_linearized(ck, &r_U, &r_W).unwrap();
    };

    run_nifs(U1, W1);
    run_nifs(U2, W2);
    run_nifs(U3, W3);
    run_nifs(U4, W4);
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
