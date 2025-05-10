//! CycleFold for Nova
use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS, NUM_FE_IN_EMULATED_POINT},
  cyclefold::{
    circuit::CycleFoldCircuit,
    util::{absorb_cyclefold_r1cs, absorb_primary_commitment, absorb_primary_r1cs},
  },
  errors::NovaError,
  frontend::{r1cs::NovaWitness, solver::SatisfyingAssignment, ConstraintSystem},
  gadgets::scalar_as_base,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROConstants, ROTrait},
  Commitment, CommitmentKey,
};
use ff::{Field, PrimeFieldBits};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

use super::layer_2::utils::{absorb_U, absorb_U_bn};

/// A SNARK for incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E>
where
  E: CurveCycleEquipped,
{
  // proof from primary fold
  pub(crate) nifs_primary: PrimaryNIFS<E>,

  // proof from first cyclefold fold
  pub(crate) comm_T1: Commitment<Dual<E>>,
  pub(crate) l_u_cyclefold_E: R1CSInstance<Dual<E>>,

  // proof from second cyclefold fold
  pub(crate) comm_T2: Commitment<Dual<E>>,
  pub(crate) l_u_cyclefold_W: R1CSInstance<Dual<E>>,
}

impl<E> NIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Produces a default `NIFS`
  pub fn default(S_secondary: &R1CSShape<Dual<E>>) -> Self {
    Self {
      nifs_primary: PrimaryNIFS::<E>::default(),
      comm_T1: Commitment::<Dual<E>>::default(),
      l_u_cyclefold_E: R1CSInstance::<Dual<E>>::default(S_secondary),
      comm_T2: Commitment::<Dual<E>>::default(),
      l_u_cyclefold_W: R1CSInstance::<Dual<E>>::default(S_secondary),
    }
}
  /// Prover algorithm for: CycleFold folding scheme applied to Nova
  pub fn prove(
    (ck, ck_secondary): (&CommitmentKey<E>, &CommitmentKey<Dual<E>>),
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    (S, S_secondary): (&R1CSShape<E>, &R1CSShape<Dual<E>>),
    (U1, W1): (&RelaxedR1CSInstance<E>, &RelaxedR1CSWitness<E>),
    (U2, W2): (&R1CSInstance<E>, &R1CSWitness<E>),
    (U1_secondary, W1_secondary): (&RelaxedR1CSInstance<Dual<E>>, &RelaxedR1CSWitness<Dual<E>>),
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>),
      (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>),
      E::Scalar,
      // Advice
      RelaxedR1CSInstance<Dual<E>>,
    ),
    NovaError,
  > {
    /*
     * Primary Fold
     */
    let (nifs_primary, (U, W), r) =
      PrimaryNIFS::prove(ck, ro_consts, pp_digest, S, (U1, W1), (U2, W2))?;

    /*
     * CycleFold instances
     */

    // ECC gadgets for scalar multiplication require the scalar to decomposed into bits
    let r_bools = r
      .to_le_bits()
      .iter()
      .map(|b| Some(*b))
      .take(NUM_CHALLENGE_BITS)
      .collect::<Option<Vec<_>>>()
      .map(|v| v.try_into().unwrap());

    // Get the committed R1CS instance and witness from first CycleFold instance computing: comm_E1 + r · comm_T
    let (l_u_cyclefold_E, l_w_cyclefold_E) = {
      let mut cs_cyclefold_E = SatisfyingAssignment::<Dual<E>>::new();
      let circuit_cyclefold_E: CycleFoldCircuit<E> =
        CycleFoldCircuit::new(Some(U1.comm_E), Some(nifs_primary.comm_T), r_bools);
      let _ = circuit_cyclefold_E.synthesize(&mut cs_cyclefold_E);
      cs_cyclefold_E
        .r1cs_instance_and_witness(S_secondary, ck_secondary)
        .map_err(|_| NovaError::UnSat)?
    };

    // Get the committed R1CS instance and witness from second CycleFold instance computing: comm_W1 + r· comm_W2
    let (l_u_cyclefold_W, l_w_cyclefold_W) = {
      let mut cs_cyclefold_W = SatisfyingAssignment::<Dual<E>>::new();
      let circuit_cyclefold_W: CycleFoldCircuit<E> =
        CycleFoldCircuit::new(Some(U1.comm_W), Some(U2.comm_W), r_bools);
      let _ = circuit_cyclefold_W.synthesize(&mut cs_cyclefold_W);
      cs_cyclefold_W
        .r1cs_instance_and_witness(S_secondary, ck_secondary)
        .map_err(|_| NovaError::UnSat)?
    };

    /*
     * Fold first cyclefold instance
     */
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U1_secondary.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(&l_u_cyclefold_E, &mut ro);
    let r_T1 = <Dual<E> as Engine>::Scalar::random(&mut OsRng);
    let (T1, comm_T1) = S_secondary.commit_T(
      ck_secondary,
      U1_secondary,
      W1_secondary,
      &l_u_cyclefold_E,
      &l_w_cyclefold_E,
      &r_T1,
    )?;
    comm_T1.absorb_in_ro(&mut ro);
    let r1 = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_secondary_temp = U1_secondary.fold(&l_u_cyclefold_E, &comm_T1, &r1);
    let W_secondary_temp = W1_secondary.fold(&l_w_cyclefold_E, &T1, &r_T1, &r1)?;

    /*
     * Fold second cyclefold instance
     */
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U_secondary_temp.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(&l_u_cyclefold_W, &mut ro);
    let r_T2 = <Dual<E> as Engine>::Scalar::random(&mut OsRng);
    let (T2, comm_T2) = S_secondary.commit_T(
      ck_secondary,
      &U_secondary_temp,
      &W_secondary_temp,
      &l_u_cyclefold_W,
      &l_w_cyclefold_W,
      &r_T2,
    )?;
    comm_T2.absorb_in_ro(&mut ro);
    let r2 = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_secondary = U_secondary_temp.fold(&l_u_cyclefold_W, &comm_T2, &r2);
    let W_secondary = W_secondary_temp.fold(&l_w_cyclefold_W, &T2, &r_T2, &r2)?;

    // The Nova-CycleFold NIFS proof
    let nifs = Self {
      nifs_primary,
      comm_T1,
      l_u_cyclefold_E,
      comm_T2,
      l_u_cyclefold_W,
    };
    Ok((
      nifs,
      (U, W),
      (U_secondary, W_secondary),
      r,
      U_secondary_temp,
    ))
  }

  #[allow(dead_code)] // Code kept here for educational purposes
  /// Verifier algorithm for: CycleFold folding scheme applied to Nova
  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &RelaxedR1CSInstance<E>,
    U2: &R1CSInstance<E>,
    U1_secondary: &RelaxedR1CSInstance<Dual<E>>,
  ) -> Result<(RelaxedR1CSInstance<E>, RelaxedR1CSInstance<Dual<E>>), NovaError> {
    /*
     * Primary fold
     */
    let U = self.nifs_primary.verify(ro_consts, pp_digest, U1, U2);

    /*
     * First CycleFold fold
     */
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U1_secondary.absorb_in_ro(&mut ro);
    self.l_u_cyclefold_E.absorb_in_ro(&mut ro);
    self.comm_T1.absorb_in_ro(&mut ro);
    let r1 = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_secondary_temp = U1_secondary.fold(&self.l_u_cyclefold_E, &self.comm_T1, &r1);

    /*
     * Second CycleFold fold
     */
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U_secondary_temp.absorb_in_ro(&mut ro);
    self.l_u_cyclefold_W.absorb_in_ro(&mut ro);
    self.comm_T2.absorb_in_ro(&mut ro);
    let r2 = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_secondary = U_secondary_temp.fold(&self.l_u_cyclefold_W, &self.comm_T2, &r2);

    Ok((U, U_secondary))
  }
}

/// NIFS for primary IVC proof
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PrimaryNIFS<E>
where
  E: CurveCycleEquipped,
{
  pub(crate) comm_T: Commitment<E>,
}

impl<E> PrimaryNIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prover implementation for NIFS
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    S: &R1CSShape<E>,
    (U1, W1): (&RelaxedR1CSInstance<E>, &RelaxedR1CSWitness<E>),
    (U2, W2): (&R1CSInstance<E>, &R1CSWitness<E>),
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>),
      E::Scalar,
    ),
    NovaError,
  > {
    let arity = U1.X.len();
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      1 + NUM_FE_IN_EMULATED_POINT + arity + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + u.X + T
    );
    ro.absorb(*pp_digest);
    absorb_primary_r1cs::<E, Dual<E>>(U2, &mut ro);
    let r_T = E::Scalar::random(&mut OsRng);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2, &r_T)?;
    absorb_primary_commitment::<E, Dual<E>>(&comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let U = U1.fold(U2, &comm_T, &r);
    let W = W1.fold(W2, &T, &r_T, &r)?;
    Ok((Self { comm_T }, (U, W), r))
  }

  /// Verifier implementation for NIFS
  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &RelaxedR1CSInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> RelaxedR1CSInstance<E> {
    let arity = U1.X.len();
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      1 + NUM_FE_IN_EMULATED_POINT + arity + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + u.X + T
    );
    ro.absorb(*pp_digest);
    absorb_primary_r1cs::<E, Dual<E>>(U2, &mut ro);
    absorb_primary_commitment::<E, Dual<E>>(&self.comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    U1.fold(U2, &self.comm_T, &r)
  }
}

impl<E> Default for PrimaryNIFS<E>
where
  E: CurveCycleEquipped,
{
  fn default() -> Self {
    Self {
      comm_T: Commitment::<E>::default(),
    }
  }
}
/// NIFS for folding the primary relaxed r1cs instance and witness
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PrimaryRelaxedNIFS<E1>
where
  E1: CurveCycleEquipped,
{
  pub(crate) comm_T: Commitment<E1>,
}

impl<E> PrimaryRelaxedNIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prove the primary relaxed NIFS
  #[tracing::instrument(skip_all, name = "PrimaryRelaxedNIFS::prove", level = "debug")]
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    S: &R1CSShape<E>,
    (U1, W1): (&RelaxedR1CSInstance<E>, &RelaxedR1CSWitness<E>),
    (U2, W2): (&RelaxedR1CSInstance<E>, &RelaxedR1CSWitness<E>),
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>),
      E::Scalar,
    ),
    NovaError,
  > {
    let arity = U1.X.len();
    if arity != U2.X.len() {
      return Err(NovaError::InvalidInputLength);
    }
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      1 + (2 * NUM_FE_IN_EMULATED_POINT + arity + 1) + NUM_FE_IN_EMULATED_POINT, // pp_digest + (U.comm_W + U.comm_E + U.X + U.u) + comm_T
    );
    ro.absorb(*pp_digest);
    absorb_U::<E>(U2, &mut ro);
    let (T, comm_T) = S.commit_T_relaxed(ck, U1, W1, U2, W2, &E::Scalar::ZERO)?;
    absorb_primary_commitment::<E, Dual<E>>(&comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let U = U1.fold_relaxed(U2, &comm_T, &r);
    let W = W1.fold_relaxed(W2, &T, &E::Scalar::ZERO, &r)?;
    Ok((Self { comm_T }, (U, W), r))
  }

  /// Verify the primary relaxed NIFS
  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &RelaxedR1CSInstance<E>,
    U2: &RelaxedR1CSInstance<E>,
  ) -> RelaxedR1CSInstance<E> {
    let arity = U1.X.len();
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      1 + (2 * NUM_FE_IN_EMULATED_POINT + arity + 1) + NUM_FE_IN_EMULATED_POINT, // pp_digest + (U.comm_W + U.comm_E + U.X + U.u) + comm_T
    );
    ro.absorb(*pp_digest);
    absorb_U::<E>(U2, &mut ro);
    absorb_primary_commitment::<E, Dual<E>>(&self.comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    U1.fold_relaxed(U2, &self.comm_T, &r)
  }
}

/// NIFS for folding two Cyclefold [`RelaxedR1CSInstance`] and [`RelaxedR1CSWitness`] instances
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CycleFoldRelaxedNIFS<E>
where
  E: CurveCycleEquipped,
{
  pub(crate) comm_T: Commitment<Dual<E>>,
}

impl<E> CycleFoldRelaxedNIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prover algorithm for folding two CycleFold [`RelaxedR1CSInstance`] and [`RelaxedR1CSWitness`] instances
  #[tracing::instrument(skip_all, name = "CycleFoldRelaxedNIFS::prove", level = "debug")]
  pub fn prove(
    ck: &CommitmentKey<Dual<E>>,
    ro_consts: &ROConstants<Dual<E>>,
    S: &R1CSShape<Dual<E>>,
    U1: &RelaxedR1CSInstance<Dual<E>>,
    W1: &RelaxedR1CSWitness<Dual<E>>,
    U2: &RelaxedR1CSInstance<Dual<E>>,
    W2: &RelaxedR1CSWitness<Dual<E>>,
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>),
      <Dual<E> as Engine>::Scalar,
    ),
    NovaError,
  > {
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      2 * (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (U) + T
    );
    absorb_U_bn(U1, &mut ro);
    absorb_U_bn(U2, &mut ro);
    let (T, comm_T) = S.commit_T_relaxed(ck, U1, W1, U2, W2, &<Dual<E> as Engine>::Scalar::ZERO)?;
    comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    let U = U1.fold_relaxed(U2, &comm_T, &r);
    let W = W1.fold_relaxed(W2, &T, &<Dual<E> as Engine>::Scalar::ZERO, &r)?;
    Ok((Self { comm_T }, (U, W), r))
  }

  /// Verifier protocol
  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    U1: &RelaxedR1CSInstance<Dual<E>>,
    U2: &RelaxedR1CSInstance<Dual<E>>,
  ) -> RelaxedR1CSInstance<Dual<E>> {
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      2 * (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (U) + T
    );
    absorb_U_bn(U1, &mut ro);
    absorb_U_bn(U2, &mut ro);
    self.comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    U1.fold_relaxed(U2, &self.comm_T, &r)
  }
}
