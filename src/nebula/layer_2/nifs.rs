//! A CycleFold influenced NIFS for folding IVC proofs.

use crate::bellpepper::r1cs::NovaWitness;
use crate::bellpepper::solver::SatisfyingAssignment;
use crate::constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS};
use crate::cyclefold::circuit::CycleFoldCircuit;
use crate::cyclefold::util::{absorb_cyclefold_r1cs, absorb_primary_commitment};
use crate::gadgets::scalar_as_base;
use crate::r1cs::R1CSWitness;
use crate::traits::AbsorbInROTrait;
use crate::traits::{CurveCycleEquipped, ROTrait};
use crate::{
  constants::NUM_FE_IN_EMULATED_POINT,
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{Dual, Engine, ROConstants},
  Commitment, CommitmentKey,
};
use ff::Field;
use serde::{Deserialize, Serialize};

use super::utils::{absorb_U, scalar_to_bools};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
/// A non-interactive folding scheme for IVC proofs.
pub struct NIFS<E>
where
  E: CurveCycleEquipped,
{
  // proof from primary fold
  pub(super) nifs_primary: PrimaryRelaxedNIFS<E>,

  // proof from first cyclefold fold
  pub(super) nifs_E1: CycleFoldNIFS<E>,
  pub(super) l_u_cyclefold_E1: R1CSInstance<Dual<E>>,

  // proof from second cyclefold fold
  pub(super) nifs_E2: CycleFoldNIFS<E>,
  pub(super) l_u_cyclefold_E2: R1CSInstance<Dual<E>>,

  // proof from third cyclefold fold
  pub(super) nifs_W: CycleFoldNIFS<E>,
  pub(super) l_u_cyclefold_W: R1CSInstance<Dual<E>>,

  // proof from fourth cyclefold fold
  pub(super) nifs_final_cyclefold: CycleFoldRelaxedNIFS<E>,
}

impl<E> NIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prover algorithm for the NIFS used in folding IVC proofs. Implemented with CycleFold.
  #[tracing::instrument(skip_all, name = "Fold Recursive SNARK", level = "debug")]
  pub fn prove(
    (ck, ck_secondary): (&CommitmentKey<E>, &CommitmentKey<Dual<E>>),
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    (S, S_secondary): (&R1CSShape<E>, &R1CSShape<Dual<E>>),
    (U1, W1): (&RelaxedR1CSInstance<E>, &RelaxedR1CSWitness<E>),
    (U2, W2): (&RelaxedR1CSInstance<E>, &RelaxedR1CSWitness<E>),
    (U1_secondary, W1_secondary): (&RelaxedR1CSInstance<Dual<E>>, &RelaxedR1CSWitness<Dual<E>>),
    (U2_secondary, W2_secondary): (&RelaxedR1CSInstance<Dual<E>>, &RelaxedR1CSWitness<Dual<E>>),
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>),
      (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>),
    ),
    NovaError,
  > {
    /*
     * *********** Primary Fold ***********
     */
    let (nifs_primary, (U, W), r) =
      PrimaryRelaxedNIFS::prove(ck, ro_consts, pp_digest, S, (U1, W1), (U2, W2))?;

    /*
     * *********** CycleFold instances ***********
     */

    // The scalar multiplication gadget requires the scalar to be decomposed into bits
    let r_bools = scalar_to_bools::<E>(r);
    let r_squared_bools = scalar_to_bools::<E>(r.square());

    // Get the committed R1CS instance and witness from first CycleFold instance.
    //
    // Computes: comm_E1 + r · comm_T
    let (l_u_cyclefold_E1, l_w_cyclefold_E1) = compute_cyclefold_instance_witness_pair::<E>(
      S_secondary,
      ck_secondary,
      U1.comm_E,
      nifs_primary.comm_T,
      r_bools,
    )?;

    // Get the committed R1CS instance and witness from second CycleFold instance
    //
    // let term1 = comm_E1 + r · comm_T;
    //  Computes:
    //
    // term1 + r^2 • comm_E2
    let E_term_1 = U1.comm_E + nifs_primary.comm_T * r;
    let (l_u_cyclefold_E2, l_w_cyclefold_E2) = compute_cyclefold_instance_witness_pair::<E>(
      S_secondary,
      ck_secondary,
      E_term_1,
      U2.comm_E,
      r_squared_bools,
    )?;

    // Get the committed R1CS instance and witness from third CycleFold instance.
    //
    // Computes: comm_W1 + r· comm_W2
    let (l_u_cyclefold_W, l_w_cyclefold_W) = compute_cyclefold_instance_witness_pair::<E>(
      S_secondary,
      ck_secondary,
      U1.comm_W,
      U2.comm_W,
      r_bools,
    )?;

    /*
     * *********** Fold first cyclefold instance ***********
     */
    let (nifs_E1, (U_secondary_temp, W_secondary_temp), _) = CycleFoldNIFS::<E>::prove(
      ck_secondary,
      ro_consts,
      S_secondary,
      U1_secondary,
      W1_secondary,
      &l_u_cyclefold_E1,
      &l_w_cyclefold_E1,
    )?;

    /*
     * *********** Fold second cyclefold instance ***********
     */
    let (nifs_E2, (U_secondary_temp_1, W_secondary_temp_1), _) = CycleFoldNIFS::<E>::prove(
      ck_secondary,
      ro_consts,
      S_secondary,
      &U_secondary_temp,
      &W_secondary_temp,
      &l_u_cyclefold_E2,
      &l_w_cyclefold_E2,
    )?;

    /*
     * *********** Fold third cyclefold instance ***********
     */
    let (nifs_W, (U_secondary_temp_2, W_secondary_temp_2), _) = CycleFoldNIFS::<E>::prove(
      ck_secondary,
      ro_consts,
      S_secondary,
      &U_secondary_temp_1,
      &W_secondary_temp_1,
      &l_u_cyclefold_W,
      &l_w_cyclefold_W,
    )?;

    /*
     * *********** Fold fourth cyclefold instance ***********
     */
    let (nifs_final_cyclefold, (U_secondary, W_secondary), _) = CycleFoldRelaxedNIFS::<E>::prove(
      ck_secondary,
      ro_consts,
      S_secondary,
      &U_secondary_temp_2,
      &W_secondary_temp_2,
      U2_secondary,
      W2_secondary,
    )?;

    Ok((
      Self {
        nifs_primary,
        nifs_E1,
        l_u_cyclefold_E1,
        nifs_E2,
        l_u_cyclefold_E2,
        nifs_W,
        l_u_cyclefold_W,
        nifs_final_cyclefold,
      },
      (U, W),
      (U_secondary, W_secondary),
    ))
  }

  /// Verifier algorithm for the NIFS used in folding IVC proofs
  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &RelaxedR1CSInstance<E>,
    U2: &RelaxedR1CSInstance<E>,
    U1_secondary: &RelaxedR1CSInstance<Dual<E>>,
    U2_secondary: &RelaxedR1CSInstance<Dual<E>>,
  ) -> Result<(RelaxedR1CSInstance<E>, RelaxedR1CSInstance<Dual<E>>), NovaError> {
    /*
     * *********** Primary Fold ***********
     */
    let U = self.nifs_primary.verify(ro_consts, pp_digest, U1, U2)?;

    /*
     * *********** Fold first cyclefold instance ***********
     */
    let U_secondary_temp = self
      .nifs_E1
      .verify(ro_consts, U1_secondary, &self.l_u_cyclefold_E1)?;

    /*
     * *********** Fold second cyclefold instance ***********
     */
    let U_secondary_temp_1 =
      self
        .nifs_E2
        .verify(ro_consts, &U_secondary_temp, &self.l_u_cyclefold_E2)?;

    /*
     * *********** Fold third cyclefold instance ***********
     */
    let U_secondary_temp_2 =
      self
        .nifs_W
        .verify(ro_consts, &U_secondary_temp_1, &self.l_u_cyclefold_W)?;

    /*
     * *********** Fold fourth cyclefold instance ***********
     */
    let U_secondary =
      self
        .nifs_final_cyclefold
        .verify(ro_consts, &U_secondary_temp_2, U2_secondary)?;

    Ok((U, U_secondary))
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
    let (T, comm_T) = S.commit_T_relaxed(ck, U1, W1, U2, W2)?;
    absorb_primary_commitment::<E, Dual<E>>(&comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let U = U1.fold_relaxed(U2, &comm_T, &r);
    let W = W1.fold_relaxed(W2, &T, &r)?;
    Ok((Self { comm_T }, (U, W), r))
  }

  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &RelaxedR1CSInstance<E>,
    U2: &RelaxedR1CSInstance<E>,
  ) -> Result<RelaxedR1CSInstance<E>, NovaError> {
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      1 + (2 * NUM_FE_IN_EMULATED_POINT + 2 + 1) + NUM_FE_IN_EMULATED_POINT, // pp_digest + (U.W + U.comm_E + U.X + U.u) + comm_T
    );
    ro.absorb(*pp_digest);
    absorb_U::<E>(U2, &mut ro);
    absorb_primary_commitment::<E, Dual<E>>(&self.comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let U = U1.fold_relaxed(U2, &self.comm_T, &r);

    Ok(U)
  }
}

/// NIFS for folding CycleFold [`R1CSInstance`] and [`R1CSWitness`] instances into the running instance
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CycleFoldNIFS<E>
where
  E: CurveCycleEquipped,
{
  pub(crate) comm_T: Commitment<Dual<E>>,
}

impl<E> CycleFoldNIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prover algorithm for folding incoming CycleFold [`R1CSInstance`] and [`R1CSWitness`] instances into running instance
  #[tracing::instrument(skip_all, name = "CycleFoldNIFS::prove", level = "debug")]
  pub fn prove(
    ck: &CommitmentKey<Dual<E>>,
    ro_consts: &ROConstants<Dual<E>>,
    S: &R1CSShape<Dual<E>>,
    U1: &RelaxedR1CSInstance<Dual<E>>,
    W1: &RelaxedR1CSWitness<Dual<E>>,
    U2: &R1CSInstance<Dual<E>>,
    W2: &R1CSWitness<Dual<E>>,
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
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U1.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(U2, &mut ro);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2)?;
    comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    let U = U1.fold(U2, &comm_T, &r);
    let W = W1.fold(W2, &T, &r)?;
    Ok((Self { comm_T }, (U, W), r))
  }

  /// Verifier algorithm for folding incoming CycleFold [`R1CSInstance`] and [`R1CSWitness`] instances into running instance
  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    U1: &RelaxedR1CSInstance<Dual<E>>,
    U2: &R1CSInstance<Dual<E>>,
  ) -> Result<RelaxedR1CSInstance<Dual<E>>, NovaError> {
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U1.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(U2, &mut ro);
    self.comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    let U = U1.fold(U2, &self.comm_T, &r);
    Ok(U)
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
      2 * (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (U) + T
    );
    U1.absorb_in_ro(&mut ro);
    U2.absorb_in_ro(&mut ro);
    let (T, comm_T) = S.commit_T_relaxed(ck, U1, W1, U2, W2)?;
    comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    let U = U1.fold_relaxed(U2, &comm_T, &r);
    let W = W1.fold_relaxed(W2, &T, &r)?;
    Ok((Self { comm_T }, (U, W), r))
  }

  /// Verifier algorithm for folding two CycleFold [`RelaxedR1CSInstance`] and [`RelaxedR1CSWitness`] instances
  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E>>,
    U1: &RelaxedR1CSInstance<Dual<E>>,
    U2: &RelaxedR1CSInstance<Dual<E>>,
  ) -> Result<RelaxedR1CSInstance<Dual<E>>, NovaError> {
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      2 * (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (U) + T
    );
    U1.absorb_in_ro(&mut ro);
    U2.absorb_in_ro(&mut ro);
    self.comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    let U = U1.fold_relaxed(U2, &self.comm_T, &r);
    Ok(U)
  }
}

/// Computes the R1CS instance and witness for the CycleFold Circuit
pub fn compute_cyclefold_instance_witness_pair<E>(
  S_cyclefold: &R1CSShape<Dual<E>>,
  ck_cyclefold: &CommitmentKey<Dual<E>>,
  commit_1: Commitment<E>,
  commit_2: Commitment<E>,
  scalar: Option<[bool; NUM_CHALLENGE_BITS]>,
) -> Result<(R1CSInstance<Dual<E>>, R1CSWitness<Dual<E>>), NovaError>
where
  E: CurveCycleEquipped,
{
  let mut cs_cyclefold_W =
    SatisfyingAssignment::<Dual<E>>::with_capacity(S_cyclefold.num_io + 1, S_cyclefold.num_vars);
  let circuit_cyclefold_W: CycleFoldCircuit<E> =
    CycleFoldCircuit::new(Some(commit_1), Some(commit_2), scalar);
  let _ = circuit_cyclefold_W.synthesize(&mut cs_cyclefold_W);
  cs_cyclefold_W
    .r1cs_instance_and_witness(S_cyclefold, ck_cyclefold)
    .map_err(|_| NovaError::UnSat)
}
