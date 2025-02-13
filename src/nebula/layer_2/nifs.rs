//! A CycleFold influenced NIFS for folding IVC proofs.

use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS},
  cyclefold::{circuit::CycleFoldCircuit, util::absorb_cyclefold_r1cs},
  errors::NovaError,
  frontend::{r1cs::NovaWitness, solver::SatisfyingAssignment, ConstraintSystem},
  nebula::nifs::{CycleFoldRelaxedNIFS, PrimaryRelaxedNIFS},
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROConstants, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use serde::{Deserialize, Serialize};

use super::utils::{absorb_U_bn, scalar_to_bools};

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
      (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    absorb_U_bn(U1, &mut ro);
    absorb_cyclefold_r1cs(U2, &mut ro);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2, &<Dual<E> as Engine>::Scalar::ZERO)?;
    comm_T.absorb_in_ro(&mut ro);
    let r = ro.squeeze(NUM_CHALLENGE_BITS);
    let U = U1.fold(U2, &comm_T, &r);
    let W = W1.fold(W2, &T, &<Dual<E> as Engine>::Scalar::ZERO, &r)?;
    Ok((Self { comm_T }, (U, W), r))
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
  let mut cs_cyclefold_W = SatisfyingAssignment::<Dual<E>>::new();
  let circuit_cyclefold_W: CycleFoldCircuit<E> =
    CycleFoldCircuit::new(Some(commit_1), Some(commit_2), scalar);
  let _ = circuit_cyclefold_W.synthesize(&mut cs_cyclefold_W);
  cs_cyclefold_W
    .r1cs_instance_and_witness(S_cyclefold, ck_cyclefold)
    .map_err(|_| NovaError::UnSat)
}
