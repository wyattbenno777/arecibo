//! A CycleFold influenced NIFS for folding IVC proofs.

use crate::bellpepper::r1cs::NovaWitness;
use crate::bellpepper::solver::SatisfyingAssignment;
use crate::constants::NUM_CHALLENGE_BITS;
use crate::cyclefold::circuit::CycleFoldCircuit;
use crate::cyclefold::util::{absorb_cyclefold_r1cs, absorb_primary_commitment};
use crate::gadgets::scalar_as_base;
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
use ff::PrimeFieldBits;

use super::utils::{absorb_U, scalar_to_bools, RelaxedFoldingData};

/// A non-interactive folding scheme for IVC proofs.
pub struct NIFS<E>
where
  E: CurveCycleEquipped,
{
  // proof from primary fold
  pub(super) comm_T: Commitment<E>,

  // proof from first cyclefold fold
  pub(super) comm_T1: Commitment<Dual<E>>,
  pub(super) l_u_cyclefold_E: R1CSInstance<Dual<E>>,

  // proof from second cyclefold fold
  pub(super) comm_T2: Commitment<Dual<E>>,
  pub(super) l_u_cyclefold_W: R1CSInstance<Dual<E>>,
}

impl<E> NIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prover algorithm for the NIFS used in folding IVC proofs. Implemented with CycleFold.
  pub fn prove(
    (ck, ck_secondary): (&CommitmentKey<E>, &CommitmentKey<Dual<E>>),
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    (S, S_secondary): (&R1CSShape<E>, &R1CSShape<Dual<E>>),
    (U1, W1): (&RelaxedR1CSInstance<E>, &RelaxedR1CSWitness<E>),
    (U2, W2): (&RelaxedR1CSInstance<E>, &RelaxedR1CSWitness<E>),
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
     * *********** Primary Fold ***********
     */
    let arity = U1.X.len();
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      1 + NUM_FE_IN_EMULATED_POINT + arity + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + u.X + T
    );
    ro.absorb(*pp_digest);
    absorb_U::<E>(U2, &mut ro);
    let (T, comm_T) = S.commit_T_relaxed(ck, U1, W1, U2, W2)?;
    absorb_primary_commitment::<E, Dual<E>>(&comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let r_squared = r.square();
    let U = U1.fold_relaxed(U2, &comm_T, &r);
    let W = W1.fold_relaxed(W2, &T, &r)?;

    // Get primary advice. i.e. the non-deterministic advice passed to the verifier circuit to do the primary fold in circuit
    let E_new_term_1 = U1.comm_E + comm_T * r;
    let E_new = U.comm_E;
    let W_new = U.comm_W;
    let data_p = RelaxedFoldingData::new(U1.clone(), U2.clone(), comm_T);

    /*
     * *********** CycleFold instances ***********
     */

    // The scalar multiplication gadget requires the scalar to be decomposed into bits
    let r_bools = scalar_to_bools::<E>(r);
    let r_squared_bools = scalar_to_bools::<E>(r_squared);

    // Get the committed R1CS instance and witness from first CycleFold instance.
    //
    // Computes: comm_E1 + r · comm_T
    let (l_u_cyclefold_E, l_w_cyclefold_E) = {
      let mut cs_cyclefold_E = SatisfyingAssignment::<Dual<E>>::with_capacity(
        S_secondary.num_io + 1,
        S_secondary.num_vars,
      );
      let circuit_cyclefold_E: CycleFoldCircuit<E> =
        CycleFoldCircuit::new(Some(U1.comm_E), Some(comm_T), r_bools);
      let _ = circuit_cyclefold_E.synthesize(&mut cs_cyclefold_E);
      cs_cyclefold_E
        .r1cs_instance_and_witness(S_secondary, ck_secondary)
        .map_err(|_| NovaError::UnSat)?
    };

    // Get the committed R1CS instance and witness from second CycleFold instance
    //
    // let term1 = comm_E1 + r · comm_T;
    //  Computes:
    //
    // first_term + r^2 • comm_E2
    let (l_u_cyclefold_E2, l_w_cyclefold_E2) = {
      let mut cs_cyclefold_E = SatisfyingAssignment::<Dual<E>>::with_capacity(
        S_secondary.num_io + 1,
        S_secondary.num_vars,
      );
      let circuit_cyclefold_E: CycleFoldCircuit<E> =
        CycleFoldCircuit::new(Some(E_new_term_1), Some(U2.comm_E), r_squared_bools);
      let _ = circuit_cyclefold_E.synthesize(&mut cs_cyclefold_E);
      cs_cyclefold_E
        .r1cs_instance_and_witness(S_secondary, ck_secondary)
        .map_err(|_| NovaError::UnSat)?
    };

    // Get the committed R1CS instance and witness from third CycleFold instance.
    //
    // Computes: comm_W1 + r· comm_W2
    let (l_u_cyclefold_W, l_w_cyclefold_W) = {
      let mut cs_cyclefold_W = SatisfyingAssignment::<Dual<E>>::with_capacity(
        S_secondary.num_io + 1,
        S_secondary.num_vars,
      );
      let circuit_cyclefold_W: CycleFoldCircuit<E> =
        CycleFoldCircuit::new(Some(U1.comm_W), Some(U2.comm_W), r_bools);
      let _ = circuit_cyclefold_W.synthesize(&mut cs_cyclefold_W);
      cs_cyclefold_W
        .r1cs_instance_and_witness(S_secondary, ck_secondary)
        .map_err(|_| NovaError::UnSat)?
    };

    /*
     * *********** Fold first cyclefold instance ***********
     */
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U1_secondary.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(&l_u_cyclefold_E, &mut ro);
    let (T1, comm_T1) = S_secondary.commit_T(
      ck_secondary,
      U1_secondary,
      W1_secondary,
      &l_u_cyclefold_E,
      &l_w_cyclefold_E,
    )?;
    comm_T1.absorb_in_ro(&mut ro);
    let r1 = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_secondary_temp = U1_secondary.fold(&l_u_cyclefold_E, &comm_T1, &r1);
    let W_secondary_temp = W1_secondary.fold(&l_w_cyclefold_E, &T1, &r1)?;

    /*
     * *********** Fold second cyclefold instance ***********
     */
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U_secondary_temp.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(&l_u_cyclefold_E2, &mut ro);
    let (T2, comm_T2) = S_secondary.commit_T(
      ck_secondary,
      &U_secondary_temp,
      &W_secondary_temp,
      &l_u_cyclefold_E2,
      &l_w_cyclefold_E2,
    )?;
    comm_T2.absorb_in_ro(&mut ro);
    let r2 = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_secondary_temp_1 = U_secondary_temp.fold(&l_u_cyclefold_W, &comm_T2, &r2);
    let W_secondary_temp_1 = W_secondary_temp.fold(&l_w_cyclefold_W, &T2, &r2)?;

    /*
     * *********** Fold third cyclefold instance ***********
     */
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U_secondary_temp_1.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(&l_u_cyclefold_W, &mut ro);
    let (T3, comm_T3) = S_secondary.commit_T(
      ck_secondary,
      &U_secondary_temp_1,
      &W_secondary_temp_1,
      &l_u_cyclefold_W,
      &l_w_cyclefold_W,
    )?;
    comm_T3.absorb_in_ro(&mut ro);
    let r3 = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_secondary = U_secondary_temp_1.fold(&l_u_cyclefold_W, &comm_T3, &r3);
    let W_secondary = W_secondary_temp_1.fold(&l_w_cyclefold_W, &T3, &r3)?;

    // The Nova-CycleFold NIFS proof
    let nifs = Self {
      comm_T,
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

  /// Verifier algorithm for the NIFS used in folding IVC proofs
  pub fn verify(&self) {}
}
