//! CycleFold for Nova
#![allow(clippy::upper_case_acronyms)]
use crate::bellpepper::r1cs::NovaWitness;
use crate::bellpepper::solver::SatisfyingAssignment;
use crate::constants::NUM_CHALLENGE_BITS;
use crate::cyclefold::circuit::CycleFoldCircuit;
use crate::cyclefold::util::{
  absorb_cyclefold_r1cs, absorb_primary_commitment, absorb_primary_r1cs,
};
use crate::gadgets::scalar_as_base;
use crate::traits::AbsorbInROTrait;
use crate::traits::{CurveCycleEquipped, ROTrait};
use crate::{
  constants::NUM_FE_IN_EMULATED_POINT,
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{Dual, Engine, ROConstants},
  Commitment, CommitmentKey,
};
use ff::PrimeFieldBits;

/// A SNARK for incremental computation
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
    let arity = U1.X.len();
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      1 + NUM_FE_IN_EMULATED_POINT + arity + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + u.X + T
    );
    ro.absorb(*pp_digest);
    absorb_primary_r1cs::<E, Dual<E>>(U2, &mut ro);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2)?;
    absorb_primary_commitment::<E, Dual<E>>(&comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let U = U1.fold(U2, &comm_T, &r);
    let W = W1.fold(W2, &T, &r)?;

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

    // Get the committed R1CS instance and witness from second CycleFold instance computing: comm_W1 + r· comm_W2
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
     * Fold first cyclefold instance
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
     * Fold second cyclefold instance
     */
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      45, // (3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U_secondary_temp.absorb_in_ro(&mut ro);
    absorb_cyclefold_r1cs(&l_u_cyclefold_W, &mut ro);
    let (T2, comm_T2) = S_secondary.commit_T(
      ck_secondary,
      &U_secondary_temp,
      &W_secondary_temp,
      &l_u_cyclefold_W,
      &l_w_cyclefold_W,
    )?;
    comm_T2.absorb_in_ro(&mut ro);
    let r2 = ro.squeeze(NUM_CHALLENGE_BITS);
    let U_secondary = U_secondary_temp.fold(&l_u_cyclefold_W, &comm_T2, &r2);
    let W_secondary = W_secondary_temp.fold(&l_w_cyclefold_W, &T2, &r2)?;

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
    let arity = U1.X.len();
    let mut ro = <Dual<E> as Engine>::RO::new(
      ro_consts.clone(),
      1 + NUM_FE_IN_EMULATED_POINT + arity + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + u.X + T
    );
    ro.absorb(*pp_digest);
    absorb_primary_r1cs::<E, Dual<E>>(U2, &mut ro);
    absorb_primary_commitment::<E, Dual<E>>(&self.comm_T, &mut ro);
    let r = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let U = U1.fold(U2, &self.comm_T, &r);

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
