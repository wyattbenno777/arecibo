//! This module contains an incremental commitment scheme implementation.

use crate::{
  constants::{DEFAULT_ABSORBS, NUM_HASH_BITS},
  cyclefold::util::absorb_primary_commitment,
  gadgets::scalar_as_base,
  hypernova::rs::IncrementalCommitment,
  traits::{
    commitment::CommitmentEngineTrait, CurveCycleEquipped, Dual, Engine, ROConstants, ROTrait,
  },
  Commitment, CommitmentKey,
};
use ff::Field;

/// Produces two incremental commitments to non-deterministic advice ω
///
/// * commits to advice with Pedersen
/// * hashes previous commitment & pedersen commitment to advice
/// * outputs hash bits as scalar
pub fn increment_ic<E>(
  ck: &CommitmentKey<E>,
  ro_consts: &ROConstants<Dual<E>>,
  prev_ic: IncrementalCommitment<E>,
  advice: (&[E::Scalar], &[E::Scalar]),
) -> (E::Scalar, E::Scalar)
where
  E: CurveCycleEquipped,
{
  // TODO: add blind.
  //       We have not added blinding yet because for sharding we need the incremental comms to be deterministic
  let comm_advice_0 = E::CE::commit(ck, advice.0, &E::Scalar::ZERO);
  let comm_advice_1 = E::CE::commit_at(ck, advice.1, &E::Scalar::ZERO, advice.0.len());
  increment_comm::<E>(ro_consts, prev_ic, (comm_advice_0, comm_advice_1))
}

/// Produce two incremental commitment to already pedersen committed non-deterministic advice ω
pub(crate) fn increment_comm<E>(
  ro_consts: &ROConstants<Dual<E>>,
  prev_ic: IncrementalCommitment<E>,
  comm_advice: (Commitment<E>, Commitment<E>),
) -> (E::Scalar, E::Scalar)
where
  E: CurveCycleEquipped,
{
  (
    increment_sole_comm::<E>(ro_consts, prev_ic.0, comm_advice.0),
    increment_sole_comm::<E>(ro_consts, prev_ic.1, comm_advice.1),
  )
}

/// Produce an incremental commitment to already pedersen committed non-deterministic advice ω
pub(crate) fn increment_sole_comm<E>(
  ro_consts: &ROConstants<Dual<E>>,
  prev_ic: E::Scalar,
  comm_advice: Commitment<E>, // commitment to non-deterministic witness ω
) -> E::Scalar
where
  E: CurveCycleEquipped,
{
  let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
  ro.absorb(prev_ic);
  absorb_primary_commitment::<E, Dual<E>>(&comm_advice, &mut ro);
  scalar_as_base::<Dual<E>>(ro.squeeze(NUM_HASH_BITS))
}
