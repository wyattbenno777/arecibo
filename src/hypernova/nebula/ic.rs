//! This module contains an incremental commitment scheme implementation.

use std::marker::PhantomData;

use crate::{
  constants::{DEFAULT_ABSORBS, NUM_HASH_BITS},
  cyclefold::util::absorb_primary_commitment,
  gadgets::scalar_as_base,
  traits::{
    commitment::CommitmentEngineTrait, CurveCycleEquipped, Dual, Engine, ROConstants, ROTrait,
  },
  CommitmentKey,
};
use ff::Field;

/// Incremental commitment engine
///
/// Produces the incremental commitments needed for CC-NIVC
pub struct ICEngine<E>
where
  E: CurveCycleEquipped,
{
  _engine: PhantomData<E>,
}

impl<E> ICEngine<E>
where
  E: CurveCycleEquipped,
{
  /// Produce all the incremental commitments needed for CC-NIVC
  pub fn incremental_comms(advice: &[Vec<E::Scalar>]) -> Vec<E::Scalar> {
    todo!()
  }
}

/// Produce an incremental commitment to a non-deterministic advice ω
///
/// * commits to advice with Pedersen
/// * hashes previous commitment & pedersen commitment to advice
/// * outputs hash bits as scalar
pub fn increment_commitment<E>(
  ck: &CommitmentKey<E>,
  ro_consts: &ROConstants<Dual<E>>,
  prev_ic: E::Scalar,
  advice: &[E::Scalar],
) -> E::Scalar
where
  E: CurveCycleEquipped,
{
  let comm_advice = E::CE::commit(ck, advice, &E::Scalar::ZERO);
  let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
  ro.absorb(prev_ic);
  absorb_primary_commitment::<E, Dual<E>>(&comm_advice, &mut ro);
  scalar_as_base::<Dual<E>>(ro.squeeze(NUM_HASH_BITS))
}
