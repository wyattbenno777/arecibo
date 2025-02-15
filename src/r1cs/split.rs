//! Implements split R1CS witness and corresponding R1CS instance, according to
//! the Nebula paper. Used to help facilitate randomized circuits.

use super::{R1CSInstance, R1CSWitness};
use crate::{
  traits::{commitment::CommitmentEngineTrait, Engine},
  Commitment, CommitmentKey, CE,
};
use ff::Field;

/// A split R1CS instance.
pub struct SplitR1CSInstance<E>
where
  E: Engine,
{
  pub(crate) aux: R1CSInstance<E>,
  pub(crate) pre_committed: (Commitment<E>, Commitment<E>),
}

/// A split R1CS witness.
pub struct SplitR1CSWitness<E>
where
  E: Engine,
{
  pub(crate) aux: R1CSWitness<E>,
  pub(crate) pre_committed: (Vec<E::Scalar>, Vec<E::Scalar>),
}

impl<E> SplitR1CSWitness<E>
where
  E: Engine,
{
  /// Create a new instance of [`SplitR1CSWitness`].
  pub fn new(aux: R1CSWitness<E>, pre_committed: (Vec<E::Scalar>, Vec<E::Scalar>)) -> Self {
    Self { aux, pre_committed }
  }

  /// Get the precommitted commitments
  pub fn commit(&self, ck: &CommitmentKey<E>) -> (Commitment<E>, Commitment<E>) {
    (
      CE::<E>::commit(ck, &self.pre_committed.0, &E::Scalar::ZERO),
      CE::<E>::commit(ck, &self.pre_committed.1, &E::Scalar::ZERO),
    )
  }
}

impl<E> SplitR1CSInstance<E>
where
  E: Engine,
{
  /// Create a new instance of [`SplitR1CSInstance`].
  pub fn new(aux: R1CSInstance<E>, pre_committed: (Commitment<E>, Commitment<E>)) -> Self {
    Self { aux, pre_committed }
  }
}
