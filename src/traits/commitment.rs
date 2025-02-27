//! This module defines a collection of traits that define the behavior of a commitment engine
//! We require the commitment engine to provide a commitment to vectors with a single group element
use crate::frontend::{ConstraintSystem, SynthesisError};
use crate::gadgets::AllocatedPoint;
use crate::{
  errors::NovaError,
  provider::traits::DlogGroup,
  traits::{AbsorbInROTrait, Engine, TranscriptReprTrait},
};
use core::{
  fmt::Debug,
  ops::{Add, Mul, MulAssign},
};
use group::prime::PrimeCurve;
use serde::{Deserialize, Serialize};

/// A helper trait for types implementing scalar multiplication.
pub trait ScalarMul<Rhs, Output = Self>: Mul<Rhs, Output = Output> + MulAssign<Rhs> {}

impl<T, Rhs, Output> ScalarMul<Rhs, Output> for T where T: Mul<Rhs, Output = Output> + MulAssign<Rhs>
{}

/// This trait defines the behavior of the commitment
pub trait CommitmentTrait<E: Engine>:
  Clone
  + Copy
  + Debug
  + Default
  + PartialEq
  + Eq
  + Send
  + Sync
  + TranscriptReprTrait<E::GE>
  + Serialize
  + for<'de> Deserialize<'de>
  + AbsorbInROTrait<E>
  + Add<Self, Output = Self>
  + ScalarMul<E::Scalar>
{
  /// Holds the type of the compressed commitment
  type CompressedCommitment: Clone
    + Debug
    + PartialEq
    + Eq
    + Send
    + Sync
    + TranscriptReprTrait<E::GE>
    + Serialize
    + for<'de> Deserialize<'de>;

  /// Compresses self into a compressed commitment
  fn compress(&self) -> Self::CompressedCommitment;

  /// Returns the coordinate representation of the commitment
  fn to_coordinates(&self) -> (E::Base, E::Base, bool);

  /// Decompresses a compressed commitment into a commitment
  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, NovaError>;

  /// Reinterpret as generator
  fn reinterpret_as_generator(&self) -> <<E as Engine>::GE as PrimeCurve>::Affine
  where
    E::GE: DlogGroup;
}

/// A trait that helps determine the length of a structure.
/// Note this does not impose any memory representation constraints on the structure.
pub trait Len {
  /// Returns the length of the structure.
  fn length(&self) -> usize;
}

/// A trait that ties different pieces of the commitment generation together
pub trait CommitmentEngineTrait<E: Engine>: Clone + Send + Sync {
  /// Holds the type of the commitment key
  /// The key should quantify its length in terms of group generators.
  type CommitmentKey: Len
    + Clone
    + PartialEq
    + Debug
    + Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>;
  /// Holds the type of the derandomization key
  type DerandKey: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Holds the type of the commitment
  type Commitment: CommitmentTrait<E>;

  /// Samples a new commitment key of a specified size
  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey;

  /// Extracts the blinding generator
  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey;

  /// Commits to the provided vector using the provided generators
  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &E::Scalar) -> Self::Commitment {
    Self::commit_at(ck, v, r, 0)
  }

  /// Commits to the provided vector using the provided generators
  fn commit_at(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &E::Scalar,
    idx: usize,
  ) -> Self::Commitment;

  /// Commits to the provided vector using the provided generators in circuit
  // TODO: Maybe pass AllocatedRelaxedR1CSInstance instead of v and r
  fn commit_gadget<CS: ConstraintSystem<E::Base>>(cs: &mut CS, _ck: &Self::CommitmentKey, _v: &[E::Scalar], _r: &E::Scalar) -> Result<AllocatedPoint<E::GE>, SynthesisError>;

  /// Remove given blind from commitment
  fn derandomize(
    dk: &Self::DerandKey,
    commit: &Self::Commitment,
    r: &E::Scalar,
  ) -> Self::Commitment;
}
