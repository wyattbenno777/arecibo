//! This library implements Nova, a high-speed recursive SNARK.
#![deny(
  // warnings,
  // unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case, clippy::upper_case_acronyms)]
// #![forbid(unsafe_code)] // Commented for development with `Abomonation`

use digest::{DigestComputer, SimpleDigestible};
use r1cs::R1CSShape;
use serde::{Deserialize, Serialize};
use traits::{
  commitment::{CommitmentEngineTrait, CommitmentTrait},
  Engine,
};

// private modules
mod digest;

// public modules
pub mod constants;
pub mod errors;
pub mod frontend;
pub mod gadgets;
pub mod hypernova;
pub mod provider;
pub(crate) mod r1cs;
pub mod spartan;
pub mod traits;

mod cyclefold;
pub mod nebula;
// pub mod supernova;
pub mod onchain;
pub use errors::NovaError;
pub(crate) use nebula::AugmentedCircuitParams;
use traits::ROConstants;

/// A type that holds parameters for the primary and secondary circuits of Nova and SuperNova
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSWithArity<E: Engine> {
  F_arity: usize,
  r1cs_shape: R1CSShape<E>,
}

impl<E: Engine> SimpleDigestible for R1CSWithArity<E> {}

impl<E: Engine> R1CSWithArity<E> {
  /// Create a new `R1CSWithArity`
  pub fn new(r1cs_shape: R1CSShape<E>, F_arity: usize) -> Self {
    Self {
      F_arity,
      r1cs_shape,
    }
  }

  /// Return the [`R1CSWithArity`]' digest.
  pub fn digest(&self) -> E::Scalar {
    let dc: DigestComputer<'_, <E as Engine>::Scalar, Self> = DigestComputer::new(self);
    dc.digest().expect("Failure in computing digest")
  }

  /// Get the shape for [`R1CSWithArity`]
  pub fn shape(&self) -> &R1CSShape<E> {
    &self.r1cs_shape
  }
}

type CommitmentKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey;
type DerandKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::DerandKey;
type Commitment<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment;
type CompressedCommitment<E> = <<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as CommitmentTrait<E>>::CompressedCommitment;
type CE<E> = <E as Engine>::CE;
