//! Implementation of commitment-carrying Hypernova

use ff::PrimeField;

use crate::frontend::{ConstraintSystem, SynthesisError};
mod augmented_circuit;
pub mod error;
pub mod nifs;

/// Step circuit used for Hypernova
pub trait StepCircuit<F: PrimeField> {
  /// Arity of the circuit. This is needed to build the public parameters
  fn arity(&self) -> usize;

  /// Synthesize the circuit
  fn synthesize<CS>(&self, cs: CS) -> Result<(), SynthesisError>
  where
    CS: ConstraintSystem<F>;
}
