//! Implementation of commitment-carrying Hypernova

use ff::PrimeField;

use crate::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};
mod augmented_circuit;
pub mod error;
pub mod nifs;
pub mod ro_sumcheck;
pub mod rs;

/// Step circuit used for Hypernova
pub trait StepCircuit<F: PrimeField> {
  /// Arity of the circuit. This is needed to build the public parameters
  fn arity(&self) -> usize;

  /// Synthesize the circuit
  fn synthesize<CS>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
  where
    CS: ConstraintSystem<F>;
}
