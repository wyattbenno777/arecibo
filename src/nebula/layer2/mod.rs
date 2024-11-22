//! This module contains components for the "second layer" which is used to fold finalized auxillary IVC proofs from the first layer.

use super::rs::PublicParams;
use crate::{
  r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::CurveCycleEquipped,
};

mod final_circuit;
mod nifs;
mod utils;

#[cfg(test)]
mod tests;

/// Trait for layer2 getters
trait Layer2Getters {
  type T;

  fn F(&self) -> Self::T;
  fn ops(&self) -> Self::T;
  fn scan(&self) -> Self::T;
}

/// Layer2 PP
type PP<'a, E1> = (
  &'a PublicParams<E1>,
  &'a PublicParams<E1>,
  &'a PublicParams<E1>,
);

impl<'a, E1> Layer2Getters for PP<'a, E1>
where
  E1: CurveCycleEquipped,
{
  type T = &'a PublicParams<E1>;

  fn F(&self) -> Self::T {
    self.0
  }

  fn ops(&self) -> Self::T {
    self.1
  }

  fn scan(&self) -> Self::T {
    self.2
  }
}

/// Finalized IVC proof
type RS<'a, E1> = (
  &'a RecursiveSNARK<E1>,
  &'a RecursiveSNARK<E1>,
  &'a RecursiveSNARK<E1>,
);

/// [`RecursiveSNARK`] used to restore incremantality from Nebula finalized IVC proofs
pub struct RecursiveSNARK<E1>
where
  E1: CurveCycleEquipped,
{
  r_U: (
    RelaxedR1CSInstance<E1>,
    RelaxedR1CSInstance<E1>,
    RelaxedR1CSInstance<E1>,
  ),
  r_W: (
    RelaxedR1CSWitness<E1>,
    RelaxedR1CSWitness<E1>,
    RelaxedR1CSWitness<E1>,
  ),
}

impl<E1> RecursiveSNARK<E1>
where
  E1: CurveCycleEquipped,
{
  /// Create a new [`RecursiveSNARK`]
  pub fn new<'a>(pp: PP<'a, E1>) -> Self {
    // Initialize the initial relaxed instance and witness pairs.
    let (r_U, r_W) = {
      let r_U_F =
        RelaxedR1CSInstance::default(&pp.F().ck_primary, &pp.F().circuit_shape_primary.r1cs_shape);
      let r_U_ops = RelaxedR1CSInstance::default(
        &pp.ops().ck_primary,
        &pp.ops().circuit_shape_primary.r1cs_shape,
      );
      let r_U_scan = RelaxedR1CSInstance::default(
        &pp.scan().ck_primary,
        &pp.scan().circuit_shape_primary.r1cs_shape,
      );

      let r_W_F = RelaxedR1CSWitness::default(&pp.F().circuit_shape_primary.r1cs_shape);
      let r_W_ops = RelaxedR1CSWitness::default(&pp.ops().circuit_shape_primary.r1cs_shape);
      let r_W_scan = RelaxedR1CSWitness::default(&pp.scan().circuit_shape_primary.r1cs_shape);

      ((r_U_F, r_U_ops, r_U_scan), (r_W_F, r_W_ops, r_W_scan))
    };

    Self { r_U, r_W }
  }

  /// updates the provided [`RecursiveSNARK`]
  /// by executing a step of the incremental computation
  pub fn prove_step<'a>(&self, pp: PP<'a, E1>, rs: RS<'a, E1>) {}
}
