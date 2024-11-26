//! This module contains components for the "second layer" which is used to fold finalized auxillary IVC proofs from the first layer.

use nifs::RelaxedNIFS;
use utils::RelaxedFoldingData;

use super::rs::{PublicParams, RecursiveSNARK};
use crate::cyclefold::util::FoldingData;
use crate::traits::commitment::CommitmentTrait;
use crate::{
  errors::NovaError,
  r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{CurveCycleEquipped, Dual},
  Commitment,
};

mod final_circuit;
pub(crate) mod gadgets;
pub(crate) mod nifs;
pub(crate) mod utils;

#[cfg(test)]
mod tests;

/// Trait for layer2 getters
trait Layer2Getters {
  type T;

  fn F(&self) -> &Self::T;

  fn ops(&self) -> &Self::T;
  fn scan(&self) -> &Self::T;
}

type M<T> = (T, T, T);

impl<T> Layer2Getters for M<T> {
  type T = T;

  fn F(&self) -> &Self::T {
    &self.0
  }

  fn ops(&self) -> &Self::T {
    &self.1
  }

  fn scan(&self) -> &Self::T {
    &self.2
  }
}

/// Layer2 PP
type PP<'a, E1> = (
  &'a PublicParams<E1>,
  &'a PublicParams<E1>,
  &'a PublicParams<E1>,
);

/// Finalized IVC proof
type RS<'a, E1> = (
  &'a RecursiveSNARK<E1>,
  &'a RecursiveSNARK<E1>,
  &'a RecursiveSNARK<E1>,
);

/// [`Layer2RS`] used to restore incremantality from Nebula finalized IVC proofs
pub struct Layer2RS<E1>
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

impl<E1> Layer2RS<E1>
where
  E1: CurveCycleEquipped,
{
  /// Create a new [`Layer2RS`]
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

  /// updates the provided [`Layer2RS`]
  /// by executing a step of the incremental computation
  pub fn prove_step<'a>(&self, pp: PP<'a, E1>, rs: RS<'a, E1>) -> Result<(), NovaError> {
    let pp = *pp.F();
    let (l_U, l_W) = rs.F().U_W();
    let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E1>::prove(
      &pp.ck_primary,
      &pp.ro_consts_primary,
      &pp.digest(),
      &pp.circuit_shape_primary.r1cs_shape,
      self.r_U.F(),
      self.r_W.F(),
      l_U,
      l_W,
    )?;
    let comm_T = Commitment::<E1>::decompress(&nifs_primary.comm_T)?;

    let data_p = RelaxedFoldingData::new(self.r_U.F().clone(), l_U.clone(), comm_T);

    Ok(())
  }
}
