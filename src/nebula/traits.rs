//! Defines behavior of Layer 1 Nebula proofs

use std::sync::Arc;

use crate::r1cs::R1CSShape;
use crate::traits::commitment::Len;
use crate::traits::Dual;
use crate::{
  nebula::{
    audit_rs::{AuditPublicParams, AuditRecursiveSNARK},
    rs::{PublicParams, RecursiveSNARK},
  },
  traits::CurveCycleEquipped,
  CommitmentKey,
};

/// Defines how a Layer1 RecursiveSNARK should be structured
pub trait Layer1RSTrait<E>
where
  E: CurveCycleEquipped,
{
  /// Returns the F instance
  fn F(&self) -> &RecursiveSNARK<E>;

  /// Returns the ops instance
  fn ops(&self) -> &RecursiveSNARK<E>;

  /// Returns the scan instance
  fn scan(&self) -> &AuditRecursiveSNARK<E>;
}

/// Defines how a Layer 1 Nebula PublicParams should be structured
pub trait Layer1PPTrait<E: CurveCycleEquipped> {
  /// Splits the PublicParams into three parts (F, ops, scan)
  fn into_parts(self) -> (PublicParams<E>, PublicParams<E>, AuditPublicParams<E>);

  /// Returns the F public params
  fn F(&self) -> &PublicParams<E>;

  /// Returns the ops public params
  fn ops(&self) -> &PublicParams<E>;

  /// Returns the scan public params
  fn scan(&self) -> &AuditPublicParams<E>;

  /// Get the biggest commitmentkey
  fn biggest_ck<'a>(&'a self) -> &'a Arc<CommitmentKey<E>>
  where
    E: 'a,
  {
    let ck_F = self.F().ck();
    let ck_ops = self.ops().ck();
    let ck_scan = self.scan().ck();
    let mut ck = ck_F;
    if ck_ops.length() > ck.length() {
      ck = ck_ops;
    }
    if ck_scan.length() > ck.length() {
      ck = ck_scan;
    }
    ck
  }

  /// Get the primary R1CS shapes
  fn primary_r1cs_shapes(&self) -> Vec<&R1CSShape<E>> {
    vec![
      &self.F().circuit_shape_primary.r1cs_shape,
      &self.ops().circuit_shape_primary.r1cs_shape,
      &self.scan().circuit_shape_primary.r1cs_shape,
    ]
  }

  /// Get the secondary R1CS shapes
  fn secondary_r1cs_shapes<'a>(&'a self) -> Vec<&'a R1CSShape<Dual<E>>>
  where
    E: 'a,
  {
    vec![
      &self.F().circuit_shape_cyclefold.r1cs_shape,
      &self.ops().circuit_shape_cyclefold.r1cs_shape,
      &self.scan().circuit_shape_cyclefold.r1cs_shape,
    ]
  }

  /// Get the secondary ck
  fn ck_secondary<'a>(&'a self) -> &'a Arc<CommitmentKey<Dual<E>>>
  where
    E: 'a,
  {
    &self.F().ck_cyclefold
  }
}

/// Get the scan commitments from the statement the Layer 1 proof is proving
pub trait MemoryCommitmentsTraits<E>
where
  E: CurveCycleEquipped,
{
  /// Get commitment to C_is
  fn C_IS(&self) -> E::Scalar;
  /// commitment to C_fs
  fn C_FS(&self) -> E::Scalar;
}
