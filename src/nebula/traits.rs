//! Defines behavior of Layer 1 Nebula proofs

use crate::{
  nebula::{
    audit_rs::{AuditPublicParams, AuditRecursiveSNARK},
    rs::{PublicParams, RecursiveSNARK},
  },
  traits::CurveCycleEquipped,
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
}
