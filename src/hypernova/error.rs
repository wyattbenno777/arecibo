//! This module defines errors returned by the library.
use core::fmt::Debug;
use thiserror::Error;

/// Errors returned by Nova
#[derive(Debug, Eq, PartialEq, Error)]
pub enum HyperNovaError {
  /// Invalid Evaluation Point
  #[error("InvalidEvaluationPoint")]
  InvalidEvaluationPoint,
  /// Invalid Targets
  #[error("InvalidTargets")]
  InvalidTargets,
}
