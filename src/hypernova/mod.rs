//! Implementation of commitment-carrying Hypernova.
//!
//! This version of HyperNova is tailor made to instantiate Nebula's solution to
//! bringing in MCC to (folding) IVC schemes.

/// private modules
mod augmented_circuit;
mod utils;

// public modules
pub mod compression;
pub mod error;
pub mod nebula;
pub mod nifs;
pub mod pp;
pub mod ro_sumcheck;
pub mod rs;
