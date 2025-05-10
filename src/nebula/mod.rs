//! This module implements commitment-carrying NIVC

// private modules
pub mod augmented_circuit;
pub mod nifs;

// public modules
pub mod audit_rs;
pub mod compression;
pub mod ic;
pub mod layer_2;
pub mod rs;
pub mod traits;

pub use augmented_circuit::AugmentedCircuitParams;
