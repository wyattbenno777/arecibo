//! This module implements an efficient read-write memory primitive
//! for recursive proof systems using commitment-carrying NIVC

pub mod api;
pub mod ic;
pub use product_circuits::convert_advice;

mod product_circuits;
#[cfg(test)]
mod tests;
