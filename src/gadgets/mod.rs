//! This module implements various gadgets necessary for Nova and applications built with Nova.
mod ecc;
pub(crate) use ecc::AllocatedPoint;

mod nonnative;
pub(crate) use nonnative::{
  bignat::{nat_to_limbs, BigNat},
  util::{f_to_nat, Num},
};

mod r1cs;
pub(crate) use r1cs::AllocatedRelaxedR1CSInstance;

mod utils;
pub(crate) use utils::{
  alloc_bignat_constant, alloc_num_equals, alloc_one, alloc_scalar_as_base, alloc_tuple,
  alloc_tuple_comms, alloc_zero, conditionally_select, conditionally_select_allocated_bit,
  conditionally_select_bignat, conditionally_select_vec, le_bits_to_num, scalar_as_base,
};

pub(crate) mod emulated;
pub(crate) mod hypernova;
pub(crate) mod int;
pub(crate) mod nebula;

#[cfg(test)]
pub(crate) use utils::conditionally_select2;
