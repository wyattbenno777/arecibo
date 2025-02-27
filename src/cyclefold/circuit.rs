//! This module defines Cyclefold circuit

use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{
    gadgets::poseidon::poseidon_hash_allocated, AllocatedBit, Boolean, ConstraintSystem,
    PoseidonConstants, SynthesisError,
  },
  gadgets::{alloc_zero, conditionally_select, le_bits_to_num, AllocatedPoint},
  traits::{commitment::CommitmentTrait, Engine},
  Commitment,
};
use ff::Field;

/// A structure containing the CycleFold circuit inputs and implementing the synthesize function
pub struct CycleFoldCircuit<E: Engine> {
  commit_1: Option<Commitment<E>>,
  commit_2: Option<Commitment<E>>,
  scalar: Option<[bool; NUM_CHALLENGE_BITS]>,
  poseidon_constants: PoseidonConstants<E::Base, generic_array::typenum::U2>,
}

impl<E: Engine> Default for CycleFoldCircuit<E> {
  fn default() -> Self {
    let poseidon_constants = PoseidonConstants::new();
    Self {
      commit_1: None,
      commit_2: None,
      scalar: None,
      poseidon_constants,
    }
  }
}
impl<E: Engine> CycleFoldCircuit<E> {
  /// Create a new CycleFold circuit with the given inputs
  pub fn new(
    commit_1: Option<Commitment<E>>,
    commit_2: Option<Commitment<E>>,
    scalar: Option<[bool; NUM_CHALLENGE_BITS]>,
  ) -> Self {
    let poseidon_constants = PoseidonConstants::new();
    Self {
      commit_1,
      commit_2,
      scalar,
      poseidon_constants,
    }
  }

  fn alloc_witness<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<
    (
      AllocatedPoint<E::GE>, // commit_1
      AllocatedPoint<E::GE>, // commit_2
      Vec<AllocatedBit>,     // scalar
    ),
    SynthesisError,
  > {
    let commit_1 = AllocatedPoint::alloc(
      cs.namespace(|| "allocate C_1"),
      self.commit_1.map(|C_1| C_1.to_coordinates()),
    )?;
    commit_1.check_on_curve(cs.namespace(|| "commit_1 on curve"))?;

    let commit_2 = AllocatedPoint::alloc(
      cs.namespace(|| "allocate C_2"),
      self.commit_2.map(|C_2| C_2.to_coordinates()),
    )?;
    commit_2.check_on_curve(cs.namespace(|| "commit_2 on curve"))?;

    let scalar: Vec<AllocatedBit> = self
      .scalar
      .unwrap_or([false; NUM_CHALLENGE_BITS])
      .into_iter()
      .enumerate()
      .map(|(idx, bit)| {
        AllocatedBit::alloc(cs.namespace(|| format!("scalar bit {idx}")), Some(bit))
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok((commit_1, commit_2, scalar))
  }

  /// Synthesize the CycleFold circuit
  pub fn synthesize<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<(), SynthesisError> {
    let (C_1, C_2, r) = self.alloc_witness(cs.namespace(|| "allocate circuit witness"))?;

    // Calculate C_final
    let r_C_2 = C_2.scalar_mul(cs.namespace(|| "r * C_2"), &r)?;

    let C_final = C_1.add(cs.namespace(|| "C_1 + r * C_2"), &r_C_2)?;

    self.inputize_point(&C_1, cs.namespace(|| "inputize C_1"))?;
    self.inputize_point(&C_2, cs.namespace(|| "inputize C_2"))?;
    self.inputize_point(&C_final, cs.namespace(|| "inputize C_final"))?;

    let scalar = le_bits_to_num(cs.namespace(|| "get scalar"), &r)?;

    scalar.inputize(cs.namespace(|| "scalar"))?;

    Ok(())
  }

  // Represent the point in the public IO as its 2-ary Poseidon hash
  fn inputize_point<CS>(
    &self,
    point: &AllocatedPoint<E::GE>,
    mut cs: CS,
  ) -> Result<(), SynthesisError>
  where
    E: Engine,
    CS: ConstraintSystem<E::Base>,
  {
    let (x, y, is_infinity) = point.get_coordinates();
    let preimage = vec![x.clone(), y.clone()];
    let val = poseidon_hash_allocated(
      cs.namespace(|| "hash point"),
      preimage,
      &self.poseidon_constants,
    )?;

    let zero = alloc_zero(cs.namespace(|| "zero"));

    let is_infinity_bit = AllocatedBit::alloc(
      cs.namespace(|| "is_infinity"),
      Some(is_infinity.get_value().unwrap_or(E::Base::ONE) == E::Base::ONE),
    )?;

    cs.enforce(
      || "infinity_bit matches",
      |lc| lc,
      |lc| lc,
      |lc| lc + is_infinity_bit.get_variable() - is_infinity.get_variable(),
    );

    // Output 0 when it is the point at infinity
    let output = conditionally_select(
      cs.namespace(|| "select output"),
      &zero,
      &val,
      &Boolean::from(is_infinity_bit),
    )?;

    output.inputize(cs.namespace(|| "inputize hash"))?;

    Ok(())
  }
}
