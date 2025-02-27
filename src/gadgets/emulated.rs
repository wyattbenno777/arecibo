//! Emulated gadgets

use ff::Field;

use super::{
  alloc_bignat_constant, conditionally_select_allocated_bit, conditionally_select_bignat, f_to_nat,
  BigNat,
};
use crate::{
  frontend::{num::AllocatedNum, AllocatedBit, Boolean, ConstraintSystem, SynthesisError},
  traits::{Group, ROCircuitTrait},
};

/// An allocated version of a curve point from the non-native curve
#[derive(Clone, Debug)]
pub struct AllocatedEmulPoint<G>
where
  G: Group,
{
  pub x: BigNat<G::Base>,
  pub y: BigNat<G::Base>,
  pub is_infinity: AllocatedBit,
}

impl<G> AllocatedEmulPoint<G>
where
  G: Group,
{
  pub fn alloc<CS>(
    mut cs: CS,
    coords: Option<(G::Scalar, G::Scalar, bool)>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<<G as Group>::Base>,
  {
    let x = BigNat::alloc_from_nat(
      cs.namespace(|| "x"),
      || {
        Ok(f_to_nat(
          &coords.map_or(<G::Scalar as Field>::ZERO, |val| val.0),
        ))
      },
      limb_width,
      n_limbs,
    )?;

    let y = BigNat::alloc_from_nat(
      cs.namespace(|| "y"),
      || {
        Ok(f_to_nat(
          &coords.map_or(<G::Scalar as Field>::ZERO, |val| val.1),
        ))
      },
      limb_width,
      n_limbs,
    )?;

    let is_infinity = AllocatedBit::alloc(
      cs.namespace(|| "alloc is_infinity"),
      coords.map_or(Some(true), |(_, _, is_infinity)| Some(is_infinity)),
    )?;

    Ok(Self { x, y, is_infinity })
  }

  pub fn absorb_in_ro<CS>(
    &self,
    mut cs: CS,
    ro: &mut impl ROCircuitTrait<G::Base>,
  ) -> Result<(), SynthesisError>
  where
    CS: ConstraintSystem<G::Base>,
  {
    let x_bn = self
      .x
      .as_limbs()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of x to num")))
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    for limb in x_bn {
      ro.absorb(&limb)
    }

    let y_bn = self
      .y
      .as_limbs()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of y to num")))
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    for limb in y_bn {
      ro.absorb(&limb)
    }

    let is_infinity_num: AllocatedNum<G::Base> =
      AllocatedNum::alloc(cs.namespace(|| "is_infinity"), || {
        self
          .is_infinity
          .get_value()
          .map_or(Err(SynthesisError::AssignmentMissing), |bit| {
            if bit {
              Ok(G::Base::ONE)
            } else {
              Ok(G::Base::ZERO)
            }
          })
      })?;

    cs.enforce(
      || "constrain num equals bit",
      |lc| lc,
      |lc| lc,
      |lc| lc + is_infinity_num.get_variable() - self.is_infinity.get_variable(),
    );

    ro.absorb(&is_infinity_num);

    Ok(())
  }

  pub fn conditionally_select<CS: ConstraintSystem<G::Base>>(
    &self,
    mut cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = conditionally_select_bignat(
      cs.namespace(|| "x = cond ? self.x : other.x"),
      &self.x,
      &other.x,
      condition,
    )?;

    let y = conditionally_select_bignat(
      cs.namespace(|| "y = cond ? self.y : other.y"),
      &self.y,
      &other.y,
      condition,
    )?;

    let is_infinity = conditionally_select_allocated_bit(
      cs.namespace(|| "is_infinity = cond ? self.is_infinity : other.is_infinity"),
      &self.is_infinity,
      &other.is_infinity,
      condition,
    )?;

    Ok(Self { x, y, is_infinity })
  }

  pub fn default<CS: ConstraintSystem<G::Base>>(
    mut cs: CS,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    let x = alloc_bignat_constant(
      cs.namespace(|| "allocate x_default = 0"),
      &f_to_nat(&G::Base::ZERO),
      limb_width,
      n_limbs,
    )?;
    let y = alloc_bignat_constant(
      cs.namespace(|| "allocate y_default = 0"),
      &f_to_nat(&G::Base::ZERO),
      limb_width,
      n_limbs,
    )?;

    let is_infinity = AllocatedBit::alloc(cs.namespace(|| "allocate is_infinity"), Some(true))?;
    cs.enforce(
      || "is_infinity = 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + CS::one() - is_infinity.get_variable(),
    );

    Ok(Self { x, y, is_infinity })
  }

  pub fn to_coordinates(&self) -> (BigNat<G::Base>, BigNat<G::Base>, AllocatedBit) {
    (self.x.clone(), self.y.clone(), self.is_infinity.clone())
  }
}
