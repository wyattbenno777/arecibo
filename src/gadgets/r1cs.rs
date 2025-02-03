//! This module implements various gadgets necessary for folding R1CS types.
use super::{
  alloc_scalar_as_base, conditionally_select_bignat,
  nonnative::{bignat::BigNat, util::f_to_nat},
  utils::conditionally_select,
};
use crate::frontend::gadgets::{boolean::Boolean, num::AllocatedNum};
use crate::frontend::{ConstraintSystem, SynthesisError};
use crate::{
  gadgets::ecc::AllocatedPoint,
  r1cs::RelaxedR1CSInstance,
  traits::{commitment::CommitmentTrait, Engine, Group, ROCircuitTrait},
};
use ff::Field;
use itertools::Itertools as _;

/// An Allocated Relaxed R1CS Instance
#[derive(Clone)]
pub struct AllocatedRelaxedR1CSInstance<E: Engine, const N: usize> {
  pub(crate) W: AllocatedPoint<E::GE>,
  pub(crate) E: AllocatedPoint<E::GE>,
  pub(crate) u: AllocatedNum<E::Base>,
  pub(crate) X: [BigNat<E::Base>; N],
}

impl<E: Engine, const N: usize> AllocatedRelaxedR1CSInstance<E, N> {
  /// Allocates the given [`RelaxedR1CSInstance`] as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    inst: Option<&RelaxedR1CSInstance<E>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    // We do not need to check that W or E are well-formed (e.g., on the curve) as we do a hash check
    // in the Nova augmented circuit, which ensures that the relaxed instance
    // came from a prior iteration of Nova.
    let W = AllocatedPoint::alloc(
      cs.namespace(|| "allocate W"),
      inst.map(|inst| inst.comm_W.to_coordinates()),
    )?;

    let E = AllocatedPoint::alloc(
      cs.namespace(|| "allocate E"),
      inst.map(|inst| inst.comm_E.to_coordinates()),
    )?;

    // u << |E::Base| despite the fact that u is a scalar.
    // So we parse all of its bytes as a E::Base element
    let u = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate u"), inst.map(|inst| inst.u))?;

    // Allocate X. If the input instance is None then allocate components as zero.
    let X = (0..N)
      .map(|idx| {
        BigNat::alloc_from_nat(
          cs.namespace(|| format!("allocate X[{idx}]")),
          || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[idx]))),
          limb_width,
          n_limbs,
        )
      })
      .collect::<Result<Vec<_>, _>>()?
      .try_into()
      .map_err(|err: Vec<_>| {
        SynthesisError::IncompatibleLengthVector(format!("{} != {N}", err.len()))
      })?;

    Ok(Self { W, E, u, X })
  }

  /// Allocates the hardcoded default `RelaxedR1CSInstance` in the circuit.
  /// W = E = 0, u = 0, X0 = X1 = 0
  pub fn default<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    let W = AllocatedPoint::default(cs.namespace(|| "allocate W"));
    let E = W.clone();

    let u = W.x.clone(); // In the default case, W.x = u = 0

    // X is allocated and in the honest prover case set to zero
    // If the prover is malicious, it can set to arbitrary values, but the resulting
    // relaxed R1CS instance with the the checked default values of W, E, and u must still be satisfying

    let X = (0..N)
      .map(|idx| {
        BigNat::alloc_from_nat(
          cs.namespace(|| format!("allocate X_default[{idx}]")),
          || Ok(f_to_nat(&E::Scalar::ZERO)),
          limb_width,
          n_limbs,
        )
      })
      .collect::<Result<Vec<_>, _>>()?
      .try_into()
      .map_err(|err: Vec<_>| {
        SynthesisError::IncompatibleLengthVector(format!("{} != {N}", err.len()))
      })?;

    Ok(Self { W, E, u, X })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    ro: &mut E::ROCircuit,
  ) -> Result<(), SynthesisError> {
    ro.absorb(&self.W.x);
    ro.absorb(&self.W.y);
    ro.absorb(&self.W.is_infinity);
    ro.absorb(&self.E.x);
    ro.absorb(&self.E.y);
    ro.absorb(&self.E.is_infinity);
    ro.absorb(&self.u);

    self.X.iter().enumerate().try_for_each(|(idx, X)| {
      X.as_limbs()
        .iter()
        .enumerate()
        .try_for_each(|(i, limb)| -> Result<(), SynthesisError> {
          ro.absorb(
            &limb.as_allocated_num(
              cs.namespace(|| format!("convert limb {i} of X_r[{idx}] to num")),
            )?,
          );
          Ok(())
        })
    })?;

    Ok(())
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    conditionally_select_alloc_relaxed_r1cs(cs, self, other, condition)
  }
}

/// c = cond ? a: b, where a, b: `AllocatedRelaxedR1CSInstance`
pub fn conditionally_select_alloc_relaxed_r1cs<
  E: Engine,
  CS: ConstraintSystem<<E as Engine>::Base>,
  const N: usize,
>(
  mut cs: CS,
  a: &AllocatedRelaxedR1CSInstance<E, N>,
  b: &AllocatedRelaxedR1CSInstance<E, N>,
  condition: &Boolean,
) -> Result<AllocatedRelaxedR1CSInstance<E, N>, SynthesisError> {
  let c_X = a
    .X
    .iter()
    .zip_eq(b.X.iter())
    .enumerate()
    .map(|(idx, (a, b))| {
      conditionally_select_bignat(
        cs.namespace(|| format!("X[{idx}] = cond ? a.X[{idx}] : b.X[{idx}]")),
        a,
        b,
        condition,
      )
    })
    .collect::<Result<Vec<_>, _>>()?;

  let c_X = c_X.try_into().map_err(|err: Vec<_>| {
    SynthesisError::IncompatibleLengthVector(format!("{} != {N}", err.len()))
  })?;

  let c = AllocatedRelaxedR1CSInstance {
    W: conditionally_select_point(
      cs.namespace(|| "W = cond ? a.W : b.W"),
      &a.W,
      &b.W,
      condition,
    )?,
    E: conditionally_select_point(
      cs.namespace(|| "E = cond ? a.E : b.E"),
      &a.E,
      &b.E,
      condition,
    )?,
    u: conditionally_select(
      cs.namespace(|| "u = cond ? a.u : b.u"),
      &a.u,
      &b.u,
      condition,
    )?,
    X: c_X,
  };
  Ok(c)
}

/// c = cond ? a: b, where a, b: `AllocatedPoint`
pub fn conditionally_select_point<G: Group, CS: ConstraintSystem<G::Base>>(
  mut cs: CS,
  a: &AllocatedPoint<G>,
  b: &AllocatedPoint<G>,
  condition: &Boolean,
) -> Result<AllocatedPoint<G>, SynthesisError> {
  let c = AllocatedPoint {
    x: conditionally_select(
      cs.namespace(|| "x = cond ? a.x : b.x"),
      &a.x,
      &b.x,
      condition,
    )?,
    y: conditionally_select(
      cs.namespace(|| "y = cond ? a.y : b.y"),
      &a.y,
      &b.y,
      condition,
    )?,
    is_infinity: conditionally_select(
      cs.namespace(|| "is_infinity = cond ? a.is_infinity : b.is_infinity"),
      &a.is_infinity,
      &b.is_infinity,
      condition,
    )?,
  };
  Ok(c)
}
