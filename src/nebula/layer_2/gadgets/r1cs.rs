use super::{AllocatedPoint, BigNat};
use crate::{
  frontend::{ConstraintSystem, SynthesisError},
  gadgets::f_to_nat,
  r1cs::RelaxedR1CSInstance,
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// An Allocated Relaxed R1CS Instance with U.u as a BigNat
#[derive(Clone)]
pub struct AllocatedRelaxedR1CSInstanceBn<E: Engine, const N: usize> {
  pub(crate) W: AllocatedPoint<E::GE>,
  pub(crate) E: AllocatedPoint<E::GE>,
  pub(crate) u: BigNat<E::Base>,
  pub(crate) X: [BigNat<E::Base>; N],
}

impl<E: Engine, const N: usize> AllocatedRelaxedR1CSInstanceBn<E, N> {
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

    let u = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate u"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.u))),
      limb_width,
      n_limbs,
    )?;

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

    self.u.as_limbs().iter().enumerate().try_for_each(
      |(i, limb)| -> Result<(), SynthesisError> {
        ro.absorb(
          &limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of u to num")))?,
        );
        Ok(())
      },
    )?;

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
}
