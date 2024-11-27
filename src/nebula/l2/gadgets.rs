use crate::traits::commitment::CommitmentTrait;
use crate::{
  cyclefold::gadgets::emulated::{AllocatedEmulPoint, AllocatedEmulRelaxedR1CSInstance},
  traits::{CurveCycleEquipped, Dual, Engine},
};
use bellpepper_core::{ConstraintSystem, SynthesisError};

use super::utils::RelaxedFoldingData;

/// The in-circuit representation of the primary folding data.
pub struct AllocatedRelaxedFoldingData<E: Engine> {
  pub(crate) U1: AllocatedEmulRelaxedR1CSInstance<E>,
  pub(crate) U2: AllocatedEmulRelaxedR1CSInstance<E>,
  pub(crate) T: AllocatedEmulPoint<E::GE>,
}

impl<E: Engine> AllocatedRelaxedFoldingData<E> {
  pub(crate) fn alloc<CS, E2>(
    mut cs: CS,
    inst: Option<&RelaxedFoldingData<E2>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<<E as Engine>::Base>,
    E2: Engine<Base = E::Scalar, Scalar = E::Base>,
  {
    let U1 = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      inst.map(|inst| &inst.U1),
      limb_width,
      n_limbs,
    )?;

    let U2 = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      inst.map(|inst| &inst.U2),
      limb_width,
      n_limbs,
    )?;

    let T = AllocatedEmulPoint::alloc(
      cs.namespace(|| "allocate T"),
      inst.map(|inst| inst.T.to_coordinates()),
      limb_width,
      n_limbs,
    )?;

    Ok(Self { U1, U2, T })
  }
}
