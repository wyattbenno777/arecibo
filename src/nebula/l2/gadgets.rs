use crate::constants::{NUM_FE_IN_EMULATED_POINT, NUM_HASH_BITS};
use crate::cyclefold::gadgets::emulated;
use crate::gadgets::{alloc_scalar_as_base, le_bits_to_num};
use crate::nebula::augmented_circuit::AugmentedCircuitParams;
use crate::traits::commitment::CommitmentTrait;
use crate::traits::ROCircuitTrait;
use crate::traits::{CurveCycleEquipped, Dual, ROConstantsCircuit};
use crate::Commitment;
use crate::{
  cyclefold::gadgets::emulated::{AllocatedEmulPoint, AllocatedEmulRelaxedR1CSInstance},
  traits::Engine,
};
use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct NIFSVerifierCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  pp_digest: Option<E1::Base>,

  data_F: Option<RelaxedFoldingData<E1>>,

  E_F: Option<Commitment<E1>>,
  W_F: Option<Commitment<E1>>,
}

impl<E1> NIFSVerifierCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    pp_digest: Option<E1::Base>,
    data_F: Option<RelaxedFoldingData<E1>>,
    E_F: Option<Commitment<E1>>,
    W_F: Option<Commitment<E1>>,
  ) -> Self {
    Self {
      pp_digest,
      data_F,
      E_F,
      W_F,
    }
  }
}

/// Folding verifier as a gadget
#[derive(Clone)]
pub struct NIFSVerifierCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  params: &'a AugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E1>>,
  inputs: Option<&'a NIFSVerifierCircuitInputs<E1>>,
}

impl<'a, E1> NIFSVerifierCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    params: &'a AugmentedCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E1>>,
    inputs: Option<&'a NIFSVerifierCircuitInputs<E1>>,
  ) -> Self {
    Self {
      params,
      ro_consts,
      inputs,
    }
  }

  fn alloc_witness<CS: ConstraintSystem<<E1 as Engine>::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<
    (
      AllocatedNum<E1::Scalar>,                               // pp_digest
      AllocatedRelaxedFoldingData<Dual<E1>>,                  //data_F
      emulated::AllocatedEmulPoint<<Dual<E1> as Engine>::GE>, // E_new
      emulated::AllocatedEmulPoint<<Dual<E1> as Engine>::GE>, // W_new
    ),
    SynthesisError,
  > {
    let pp_digest = alloc_scalar_as_base::<Dual<E1>, _>(
      cs.namespace(|| "params"),
      self.inputs.as_ref().and_then(|inputs| inputs.pp_digest),
    )?;

    let data_F = AllocatedRelaxedFoldingData::alloc(
      cs.namespace(|| "data_F"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.data_F.as_ref()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let E_new = emulated::AllocatedEmulPoint::alloc(
      cs.namespace(|| "E_new"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.E_F)
        .map(|E| E.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let W_new = emulated::AllocatedEmulPoint::alloc(
      cs.namespace(|| "W_new"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.W_F)
        .map(|W| W.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    Ok((pp_digest, data_F, E_new, W_new))
  }

  pub fn synthesize<CS: ConstraintSystem<E1::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedNum<E1::Scalar>, SynthesisError> {
    // Allocate the witness
    let (pp_digest, data_F, E_new, W_new) = self.alloc_witness(cs.namespace(|| "alloc_witness"))?;

    let U_F = data_F.U1.fold_with_relaxed_r1cs(
      cs.namespace(|| "fold U2 into U1"),
      &pp_digest,
      &data_F.U2,
      W_new,
      E_new,
      &data_F.T,
      self.ro_consts.clone(),
    )?;

    // Calculate h_int = H(U_F)
    let mut ro = <Dual<E1> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      2 * NUM_FE_IN_EMULATED_POINT + 2 + 1, // U.comm_W + U.comm_E + U.X + U.u
    );

    U_F.absorb_in_ro(cs.namespace(|| "absorb U_F"), &mut ro)?;

    let hash_bits = ro.squeeze(cs.namespace(|| "hash_bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "hash"), &hash_bits)?;

    Ok(hash)
  }
}
