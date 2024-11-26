use abomonation_derive::Abomonation;
use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use serde::{Deserialize, Serialize};

use crate::constants::{NUM_FE_IN_EMULATED_POINT, NUM_HASH_BITS};
use crate::gadgets::{alloc_scalar_as_base, le_bits_to_num};
use crate::nebula::augmented_circuit::AugmentedCircuitParams;
use crate::nebula::rs::StepCircuit;
use crate::traits::commitment::CommitmentTrait;
use crate::traits::ROCircuitTrait;
use crate::{
  cyclefold::gadgets::emulated,
  traits::{CurveCycleEquipped, Dual, Engine, ROConstantsCircuit},
  Commitment,
};

use crate::nebula::layer2::gadgets::AllocatedRelaxedFoldingData;
use crate::nebula::layer2::utils::RelaxedFoldingData;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Abomonation)]
pub struct FinalCircuitParams {
  limb_width: usize,
  n_limbs: usize,
}

impl From<&AugmentedCircuitParams> for FinalCircuitParams {
  fn from(value: &AugmentedCircuitParams) -> Self {
    Self {
      limb_width: value.limb_width,
      n_limbs: value.n_limbs,
    }
  }
}

impl FinalCircuitParams {
  pub const fn new(limb_width: usize, n_limbs: usize) -> Self {
    Self {
      limb_width,
      n_limbs,
    }
  }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct FinalCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  pp_digest: E1::Base,

  data_F: RelaxedFoldingData<E1>,

  E_F: Commitment<E1>,
  W_F: Commitment<E1>,
}

impl<E1> FinalCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    pp_digest: E1::Base,
    data_F: RelaxedFoldingData<E1>,
    E_F: Commitment<E1>,
    W_F: Commitment<E1>,
  ) -> Self {
    Self {
      pp_digest,
      data_F,
      E_F,
      W_F,
    }
  }
}

#[derive(Clone)]
pub struct FinalCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  params: &'a FinalCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E1>>,
  inputs: FinalCircuitInputs<E1>,
}

impl<'a, E1> FinalCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    params: &'a FinalCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E1>>,
    inputs: FinalCircuitInputs<E1>,
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
    let pp_digest =
      alloc_scalar_as_base::<Dual<E1>, _>(cs.namespace(|| "params"), Some(self.inputs.pp_digest))?;

    let data_F = AllocatedRelaxedFoldingData::alloc(
      cs.namespace(|| "data_F"),
      &self.inputs.data_F,
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let E_new = emulated::AllocatedEmulPoint::alloc(
      cs.namespace(|| "E_new"),
      Some(self.inputs.E_F.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let W_new = emulated::AllocatedEmulPoint::alloc(
      cs.namespace(|| "W_new"),
      Some(self.inputs.W_F.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    Ok((pp_digest, data_F, E_new, W_new))
  }
}

impl<'a, E1> StepCircuit<E1::Scalar> for FinalCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  fn arity(&self) -> usize {
    1
  }

  fn non_deterministic_advice(&self) -> Vec<E1::Scalar> {
    vec![]
  }

  fn synthesize<CS: ConstraintSystem<E1::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E1::Scalar>],
  ) -> Result<Vec<AllocatedNum<E1::Scalar>>, SynthesisError> {
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

    hash.inputize(cs.namespace(|| "inputize hash"))?;
    Ok(z.to_vec())
  }
}
