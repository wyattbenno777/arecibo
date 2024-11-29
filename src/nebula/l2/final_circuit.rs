use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use serde::{Deserialize, Serialize};

use crate::{
  nebula::{augmented_circuit::AugmentedCircuitParams, rs::StepCircuit},
  traits::{CurveCycleEquipped, Dual, ROConstantsCircuit},
};

use super::gadgets::{NIFSVerifierCircuit, NIFSVerifierCircuitInputs};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct FinalCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  inputs_F: Option<NIFSVerifierCircuitInputs<E1>>,
  inputs_ops: Option<NIFSVerifierCircuitInputs<E1>>,
  inputs_scan: Option<NIFSVerifierCircuitInputs<E1>>,
}

impl<E1> FinalCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    inputs_F: Option<NIFSVerifierCircuitInputs<E1>>,
    inputs_ops: Option<NIFSVerifierCircuitInputs<E1>>,
    inputs_scan: Option<NIFSVerifierCircuitInputs<E1>>,
  ) -> Self {
    Self {
      inputs_F,
      inputs_ops,
      inputs_scan,
    }
  }
}

#[derive(Clone)]
pub struct FinalCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  params: &'a AugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E1>>,
  inputs: Option<FinalCircuitInputs<E1>>,
}

impl<'a, E1> FinalCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    params: &'a AugmentedCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E1>>,
    inputs: Option<FinalCircuitInputs<E1>>,
  ) -> Self {
    Self {
      params,
      ro_consts,
      inputs,
    }
  }
}

impl<'a, E1> StepCircuit<E1::Scalar> for FinalCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  fn arity(&self) -> usize {
    3
  }

  fn non_deterministic_advice(&self) -> Vec<E1::Scalar> {
    vec![]
  }

  fn synthesize<CS: ConstraintSystem<E1::Scalar>>(
    &self,
    cs: &mut CS,
    _z: &[AllocatedNum<E1::Scalar>],
  ) -> Result<Vec<AllocatedNum<E1::Scalar>>, SynthesisError> {
    let hash_F = NIFSVerifierCircuit::new(
      self.params,
      self.ro_consts.clone(),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.inputs_F.as_ref()),
    )
    .synthesize(cs.namespace(|| "fold U_ops"))?;

    let hash_ops = NIFSVerifierCircuit::new(
      self.params,
      self.ro_consts.clone(),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.inputs_ops.as_ref()),
    )
    .synthesize(cs.namespace(|| "fold U_scan"))?;

    let hash_scan = NIFSVerifierCircuit::new(
      self.params,
      self.ro_consts.clone(),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.inputs_scan.as_ref()),
    )
    .synthesize(cs.namespace(|| "fold U_F"))?;

    Ok(vec![hash_F, hash_ops, hash_scan])
  }
}
