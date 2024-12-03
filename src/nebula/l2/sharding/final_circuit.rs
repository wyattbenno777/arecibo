use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use serde::{Deserialize, Serialize};

use crate::nebula::l2::gadgets::{NIFSVerifierCircuit, NIFSVerifierCircuitInputs};
use crate::{
  nebula::{augmented_circuit::AugmentedCircuitParams, rs::StepCircuit},
  traits::{CurveCycleEquipped, Dual, ROConstantsCircuit},
};
use bellpepper::gadgets::Assignment;
use ff::Field;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct FinalCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  inputs_F: Option<NIFSVerifierCircuitInputs<E1>>,
  inputs_ops: Option<NIFSVerifierCircuitInputs<E1>>,
  inputs_scan: Option<NIFSVerifierCircuitInputs<E1>>,
  C_IS: Option<E1::Scalar>,
  C_FS: Option<E1::Scalar>,
}

impl<E1> FinalCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    inputs_F: Option<NIFSVerifierCircuitInputs<E1>>,
    inputs_ops: Option<NIFSVerifierCircuitInputs<E1>>,
    inputs_scan: Option<NIFSVerifierCircuitInputs<E1>>,
    C_IS: Option<E1::Scalar>,
    C_FS: Option<E1::Scalar>,
  ) -> Self {
    Self {
      inputs_F,
      inputs_ops,
      inputs_scan,
      C_IS,
      C_FS,
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
    4
  }

  fn non_deterministic_advice(&self) -> Vec<E1::Scalar> {
    vec![]
  }

  fn synthesize<CS: ConstraintSystem<E1::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E1::Scalar>],
  ) -> Result<Vec<AllocatedNum<E1::Scalar>>, SynthesisError> {
    let C_i = z[0].clone();

    let C_IS = AllocatedNum::alloc(cs.namespace(|| "prev_IC"), || {
      Ok(
        *self
          .inputs
          .get()?
          .C_IS
          .as_ref()
          .unwrap_or(&E1::Scalar::ZERO),
      )
    })?;

    let C_FS = AllocatedNum::alloc(cs.namespace(|| "prev_IC"), || {
      Ok(
        *self
          .inputs
          .get()?
          .C_FS
          .as_ref()
          .unwrap_or(&E1::Scalar::ZERO),
      )
    })?;

    // 3. check that Ci =? CIS // finalized proof starts with previous memory
    cs.enforce(
      || "C_i = C_IS",
      |lc| lc + C_i.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + C_IS.get_variable(),
    );

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

    Ok(vec![C_FS, hash_F, hash_ops, hash_scan])
  }
}
