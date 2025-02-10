use super::StepCircuit;
use crate::frontend::gadgets::Assignment;
use crate::gadgets::{alloc_num_equals, alloc_zero};
use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::alloc_scalar_as_base,
  traits::{CurveCycleEquipped, Dual, ROConstantsCircuit},
  AugmentedCircuitParams,
};
use ff::Field;
use serde::{Deserialize, Serialize};

pub struct AugmentedCircuit<'a, E, SC>
where
  SC: StepCircuit<E::Scalar>,
  E: CurveCycleEquipped,
{
  step_circuit: &'a SC,
  params: &'a AugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E>>,
  inputs: Option<AugmentedCircuitInputs<E>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AugmentedCircuitInputs<E>
where
  E: CurveCycleEquipped,
{
  pp_digest: E::Base,
  i: E::Scalar,
  z0: Vec<E::Scalar>,
  zi: Option<Vec<E::Scalar>>,
}

impl<E> AugmentedCircuitInputs<E>
where
  E: CurveCycleEquipped,
{
  pub fn new(
    pp_digest: E::Base,
    i: E::Scalar,
    z0: Vec<E::Scalar>,
    zi: Option<Vec<E::Scalar>>,
  ) -> Self {
    Self {
      pp_digest,
      i,
      z0,
      zi,
    }
  }
}

impl<'a, E, SC> AugmentedCircuit<'a, E, SC>
where
  E: CurveCycleEquipped,
  SC: StepCircuit<E::Scalar>,
{
  pub const fn new(
    params: &'a AugmentedCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    inputs: Option<AugmentedCircuitInputs<E>>,
    step_circuit: &'a SC,
  ) -> Self {
    Self {
      params,
      ro_consts,
      inputs,
      step_circuit,
    }
  }
  fn alloc_witness<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,      // pp_digest
      AllocatedNum<E::Scalar>,      // i
      Vec<AllocatedNum<E::Scalar>>, // z0
      Vec<AllocatedNum<E::Scalar>>, // zi
    ),
    SynthesisError,
  > {
    let pp_digest = alloc_scalar_as_base::<Dual<E>, _>(
      cs.namespace(|| "params"),
      self.inputs.as_ref().map(|inputs| inputs.pp_digest),
    )?;
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;
    let z_0 = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("z0_{i}")), || {
          Ok(self.inputs.get()?.z0[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero_vec = vec![E::Scalar::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero_vec)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?;
    Ok((pp_digest, i, z_0, z_i))
  }

  pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // Allocate the witness
    let arity = self.step_circuit.arity();
    let (pp_digest, i, z_0, z_i) = self.alloc_witness(cs.namespace(|| "alloc_witness"), arity)?;
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "is base case"), &i, &zero)?;
    todo!()
  }
}
