use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::{
  errors::NovaError,
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
  spartan::{
    verify_circuit::{
      circuit::batched::SpartanVerifyCircuit, gadgets::nonnative::ipa::EvaluationEngineGadget,
    },
    PolyEvalInstance,
  },
  traits::{CurveCycleEquipped, Engine},
  CommitmentKey,
};

use super::AggregatorSNARKData;

fn ivc_aggregate<E1: CurveCycleEquipped>(snarks_data: &[AggregatorSNARKData<'_, E1>]) {}

#[derive(Clone)]
struct IOPCircuit<'a, E: Engine> {
  snark_data: &'a AggregatorSNARKData<'a, E>,
}

impl<'a, E: Engine> IOPCircuit<'a, E> {
  pub fn new(snark_data: &'a AggregatorSNARKData<'a, E>) -> Result<Self, NovaError> {
    Ok(Self { snark_data })
  }
}

impl<'a, E: Engine> IOPCircuit<'a, E> {
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    SpartanVerifyCircuit::synthesize(
      cs.namespace(|| "verify IOP"),
      &self.snark_data.vk,
      &self.snark_data.U,
      &self.snark_data.snark,
    )?;
    Ok(vec![])
  }
}

#[derive(Clone)]
struct FFACircuit<'a, E1: Engine> {
  snark_data: &'a AggregatorSNARKData<'a, E1>,
  arg: PolyEvalInstance<E1>,
}

impl<'a, E1: Engine> FFACircuit<'a, E1>
where
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  pub fn new(snark_data: &'a AggregatorSNARKData<'a, E1>) -> Result<Self, NovaError> {
    let arg = snark_data
      .snark
      .verify_execution_trace(&snark_data.vk, &snark_data.U)?;
    Ok(Self { snark_data, arg })
  }
}

impl<'a, E1> FFACircuit<'a, E1>
where
  E1: CurveCycleEquipped,
  <E1 as Engine>::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  fn synthesize<CS: ConstraintSystem<E1::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<Vec<AllocatedNum<E1::Base>>, SynthesisError> {
    let _ = EvaluationEngineGadget::<E1>::verify(
      cs.namespace(|| "EE::verify"),
      &self.snark_data.vk.vk_ee,
      &self.arg.c,
      &self.arg.x,
      &self.arg.e,
      &self.snark_data.snark.eval_arg,
    )
    .map_err(|_| SynthesisError::AssignmentMissing);
    Ok(vec![])
  }
}

fn build_circuits<'a, E1>(
  snarks_data: &'a [AggregatorSNARKData<'a, E1>],
) -> Result<Vec<(IOPCircuit<'a, E1>, FFACircuit<'a, E1>)>, NovaError>
where
  E1: CurveCycleEquipped,
  <E1 as Engine>::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  snarks_data
    .iter()
    .map(|snark_data| {
      let iop_circuit = IOPCircuit::new(snark_data)?;
      let ffa_circuit = FFACircuit::new(snark_data)?;
      Ok((iop_circuit, ffa_circuit))
    })
    .collect::<Result<_, NovaError>>()
}
