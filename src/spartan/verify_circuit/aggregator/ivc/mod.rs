use std::marker::PhantomData;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::{
  errors::NovaError,
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
  r1cs::CommitmentKeyHint,
  spartan::{
    verify_circuit::{
      circuit::batched::SpartanVerifyCircuit, gadgets::nonnative::ipa::EvaluationEngineGadget,
    },
    PolyEvalInstance,
  },
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::RelaxedR1CSSNARKTrait,
    CurveCycleEquipped, Dual, Engine,
  },
  CommitmentKey, CompressedSNARK, PublicParams, RecursiveSNARK, StepCounterType,
};
use ff::Field;

use super::AggregatorSNARKData;

#[cfg(test)]
mod tests;

pub struct Aggregator;

impl Aggregator {
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
}

pub struct AggregatorPublicParams<E1>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
{
  pp_iop: PublicParams<E1>,
  pp_ffa: PublicParams<Dual<E1>>,
}

impl<E1> AggregatorPublicParams<E1>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
{
  pub fn setup(
    circuit_iop: &IOPCircuit<E1>,
    circuit_ffa: &FFACircuit<E1>,
    ck_hint1: &CommitmentKeyHint<E1>,
    ck_hint2: &CommitmentKeyHint<Dual<E1>>,
  ) -> Result<AggregatorPublicParams<E1>, NovaError>
  where
    E1::GE: DlogGroup,
    CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
  {
    let trivial_circuit_secondary = TrivialCircuit::<E1::Base>::default();
    let trivial_circuit_primary = TrivialCircuit::<E1::Scalar>::default();
    Ok(AggregatorPublicParams {
      pp_iop: PublicParams::setup(circuit_iop, &trivial_circuit_secondary, ck_hint1, ck_hint2)?,
      pp_ffa: PublicParams::<Dual<E1>>::setup(
        circuit_ffa,
        &trivial_circuit_primary,
        ck_hint2,
        ck_hint1,
      )?,
    })
  }
  pub fn iop(&self) -> &PublicParams<E1> {
    &self.pp_iop
  }

  pub fn ffa(&self) -> &PublicParams<Dual<E1>> {
    &self.pp_ffa
  }
}

pub struct RecursiveAggregatedSNARK<E1>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
{
  rs_iop: RecursiveSNARK<E1>,
  rs_ffa: RecursiveSNARK<Dual<E1>>,
}

impl<E1> RecursiveAggregatedSNARK<E1>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  pub fn new(
    pp: &AggregatorPublicParams<E1>,
    iop_circuit: &IOPCircuit<E1>,
    ffa_circuit: &FFACircuit<E1>,
  ) -> Result<Self, NovaError> {
    let rs_iop = RecursiveSNARK::new(
      pp.iop(),
      iop_circuit,
      &TrivialCircuit::default(),
      &[<E1 as Engine>::Scalar::ZERO],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )?;

    let rs_ffa = RecursiveSNARK::new(
      pp.ffa(),
      ffa_circuit,
      &TrivialCircuit::default(),
      &[<Dual<E1> as Engine>::Scalar::ZERO],
      &[<E1 as Engine>::Scalar::ZERO],
    )?;

    Ok(Self { rs_iop, rs_ffa })
  }

  pub fn prove_step(
    &mut self,
    pp: &AggregatorPublicParams<E1>,
    iop_circuit: &IOPCircuit<E1>,
    ffa_circuit: &FFACircuit<E1>,
  ) -> Result<(), NovaError> {
    self
      .rs_iop
      .prove_step(pp.iop(), iop_circuit, &TrivialCircuit::default())?;

    self
      .rs_ffa
      .prove_step(pp.ffa(), ffa_circuit, &TrivialCircuit::default())?;

    Ok(())
  }

  pub fn verify(&self, pp: &AggregatorPublicParams<E1>, num_steps: usize) -> Result<(), NovaError> {
    self.rs_iop.verify(
      pp.iop(),
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    );

    self.rs_iop.verify(
      pp.iop(),
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    );

    Ok(())
  }
}

pub struct AggregatedSNARK<E1, S1, S2> {
  _p: PhantomData<(E1, S1, S2)>,
}

impl<E1, S1, S2> AggregatedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  fn prove(
    pp: &AggregatorPublicParams<E1>,
    init: Option<RecursiveAggregatedSNARK<E1>>,
    steps: Vec<(IOPCircuit<E1>, FFACircuit<E1>)>,
  ) {
  }
}

pub fn ivc_aggregate_with<E1, S1, S2>(
  snarks_data: &[AggregatorSNARKData<'_, E1>],
) -> Result<CompressedSNARK<E1, S1, S2>, NovaError>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  let circuits = build_circuits(snarks_data)?;
  let num_steps = circuits.len();
  let trivial_circuit_secondary = TrivialCircuit::default();
  let trivial_circuit_primary = TrivialCircuit::<E1::Scalar>::default();

  let (pp_iop, pp_ffa) = {
    let (init_circuit_iop, init_circuit_ffa) = &circuits[0];
    (
      PublicParams::setup(
        init_circuit_iop,
        &trivial_circuit_secondary,
        &*S1::ck_floor(),
        &*S2::ck_floor(),
      )?,
      PublicParams::<Dual<E1>>::setup(
        init_circuit_ffa,
        &trivial_circuit_primary,
        &*S2::ck_floor(),
        &*S1::ck_floor(),
      )?,
    )
  };

  let mut rs_option_iop: Option<RecursiveSNARK<E1>> = None;
  let mut rs_option_ffa: Option<RecursiveSNARK<Dual<E1>>> = None;
  for (iop_circuit, ffa_circuit) in circuits.iter() {
    let mut rs_iop = rs_option_iop.unwrap_or_else(|| {
      RecursiveSNARK::new(
        &pp_iop,
        iop_circuit,
        &trivial_circuit_secondary,
        &[<E1 as Engine>::Scalar::ZERO],
        &[<Dual<E1> as Engine>::Scalar::ZERO],
      )
      .unwrap()
    });

    rs_iop.prove_step(&pp_iop, iop_circuit, &TrivialCircuit::default())?;

    rs_option_iop = Some(rs_iop);

    let mut rs_ffa = rs_option_ffa.unwrap_or_else(|| {
      RecursiveSNARK::new(
        &pp_ffa,
        ffa_circuit,
        &trivial_circuit_primary,
        &[<Dual<E1> as Engine>::Scalar::ZERO],
        &[<E1 as Engine>::Scalar::ZERO],
      )
      .unwrap()
    });

    rs_ffa.prove_step(&pp_ffa, ffa_circuit, &TrivialCircuit::default())?;

    rs_option_ffa = Some(rs_ffa);
  }

  assert!(rs_option_iop.is_some());
  let rs_iop = rs_option_iop.ok_or(NovaError::UnSat)?;
  rs_iop.verify(
    &pp_iop,
    num_steps,
    &[<E1 as Engine>::Scalar::ZERO],
    &[<Dual<E1> as Engine>::Scalar::ZERO],
  )?;

  let (pk_iop, vk_iop) = CompressedSNARK::<_, S1, S2>::setup(&pp_iop)?;
  let snark_iop = CompressedSNARK::<_, S1, S2>::prove(&pp_iop, &pk_iop, &rs_iop)?;

  assert!(rs_option_ffa.is_some());
  let rs_ffa = rs_option_ffa.ok_or(NovaError::UnSat)?;

  rs_ffa.verify(
    &pp_ffa,
    num_steps,
    &[<Dual<E1> as Engine>::Scalar::ZERO],
    &[<E1 as Engine>::Scalar::ZERO],
  )?;

  let (pk_ffa, vk_ffa) = CompressedSNARK::<_, S2, S1>::setup(&pp_ffa)?;
  let snark_ffa = CompressedSNARK::<_, S2, S1>::prove(&pp_ffa, &pk_ffa, &rs_ffa)?;

  snark_iop.verify(
    &vk_iop,
    num_steps,
    &[<E1 as Engine>::Scalar::ZERO],
    &[<Dual<E1> as Engine>::Scalar::ZERO],
  )?;

  snark_ffa.verify(
    &vk_ffa,
    num_steps,
    &[<Dual<E1> as Engine>::Scalar::ZERO],
    &[<E1 as Engine>::Scalar::ZERO],
  )?;

  Ok(snark_iop)
}

pub fn ivc_aggregate_with2<E1, S1, S2>(
  snarks_data: &[AggregatorSNARKData<'_, E1>],
) -> Result<(), NovaError>
// -> Result<CompressedSNARK<E1, S1, S2>, NovaError>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  let circuits = build_circuits(snarks_data)?;
  let num_steps = circuits.len();
  let (init_circuit_iop, init_circuit_ffa) = &circuits[0];

  let pp = AggregatorPublicParams::setup(
    init_circuit_iop,
    init_circuit_ffa,
    &*S1::ck_floor(),
    &*S2::ck_floor(),
  )?;

  let mut rs_option: Option<RecursiveAggregatedSNARK<E1>> = None;

  for (iop_circuit, ffa_circuit) in circuits.iter() {
    let mut rs = rs_option
      .unwrap_or_else(|| RecursiveAggregatedSNARK::new(&pp, iop_circuit, ffa_circuit).unwrap());

    rs.prove_step(&pp, iop_circuit, ffa_circuit)?;
    rs_option = Some(rs)
  }

  debug_assert!(rs_option.is_some());
  let rs_iop = rs_option.ok_or(NovaError::UnSat)?;
  rs_iop.verify(&pp, num_steps)?;

  // let (pk_iop, vk_iop) = CompressedSNARK::<_, S1, S2>::setup(&pp_iop)?;
  // let snark_iop = CompressedSNARK::<_, S1, S2>::prove(&pp_iop, &pk_iop, &rs_iop)?;

  // assert!(rs_option_ffa.is_some());
  // let rs_ffa = rs_option_ffa.ok_or(NovaError::UnSat)?;

  // rs_ffa.verify(
  //   &pp_ffa,
  //   num_steps,
  //   &[<Dual<E1> as Engine>::Scalar::ZERO],
  //   &[<E1 as Engine>::Scalar::ZERO],
  // )?;

  // let (pk_ffa, vk_ffa) = CompressedSNARK::<_, S2, S1>::setup(&pp_ffa)?;
  // let snark_ffa = CompressedSNARK::<_, S2, S1>::prove(&pp_ffa, &pk_ffa, &rs_ffa)?;

  // snark_iop.verify(
  //   &vk_iop,
  //   num_steps,
  //   &[<E1 as Engine>::Scalar::ZERO],
  //   &[<Dual<E1> as Engine>::Scalar::ZERO],
  // )?;

  // snark_ffa.verify(
  //   &vk_ffa,
  //   num_steps,
  //   &[<Dual<E1> as Engine>::Scalar::ZERO],
  //   &[<E1 as Engine>::Scalar::ZERO],
  // )?;

  // Ok(snark_iop)
  Ok(())
}

#[derive(Clone)]
struct IOPCircuit<'a, E: Engine> {
  snark_data: &'a AggregatorSNARKData<'a, E>,
}

impl<'a, E: Engine> IOPCircuit<'a, E> {
  pub fn new(snark_data: &'a AggregatorSNARKData<'a, E>) -> Result<Self, NovaError> {
    Ok(Self { snark_data })
  }
}

impl<'a, E: Engine> StepCircuit<E::Scalar> for IOPCircuit<'a, E> {
  fn arity(&self) -> usize {
    1
  }
  fn get_counter_type(&self) -> StepCounterType {
    StepCounterType::Incremental
  }
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    SpartanVerifyCircuit::synthesize(
      cs.namespace(|| "verify IOP"),
      &self.snark_data.vk,
      &self.snark_data.U,
      &self.snark_data.snark,
    )?;
    Ok(z.to_vec())
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

impl<'a, E1> StepCircuit<E1::Base> for FFACircuit<'a, E1>
where
  E1: CurveCycleEquipped,
  <E1 as Engine>::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  fn arity(&self) -> usize {
    1
  }
  fn get_counter_type(&self) -> StepCounterType {
    StepCounterType::Incremental
  }
  fn synthesize<CS: ConstraintSystem<E1::Base>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E1::Base>],
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
    Ok(z.to_vec())
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
