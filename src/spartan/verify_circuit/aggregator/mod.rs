use std::marker::PhantomData;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::{
  errors::NovaError,
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
  r1cs::{CommitmentKeyHint, RelaxedR1CSInstance},
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
  CommitmentKey, CompressedSNARK, ProverKey, PublicParams, RecursiveSNARK, StepCounterType,
  VerifierKey,
};
use ff::Field;

use super::ipa_prover_poseidon::batched;

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

    self.rs_ffa.verify(
      pp.ffa(),
      num_steps,
      &[<Dual<E1> as Engine>::Scalar::ZERO],
      &[<E1 as Engine>::Scalar::ZERO],
    );

    Ok(())
  }

  pub fn iop(&self) -> &RecursiveSNARK<E1> {
    &self.rs_iop
  }

  pub fn ffa(&self) -> &RecursiveSNARK<Dual<E1>> {
    &self.rs_ffa
  }
}

pub struct AggregatedProverKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  pk_iop: ProverKey<E1, S1, S2>,
  pk_ffa: ProverKey<Dual<E1>, S2, S1>,
}

impl<E1, S1, S2> AggregatedProverKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  fn new(pk_iop: ProverKey<E1, S1, S2>, pk_ffa: ProverKey<Dual<E1>, S2, S1>) -> Self {
    Self { pk_iop, pk_ffa }
  }

  fn iop(&self) -> &ProverKey<E1, S1, S2> {
    &self.pk_iop
  }

  fn ffa(&self) -> &ProverKey<Dual<E1>, S2, S1> {
    &self.pk_ffa
  }
}

pub struct AggregatedVerifierKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  vk_iop: VerifierKey<E1, S1, S2>,
  vk_ffa: VerifierKey<Dual<E1>, S2, S1>,
}

impl<E1, S1, S2> AggregatedVerifierKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  fn new(vk_iop: VerifierKey<E1, S1, S2>, vk_ffa: VerifierKey<Dual<E1>, S2, S1>) -> Self {
    Self { vk_iop, vk_ffa }
  }

  fn iop(&self) -> &VerifierKey<E1, S1, S2> {
    &self.vk_iop
  }

  fn ffa(&self) -> &VerifierKey<Dual<E1>, S2, S1> {
    &self.vk_ffa
  }
}

pub struct CompressedAggregatedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  snark_iop: CompressedSNARK<E1, S1, S2>,
  snark_ffa: CompressedSNARK<Dual<E1>, S2, S1>,
}

impl<E1, S1, S2> CompressedAggregatedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  Dual<E1>: CurveCycleEquipped<Secondary = E1>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  fn setup(
    pp: &AggregatorPublicParams<E1>,
  ) -> Result<
    (
      AggregatedProverKey<E1, S1, S2>,
      AggregatedVerifierKey<E1, S1, S2>,
    ),
    NovaError,
  > {
    let (pk_iop, vk_iop) = CompressedSNARK::<_, S1, S2>::setup(pp.iop())?;
    let (pk_ffa, vk_ffa) = CompressedSNARK::<_, S2, S1>::setup(pp.ffa())?;

    let pk = AggregatedProverKey::new(pk_iop, pk_ffa);
    let vk = AggregatedVerifierKey::new(vk_iop, vk_ffa);

    Ok((pk, vk))
  }

  fn prove(
    pp: &AggregatorPublicParams<E1>,
    pk: &AggregatedProverKey<E1, S1, S2>,
    vk: &AggregatedVerifierKey<E1, S1, S2>,
    rs: &RecursiveAggregatedSNARK<E1>,
  ) -> Result<Self, NovaError> {
    let snark_iop = CompressedSNARK::<_, S1, S2>::prove(pp.iop(), pk.iop(), rs.iop())?;
    let snark_ffa = CompressedSNARK::<_, S2, S1>::prove(pp.ffa(), pk.ffa(), rs.ffa())?;

    Ok(Self {
      snark_iop,
      snark_ffa,
    })
  }

  fn verify(
    &self,
    vk: &AggregatedVerifierKey<E1, S1, S2>,
    num_steps: usize,
  ) -> Result<(), NovaError> {
    self.snark_iop.verify(
      vk.iop(),
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )?;
    self.snark_ffa.verify(
      vk.ffa(),
      num_steps,
      &[<Dual<E1> as Engine>::Scalar::ZERO],
      &[<E1 as Engine>::Scalar::ZERO],
    )?;

    Ok(())
  }
}

pub fn ivc_aggregate_with<E1, S1, S2>(
  snarks_data: &[AggregatorSNARKData<'_, E1>],
) -> Result<CompressedAggregatedSNARK<E1, S1, S2>, NovaError>
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

  let (pk, vk) = CompressedAggregatedSNARK::<E1, S1, S2>::setup(&pp)?;
  let snark = CompressedAggregatedSNARK::<E1, S1, S2>::prove(&pp, &pk, &vk, &rs_iop)?;

  snark.verify(&vk, num_steps)?;
  Ok((snark))
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

/// Data structure that holds the required data needed for proof aggregation
pub struct AggregatorSNARKData<'a, E: Engine> {
  snark: batched::BatchedRelaxedR1CSSNARK<E>,
  vk: &'a batched::VerifierKey<E>,
  U: Vec<RelaxedR1CSInstance<E>>,
}

impl<'a, E: Engine> AggregatorSNARKData<'a, E> {
  /// Create a new instance of `AggregatorSNARKData`
  pub fn new(
    snark: batched::BatchedRelaxedR1CSSNARK<E>,
    vk: &'a batched::VerifierKey<E>,
    U: Vec<RelaxedR1CSInstance<E>>,
  ) -> Self {
    Self { snark, vk, U }
  }
}
