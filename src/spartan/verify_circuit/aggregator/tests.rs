use std::{fmt::Debug, marker::PhantomData};

use crate::{
  errors::NovaError,
  provider::{ipa_pc, pedersen::CommitmentKeyExtTrait, traits::DlogGroup, PallasEngine},
  spartan::{
    batched,
    snark::RelaxedR1CSSNARK,
    verify_circuit::{
      aggregator::{
        Aggregator, AggregatorPublicParams, AggregatorSNARKData, RecursiveAggregatedSNARK,
      },
      ipa_prover_poseidon,
    },
  },
  supernova::{
    snark::{CompressedSNARK, VerifierKey},
    utils::get_selector_vec_from_index,
    NonUniformCircuit, PublicParams, RecursiveSNARK, StepCircuit, TrivialSecondaryCircuit,
  },
  traits::{
    snark::{BatchedRelaxedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    CurveCycleEquipped, Dual, Engine,
  },
  CommitmentKey,
};
use abomonation::Abomonation;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use ff::PrimeField;
use itertools::Itertools as _;
use serde::Serialize;

use super::CompressedAggregatedSNARK;

type E1 = PallasEngine;
type E2 = Dual<E1>;
type EE1 = ipa_pc::EvaluationEngine<E1>;
type EE2 = ipa_pc::EvaluationEngine<E2>;
type AS1 = ipa_prover_poseidon::batched::BatchedRelaxedR1CSSNARK<E1>;
type S1 = RelaxedR1CSSNARK<E1, EE1>;
type S2 = RelaxedR1CSSNARK<E2, EE2>;

#[test]
fn test_ivc_aggregate() {
  let num_nodes = 3;
  let snarks_data = sim_nw(num_nodes);
  let snarks_data: Vec<AggregatorSNARKData<E1>> = snarks_data
    .iter()
    .map(|(snark, vk)| {
      let (snark, U) = snark.primary_snark_and_U();
      let vk = vk.primary();
      AggregatorSNARKData::new(snark, vk, U)
    })
    .collect();

  let snark = ivc_aggregate_with::<E1, S1, S2>(&snarks_data).unwrap();
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
  S1::ProverKey: Clone + Debug,
  S2::ProverKey: Clone + Debug,
  S2::VerifierKey: Serialize + Debug + Clone,
  S1::VerifierKey: Serialize + Debug + Clone,
{
  let circuits = Aggregator::build_circuits(snarks_data)?;
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

fn sim_nw(num_nodes: usize) -> Vec<(CompressedSNARK<E1, AS1, S2>, VerifierKey<E1, AS1, S2>)> {
  const NUM_STEPS: usize = 4;
  (0..num_nodes)
    .map(|_| test_compression_with::<_, _>(NUM_STEPS, BigTestCircuit::new))
    .collect()
}

fn test_compression_with<F, C>(
  num_steps: usize,
  circuits_factory: F,
) -> (CompressedSNARK<E1, AS1, S2>, VerifierKey<E1, AS1, S2>)
where
  C: NonUniformCircuit<E1, C1 = C, C2 = TrivialSecondaryCircuit<<Dual<E1> as Engine>::Scalar>>
    + StepCircuit<<E1 as Engine>::Scalar>,
  F: Fn(usize) -> Vec<C>,
{
  let secondary_circuit = TrivialSecondaryCircuit::default();
  let test_circuits = circuits_factory(num_steps);

  let pp = PublicParams::setup(
    &test_circuits[0],
    &*<AS1 as BatchedRelaxedR1CSSNARKTrait<E1>>::ck_floor(),
    &*<S2 as RelaxedR1CSSNARKTrait<E2>>::ck_floor(),
  );

  let z0_primary = vec![<E1 as Engine>::Scalar::from(17u64)];
  let z0_secondary = vec![<Dual<E1> as Engine>::Scalar::ZERO];

  let mut recursive_snark = RecursiveSNARK::new(
    &pp,
    &test_circuits[0],
    &test_circuits[0],
    &secondary_circuit,
    &z0_primary,
    &z0_secondary,
  )
  .unwrap();

  for circuit in test_circuits.iter().take(num_steps) {
    recursive_snark
      .prove_step(&pp, circuit, &secondary_circuit)
      .unwrap();

    recursive_snark
      .verify(&pp, &z0_primary, &z0_secondary)
      .unwrap();
  }

  let (prover_key, verifier_key) = CompressedSNARK::<_, AS1, S2>::setup(&pp).unwrap();

  let compressed_snark = CompressedSNARK::prove(&pp, &prover_key, &recursive_snark).unwrap();

  (compressed_snark, verifier_key)
}

#[derive(Clone)]
enum BigTestCircuit<E: Engine> {
  Square(SquareCircuit<E>),
  BigPower(BigPowerCircuit<E>),
}

impl<E: Engine> BigTestCircuit<E> {
  fn new(num_steps: usize) -> Vec<Self> {
    let mut circuits = Vec::new();

    for idx in 0..num_steps {
      if idx % 2 == 0 {
        circuits.push(Self::Square(SquareCircuit { _p: PhantomData }))
      } else {
        circuits.push(Self::BigPower(BigPowerCircuit { _p: PhantomData }))
      }
    }

    circuits
  }
}

impl<E: Engine> StepCircuit<E::Scalar> for BigTestCircuit<E> {
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    match self {
      Self::Square(c) => c.circuit_index(),
      Self::BigPower(c) => c.circuit_index(),
    }
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<E::Scalar>>,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<
    (
      Option<AllocatedNum<E::Scalar>>,
      Vec<AllocatedNum<E::Scalar>>,
    ),
    SynthesisError,
  > {
    match self {
      Self::Square(c) => c.synthesize(cs, pc, z),
      Self::BigPower(c) => c.synthesize(cs, pc, z),
    }
  }
}

impl<E1: CurveCycleEquipped> NonUniformCircuit<E1> for BigTestCircuit<E1> {
  type C1 = Self;
  type C2 = TrivialSecondaryCircuit<<Dual<E1> as Engine>::Scalar>;

  fn num_circuits(&self) -> usize {
    2
  }

  fn primary_circuit(&self, circuit_index: usize) -> Self {
    match circuit_index {
      0 => Self::Square(SquareCircuit { _p: PhantomData }),
      1 => Self::BigPower(BigPowerCircuit { _p: PhantomData }),
      _ => panic!("Invalid circuit index"),
    }
  }

  fn secondary_circuit(&self) -> Self::C2 {
    Default::default()
  }
}

#[derive(Clone)]
struct SquareCircuit<E> {
  _p: PhantomData<E>,
}

impl<E: Engine> StepCircuit<E::Scalar> for SquareCircuit<E> {
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _pc: Option<&AllocatedNum<E::Scalar>>,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<
    (
      Option<AllocatedNum<E::Scalar>>,
      Vec<AllocatedNum<E::Scalar>>,
    ),
    SynthesisError,
  > {
    let z_i = &z[0];

    let z_next = z_i.square(cs.namespace(|| "z_i^2"))?;

    let next_pc = AllocatedNum::alloc(cs.namespace(|| "next_pc"), || Ok(E::Scalar::from(1u64)))?;

    cs.enforce(
      || "next_pc = 1",
      |lc| lc + CS::one(),
      |lc| lc + next_pc.get_variable(),
      |lc| lc + CS::one(),
    );

    Ok((Some(next_pc), vec![z_next]))
  }
}

#[derive(Clone)]
struct BigPowerCircuit<E> {
  _p: PhantomData<E>,
}

impl<E: Engine> StepCircuit<E::Scalar> for BigPowerCircuit<E> {
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    _pc: Option<&AllocatedNum<E::Scalar>>,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<
    (
      Option<AllocatedNum<E::Scalar>>,
      Vec<AllocatedNum<E::Scalar>>,
    ),
    SynthesisError,
  > {
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..10_000 {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }

    let next_pc = AllocatedNum::alloc(cs.namespace(|| "next_pc"), || Ok(E::Scalar::from(0u64)))?;

    cs.enforce(
      || "next_pc = 0",
      |lc| lc + CS::one(),
      |lc| lc + next_pc.get_variable(),
      |lc| lc,
    );

    Ok((Some(next_pc), vec![y]))
  }
}
