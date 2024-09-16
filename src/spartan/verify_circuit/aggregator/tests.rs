use crate::provider::ipa_pc;

use crate::provider::PallasEngine;

use crate::r1cs::RelaxedR1CSInstance;
use crate::r1cs::RelaxedR1CSWitness;
use crate::spartan;

use crate::spartan::verify_circuit::aggregator::AggregatedSNARK;
use crate::spartan::verify_circuit::aggregator::AggregatorSNARKData;
use crate::spartan::verify_circuit::ipa_prover_poseidon;
use crate::supernova::circuit::{StepCircuit, TrivialSecondaryCircuit};
use crate::traits::snark::default_ck_hint;
use crate::traits::snark::BatchedRelaxedR1CSSNARKTrait;
use crate::traits::CurveCycleEquipped;
use crate::traits::Dual;
use crate::traits::Engine;

use bellpepper_core::num::AllocatedNum;

use bellpepper_core::{ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use error::SuperNovaError;

use ff::Field;
use ff::PrimeField;
use itertools::Itertools as _;

use crate::supernova::{utils::get_selector_vec_from_index, *};

use super::PublicParams as AggPublicParams;

type E1 = PallasEngine;
type E2 = Dual<PallasEngine>;

// Final Aggregation SNARKS
type EE1 = ipa_pc::EvaluationEngine<E1>;
type EE2 = ipa_pc::EvaluationEngine<Dual<E1>>;
type AS1 = spartan::snark::RelaxedR1CSSNARK<E1, EE1>;
type AS2 = spartan::snark::RelaxedR1CSSNARK<E2, EE2>;

#[test]
fn test_aggregator() -> anyhow::Result<()> {
  let num_snarks = 3;
  let snarks_data = sim_network(num_snarks);

  println!("Producing PP for Aggregated SNARK...");
  let pp = AggPublicParams::setup(
    &snarks_data,
    &crate::traits::snark::default_ck_hint(),
    &crate::traits::snark::default_ck_hint(),
  )?;

  let (pk, vk) = AggregatedSNARK::<E1, AS1, AS2>::setup(&pp)?;

  println!("Proving Aggregated SNARK...");
  let snark = AggregatedSNARK::<E1, AS1, AS2>::prove(&pp, &pk, &snarks_data)?;

  println!("Verifying Aggregated SNARK...");
  snark.verify(&vk)?;
  Ok(())
}

#[test]
fn test_aggregator_single() -> anyhow::Result<()> {
  let num_snarks = 1;
  let snarks_data = sim_network(num_snarks);

  println!("Producing PP for Aggregated SNARK...");
  let pp = AggPublicParams::setup(
    &snarks_data,
    &crate::traits::snark::default_ck_hint(),
    &crate::traits::snark::default_ck_hint(),
  )?;

  let (pk, vk) = AggregatedSNARK::<E1, AS1, AS2>::setup(&pp)?;

  println!("Proving Aggregated SNARK...");
  let snark = AggregatedSNARK::<E1, AS1, AS2>::prove(&pp, &pk, &snarks_data)?;

  println!("Verifying Aggregated SNARK...");
  let _ = snark.verify(&vk);
  Ok(())
}

fn sim_network(num_snarks: usize) -> Vec<AggregatorSNARKData<E1>> {
  println!("simulating proving nodes (sequentially)");
  let mut snarks_data_for_agg = vec![];

  for i in 0..num_snarks {
    println!("node {i}: Producing proof to be aggregated");
    let (snark, vk) = test_nivc_with::<E1, S1>();

    let TestSNARK { snark, instance: U } = snark;

    let aggregator_snark_data = AggregatorSNARKData::new(snark, vk, U);

    snarks_data_for_agg.push(aggregator_snark_data)
  }

  snarks_data_for_agg
}

type S1 = ipa_prover_poseidon::batched::BatchedRelaxedR1CSSNARK<E1>;
type _S2 = ipa_prover_poseidon::batched::BatchedRelaxedR1CSSNARK<E2>;

fn test_nivc_with<E1, S1>() -> (TestSNARK<E1, S1>, S1::VerifierKey)
where
  E1: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
{
  // Here demo a simple RAM machine
  // - with 2 argumented circuit
  // - each argumented circuit contains primary and secondary circuit
  // - a memory commitment via a public IO `rom` (like a program) to constraint the sequence execution

  // This test also ready to add more argumented circuit and ROM can be arbitrary length

  // ROM is for constraints the sequence of execution order for opcode

  // TODO: replace with memory commitment along with suggestion from Supernova 4.4 optimisations

  // This is mostly done with the existing Nova code. With additions of U_i[] and program_counter checks
  // in the augmented circuit.

  let rom = vec![OPCODE_0, OPCODE_1]; // Rom can be arbitrary length.

  let test_rom = TestROM::<E1>::new(rom);

  let pp = PublicParams::setup(&test_rom, &*default_ck_hint(), &*default_ck_hint());

  // extend z0_primary/secondary with rom content
  let mut z0_primary = vec![<E1 as Engine>::Scalar::ONE];
  z0_primary.push(<E1 as Engine>::Scalar::ZERO); // rom_index = 0
  z0_primary.extend(
    test_rom
      .rom
      .iter()
      .map(|opcode| <E1 as Engine>::Scalar::from(*opcode as u64)),
  );
  let z0_secondary = vec![<Dual<E1> as Engine>::Scalar::ONE];

  let mut recursive_snark_option: Option<RecursiveSNARK<E1>> = None;

  for &op_code in test_rom.rom.iter() {
    let circuit_primary = test_rom.primary_circuit(op_code);
    let circuit_secondary = test_rom.secondary_circuit();

    let mut recursive_snark = recursive_snark_option.unwrap_or_else(|| {
      RecursiveSNARK::new(
        &pp,
        &test_rom,
        &circuit_primary,
        &circuit_secondary,
        &z0_primary,
        &z0_secondary,
      )
      .unwrap()
    });

    recursive_snark
      .prove_step(&pp, &circuit_primary, &circuit_secondary)
      .unwrap();
    recursive_snark
      .verify(&pp, &z0_primary, &z0_secondary)
      .unwrap();

    recursive_snark_option = Some(recursive_snark)
  }

  assert!(recursive_snark_option.is_some());
  let recursive_snark = recursive_snark_option.unwrap();

  let (pk, vk) = TestSNARK::<E1, S1>::setup(&pp).unwrap();
  let snark = TestSNARK::<E1, S1>::prove(&pp, &pk, &recursive_snark).unwrap();
  snark.verify(&pp, &vk).unwrap();

  (snark, vk)
}

pub struct TestSNARK<E1, S1>
where
  E1: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
{
  snark: S1,
  instance: Vec<RelaxedR1CSInstance<E1>>,
}

impl<E1, S1> TestSNARK<E1, S1>
where
  E1: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
{
  /// Creates prover and verifier keys for `CompressedSNARK`
  pub fn setup(pp: &PublicParams<E1>) -> Result<(S1::ProverKey, S1::VerifierKey), SuperNovaError> {
    let (pk, vk) = S1::setup(pp.ck_primary.clone(), pp.primary_r1cs_shapes())?;

    Ok((pk, vk))
  }

  /// Create a new `CompressedSNARK`
  pub fn prove(
    pp: &PublicParams<E1>,
    pk: &S1::ProverKey,
    recursive_snark: &RecursiveSNARK<E1>,
  ) -> Result<Self, SuperNovaError> {
    // Prepare the list of primary Relaxed R1CS instances (a default instance is provided for
    // uninitialized circuits)
    let r_U_primary = recursive_snark
      .r_U_primary
      .iter()
      .enumerate()
      .map(|(idx, r_U)| {
        r_U
          .clone()
          .unwrap_or_else(|| RelaxedR1CSInstance::default(&*pp.ck_primary, &pp[idx].r1cs_shape))
      })
      .collect::<Vec<_>>();

    // Prepare the list of primary relaxed R1CS witnesses (a default witness is provided for
    // uninitialized circuits)
    let r_W_primary: Vec<RelaxedR1CSWitness<E1>> = recursive_snark
      .r_W_primary
      .iter()
      .enumerate()
      .map(|(idx, r_W)| {
        r_W
          .clone()
          .unwrap_or_else(|| RelaxedR1CSWitness::default(&pp[idx].r1cs_shape))
      })
      .collect::<Vec<_>>();

    // Generate a primary SNARK proof for the list of primary circuits
    let r_W_snark_primary = S1::prove(
      &pp.ck_primary,
      &pk,
      pp.primary_r1cs_shapes(),
      &r_U_primary,
      &r_W_primary,
    )?;

    let compressed_snark = Self {
      snark: r_W_snark_primary,
      instance: r_U_primary,
    };

    Ok(compressed_snark)
  }

  /// Verify the correctness of the `CompressedSNARK`
  pub fn verify(&self, _pp: &PublicParams<E1>, vk: &S1::VerifierKey) -> Result<(), SuperNovaError> {
    // Verify the primary SNARK
    let res_primary = self.snark.verify(vk, &self.instance);

    res_primary?;

    Ok(())
  }
}

#[derive(Clone, Debug, Default)]
struct CubicCircuit<F> {
  _p: PhantomData<F>,
  circuit_index: usize,
  rom_size: usize,
}

impl<F> CubicCircuit<F> {
  fn new(circuit_index: usize, rom_size: usize) -> Self {
    Self {
      circuit_index,
      rom_size,
      _p: PhantomData,
    }
  }
}

fn next_rom_index_and_pc<F: PrimeField, CS: ConstraintSystem<F>>(
  cs: &mut CS,
  rom_index: &AllocatedNum<F>,
  allocated_rom: &[AllocatedNum<F>],
  pc: &AllocatedNum<F>,
) -> Result<(AllocatedNum<F>, AllocatedNum<F>), SynthesisError> {
  // Compute a selector for the current rom_index in allocated_rom
  let current_rom_selector = get_selector_vec_from_index(
    cs.namespace(|| "rom selector"),
    rom_index,
    allocated_rom.len(),
  )?;

  // Enforce that allocated_rom[rom_index] = pc
  for (rom, bit) in allocated_rom.iter().zip_eq(current_rom_selector.iter()) {
    // if bit = 1, then rom = pc
    // bit * (rom - pc) = 0
    cs.enforce(
      || "enforce bit = 1 => rom = pc",
      |lc| lc + &bit.lc(CS::one(), F::ONE),
      |lc| lc + rom.get_variable() - pc.get_variable(),
      |lc| lc,
    );
  }

  // Get the index of the current rom, or the index of the invalid rom if no match
  let current_rom_index = current_rom_selector
    .iter()
    .position(|bit| bit.get_value().is_some_and(|v| v))
    .unwrap_or_default();
  let next_rom_index = current_rom_index + 1;

  let rom_index_next = AllocatedNum::alloc_infallible(cs.namespace(|| "next rom index"), || {
    F::from(next_rom_index as u64)
  });
  cs.enforce(
    || " rom_index + 1 - next_rom_index_num = 0",
    |lc| lc,
    |lc| lc,
    |lc| lc + rom_index.get_variable() + CS::one() - rom_index_next.get_variable(),
  );

  // Allocate the next pc without checking.
  // The next iteration will check whether the next pc is valid.
  let pc_next = AllocatedNum::alloc_infallible(cs.namespace(|| "next pc"), || {
    allocated_rom
      .get(next_rom_index)
      .and_then(|v| v.get_value())
      .unwrap_or(-F::ONE)
  });

  Ok((rom_index_next, pc_next))
}

impl<F> StepCircuit<F> for CubicCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    2 + self.rom_size // value + rom_pc + rom[].len()
  }

  fn circuit_index(&self) -> usize {
    self.circuit_index
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    let rom_index = &z[1];
    let allocated_rom = &z[2..];

    let (rom_index_next, pc_next) = next_rom_index_and_pc(
      &mut cs.namespace(|| "next and rom_index and pc"),
      rom_index,
      allocated_rom,
      pc.ok_or(SynthesisError::AssignmentMissing)?,
    )?;

    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = &z[0];
    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
    })?;

    cs.enforce(
      || "y = x^3 + x + 5",
      |lc| {
        lc + x_cu.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    let mut z_next = vec![y];
    z_next.push(rom_index_next);
    z_next.extend(z[2..].iter().cloned());
    Ok((Some(pc_next), z_next))
  }
}

#[derive(Clone, Debug, Default)]
struct SquareCircuit<F> {
  _p: PhantomData<F>,
  circuit_index: usize,
  rom_size: usize,
}

impl<F> SquareCircuit<F> {
  fn new(circuit_index: usize, rom_size: usize) -> Self {
    Self {
      circuit_index,
      rom_size,
      _p: PhantomData,
    }
  }
}

impl<F> StepCircuit<F> for SquareCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    2 + self.rom_size // value + rom_pc + rom[].len()
  }

  fn circuit_index(&self) -> usize {
    self.circuit_index
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    let rom_index = &z[1];
    let allocated_rom = &z[2..];

    let (rom_index_next, pc_next) = next_rom_index_and_pc(
      &mut cs.namespace(|| "next and rom_index and pc"),
      rom_index,
      allocated_rom,
      pc.ok_or(SynthesisError::AssignmentMissing)?,
    )?;

    // Consider an equation: `x^2 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = &z[0];
    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_sq.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
    })?;

    cs.enforce(
      || "y = x^2 + x + 5",
      |lc| {
        lc + x_sq.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    let mut z_next = vec![y];
    z_next.push(rom_index_next);
    z_next.extend(z[2..].iter().cloned());
    Ok((Some(pc_next), z_next))
  }
}

const OPCODE_0: usize = 0;
const OPCODE_1: usize = 1;

struct TestROM<E1> {
  rom: Vec<usize>,
  _p: PhantomData<E1>,
}

#[derive(Debug, Clone)]
enum TestROMCircuit<F: PrimeField> {
  Cubic(CubicCircuit<F>),
  Square(SquareCircuit<F>),
}

impl<F: PrimeField> StepCircuit<F> for TestROMCircuit<F> {
  fn arity(&self) -> usize {
    match self {
      Self::Cubic(x) => x.arity(),
      Self::Square(x) => x.arity(),
    }
  }

  fn circuit_index(&self) -> usize {
    match self {
      Self::Cubic(x) => x.circuit_index(),
      Self::Square(x) => x.circuit_index(),
    }
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    match self {
      Self::Cubic(x) => x.synthesize(cs, pc, z),
      Self::Square(x) => x.synthesize(cs, pc, z),
    }
  }
}

impl<E1> NonUniformCircuit<E1> for TestROM<E1>
where
  E1: CurveCycleEquipped,
{
  type C1 = TestROMCircuit<E1::Scalar>;
  type C2 = TrivialSecondaryCircuit<<Dual<E1> as Engine>::Scalar>;

  fn num_circuits(&self) -> usize {
    2
  }

  fn primary_circuit(&self, circuit_index: usize) -> Self::C1 {
    match circuit_index {
      0 => TestROMCircuit::Cubic(CubicCircuit::new(circuit_index, self.rom.len())),
      1 => TestROMCircuit::Square(SquareCircuit::new(circuit_index, self.rom.len())),
      _ => panic!("unsupported primary circuit index"),
    }
  }

  fn secondary_circuit(&self) -> Self::C2 {
    Default::default()
  }

  fn initial_circuit_index(&self) -> usize {
    self.rom[0]
  }
}

impl<E1> TestROM<E1> {
  fn new(rom: Vec<usize>) -> Self {
    Self {
      rom,
      _p: Default::default(),
    }
  }
}
