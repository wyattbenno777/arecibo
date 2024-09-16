use super::SpartanVerifyCircuit;
use crate::bellpepper::r1cs::{NovaShape, NovaWitness};
use crate::bellpepper::shape_cs::ShapeCS;
use crate::bellpepper::solver::SatisfyingAssignment;

use crate::traits::commitment::CommitmentEngineTrait;
use crate::{
  provider::PallasEngine,
  r1cs::{
    commitment_key_size, CommitmentKeyHint, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness,
  },
  traits::{snark::BatchedRelaxedR1CSSNARKTrait, Engine},
  CommitmentKey,
};
use bellpepper_core::test_cs::TestConstraintSystem;
use bellpepper_core::{num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use itertools::Itertools;
use pasta_curves::Fq;
use std::marker::PhantomData;
use std::sync::Arc;
use tracing_test::traced_test;

use crate::spartan::verify_circuit::ipa_prover_poseidon::batched_ppsnark::{
  ProverKey, VerifierKey,
};

/// Proving config
type E = PallasEngine;
type SNARK =
  crate::spartan::verify_circuit::ipa_prover_poseidon::batched_ppsnark::BatchedRelaxedR1CSSNARK<E>;

#[test]
#[traced_test]
fn test_verify_circuit() -> anyhow::Result<()> {
  tracing::info!("setting up SNARK...");
  let (ck, pk, vk, S) = NonUniformSNARK::setup(PolyCircuit::default())?;

  tracing::info!("proving SNARK...");
  let poly_circuits = PolyCircuit::new(2, 10);
  let proof = NonUniformSNARK::prove(&ck, &pk, S, poly_circuits)?;
  let NonUniformSNARK {
    inner_proof,
    instance,
  } = proof;

  let mut cs = TestConstraintSystem::<Fq>::new();

  let verify_circuit = SpartanVerifyCircuit::new(inner_proof);

  tracing::info!("synthesizing verify circuit...");
  let _ = verify_circuit.synthesize(cs.namespace(|| "verify"), &vk, &instance)?;
  assert!(cs.is_satisfied());
  Ok(())
}

#[test]
#[traced_test]
fn test_verify_cs() -> anyhow::Result<()> {
  tracing::info!("setting up SNARK...");
  let (ck, pk, vk, S) = NonUniformSNARK::setup(PolyCircuit::default())?;

  tracing::info!("proving SNARK...");
  let poly_circuits = PolyCircuit::new(2, 10);
  let proof = NonUniformSNARK::prove(&ck, &pk, S, poly_circuits)?;
  let NonUniformSNARK {
    inner_proof,
    instance,
  } = proof;

  let mut cs = TestConstraintSystem::<Fq>::new();
  inner_proof.verify_cs(cs.namespace(|| "verify cs"), &vk, &instance)?;

  assert!(cs.is_satisfied());
  Ok(())
}

struct NonUniformSNARK {
  inner_proof: SNARK,
  instance: Vec<RelaxedR1CSInstance<E>>,
}

impl NonUniformSNARK {
  /// Setup for SNARK proving
  fn setup(
    circuits: Vec<PolyCircuit<Fq>>,
  ) -> anyhow::Result<(
    CommitmentKey<E>,  // ck
    ProverKey<E>,      // pk
    VerifierKey<E>,    // vk
    Vec<R1CSShape<E>>, // r1cs shapes
  )> {
    // Get circuit shapes
    let circuit_shapes: Vec<R1CSShape<E>> = circuits
      .into_iter()
      .map(|sub_circuit| {
        let mut cs: ShapeCS<E> = ShapeCS::new();

        let _ = sub_circuit
          .synthesize(&mut cs)
          .expect("failed to synthesize");

        cs.r1cs_shape()
      })
      .collect();

    let ck = compute_ck_primary(&circuit_shapes, &*SNARK::ck_floor());

    let (pk, vk) = SNARK::setup(Arc::new(ck.clone()), circuit_shapes.iter().collect())
      .expect("failed to setup PP SNARK");

    Ok((ck, pk, vk, circuit_shapes))
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &ProverKey<E>,
    S: Vec<R1CSShape<E>>,
    circuits: Vec<PolyCircuit<Fq>>,
  ) -> anyhow::Result<Self> {
    // Get circuit instance's and witnesses
    let mut U: Vec<RelaxedR1CSInstance<E>> = Vec::new();
    let mut W: Vec<RelaxedR1CSWitness<E>> = Vec::new();
    for (sub_circuit, shape) in circuits.into_iter().zip_eq(S.iter()) {
      let mut cs = SatisfyingAssignment::<E>::new();

      let _ = sub_circuit
        .synthesize(&mut cs)
        .expect("failed to synthesize");

      let (U_i, W_i) = cs
        .r1cs_instance_and_witness(shape, ck)
        .expect("failed to synthesize circuit");

      U.push(RelaxedR1CSInstance::from_r1cs_instance(ck, shape, U_i));
      W.push(RelaxedR1CSWitness::from_r1cs_witness(shape, W_i));
    }

    let S = S.iter().collect();
    let proof = SNARK::prove(&ck, &pk, S, &U, &W)?;
    Ok(Self {
      inner_proof: proof,
      instance: U,
    })
  }

  // verify the SNARK
  pub fn verify(&self, vk: &VerifierKey<E>) -> anyhow::Result<()> {
    self.inner_proof.verify(vk, &self.instance)?;
    Ok(())
  }
}

/// Compute primary and secondary commitment keys sized to handle the largest of the circuits in the provided
/// `R1CSWithArity`.
fn compute_ck_primary(
  circuit_shapes: &[R1CSShape<E>],
  ck_hint1: &CommitmentKeyHint<E>,
) -> CommitmentKey<E> {
  let size_primary = circuit_shapes
    .iter()
    .map(|shape| commitment_key_size(&shape, ck_hint1))
    .max()
    .unwrap();

  <E as Engine>::CE::setup(b"ck", size_primary)
}

/// Holds circuits to produce the Vec<R1CS> to prove
#[derive(Debug, Clone)]
enum PolyCircuit<F: PrimeField> {
  Cubic(CubicCircuit<F>),
  Square(SquareCircuit<F>),
}

impl<F: PrimeField> PolyCircuit<F> {
  fn default() -> Vec<Self> {
    let mut circuits = Vec::new();
    circuits.push(Self::Square(SquareCircuit::default()));
    circuits.push(Self::Cubic(CubicCircuit::default()));
    circuits
  }

  fn new(x_sq: u64, x_cu: u64) -> Vec<Self> {
    let mut circuits = Vec::new();

    circuits.push(Self::Square(SquareCircuit {
      x: Some(x_sq),
      _p: PhantomData,
    }));

    circuits.push(Self::Cubic(CubicCircuit {
      x: Some(x_cu),
      _p: PhantomData,
    }));

    circuits
  }
}

impl<F: PrimeField> Circuit<F> for PolyCircuit<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    match self {
      Self::Cubic(c) => c.synthesize(cs),
      Self::Square(c) => c.synthesize(cs),
    }
  }
}

/// Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
#[derive(Clone, Debug, Default)]
struct CubicCircuit<F: PrimeField> {
  x: Option<u64>,
  _p: PhantomData<F>,
}

impl<F> Circuit<F> for CubicCircuit<F>
where
  F: PrimeField,
{
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = self.x;
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      x.map_or(Ok(F::ZERO), |x| Ok(F::from(x)))
    })?;
    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
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

    Ok(())
  }
}

// Consider an equation: `x^2 + x + 5 = y`, where `x` and `y` are respectively the input and output.
#[derive(Clone, Debug, Default)]
struct SquareCircuit<F: PrimeField> {
  x: Option<u64>,
  _p: PhantomData<F>,
}

impl<F> Circuit<F> for SquareCircuit<F>
where
  F: PrimeField,
{
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // Consider an equation: `x^2 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = self.x;
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      x.map_or(Ok(F::ZERO), |x| Ok(F::from(x)))
    })?;

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

    Ok(())
  }
}

#[test]
#[traced_test]
fn test_batched_ppsnark() -> anyhow::Result<()> {
  println!("setting up SNARK...");
  let (ck, pk, vk, S) = NonUniformSNARK::setup(PolyCircuit::default())?;

  println!("proving SNARK...");
  let poly_circuits = PolyCircuit::new(2, 10);
  let proof = NonUniformSNARK::prove(&ck, &pk, S, poly_circuits)?;

  println!("verifying SNARK...");
  proof.verify(&vk)?;
  Ok(())
}
