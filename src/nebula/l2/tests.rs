use std::marker::PhantomData;

use super::{L2AggregationPublicParams, L2AggregationRS, Layer1PP, Layer1RSTrait};
use crate::nebula::rs::{PublicParams, RecursiveSNARK};
use crate::traits::snark::default_ck_hint;
use crate::{nebula::rs::StepCircuit, provider::Bn256EngineIPA, traits::Engine};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use ff::PrimeField;

type E1 = Bn256EngineIPA;
type F = <E1 as Engine>::Scalar;

#[test]
fn test_ivc_folding() {
  tracing_texray::init();
  let (node_pp, nodes_rs) = node_nw(3);
  tracing_texray::examine(tracing::info_span!("aggregation"))
    .in_scope(|| aggregation_node(node_pp, &nodes_rs));
}

fn aggregation_node(node_pp: NodePP, nodes_rs: &[NodeRS]) {
  let aggregation_pp = L2AggregationPublicParams::setup(node_pp);
  let mut aggregation_engine = L2AggregationRS::new(&aggregation_pp, &nodes_rs[0]).unwrap();

  for node_rs in nodes_rs.iter() {
    aggregation_engine
      .prove_step(&aggregation_pp, node_rs)
      .unwrap();
  }

  aggregation_engine.verify(&aggregation_pp).unwrap();
}

struct NodePP {
  pp1: PublicParams<E1>,
  pp2: PublicParams<E1>,
  pp3: PublicParams<E1>,
}

impl Layer1PP<E1> for NodePP {
  fn into_parts(self) -> (PublicParams<E1>, PublicParams<E1>, PublicParams<E1>) {
    (self.pp1, self.pp2, self.pp3)
  }
}

struct NodeRS {
  rs1: RecursiveSNARK<E1>,
  rs2: RecursiveSNARK<E1>,
  rs3: RecursiveSNARK<E1>,
}

impl Layer1RSTrait<E1> for NodeRS {
  fn F(&self) -> &RecursiveSNARK<E1> {
    &self.rs1
  }

  fn ops(&self) -> &RecursiveSNARK<E1> {
    &self.rs2
  }

  fn scan(&self) -> &RecursiveSNARK<E1> {
    &self.rs3
  }
}

fn node_nw(num_proofs: usize) -> (NodePP, Vec<NodeRS>) {
  let pp1 = tracing::info_span!("node_pp").in_scope(|| {
    PublicParams::<E1>::setup(
      &LineCircuit::default(),
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
  });

  let pp2 = tracing::info_span!("node_pp").in_scope(|| {
    PublicParams::<E1>::setup(
      &Add32Circuit::default(),
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
  });

  let pp3 = tracing::info_span!("node_pp").in_scope(|| {
    PublicParams::<E1>::setup(
      &Mul32Circuit::default(),
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
  });

  let line_circuits = [LineCircuit { x: 4 }, LineCircuit { x: 10 }];
  let add32_circuits = [Add32Circuit { a: 12, b: 10 }, Add32Circuit { a: 8, b: 4 }];
  let mul32_circuits = [Mul32Circuit { a: 10, b: 4 }, Mul32Circuit { a: 20, b: 7 }];

  let proofs = (0..num_proofs)
    .map(|_| {
      let line_rs = RSNARK(&pp1, &line_circuits);
      let add32_rs = RSNARK(&pp2, &add32_circuits);
      let mul32_rs = RSNARK(&pp3, &mul32_circuits);

      NodeRS {
        rs1: line_rs,
        rs2: add32_rs,
        rs3: mul32_rs,
      }
    })
    .collect::<Vec<_>>();

  (NodePP { pp1, pp2, pp3 }, proofs)
}

fn RSNARK(pp: &PublicParams<E1>, C: &[impl StepCircuit<F>]) -> RecursiveSNARK<E1>
where
  F: PrimeField,
{
  let z0 = vec![F::from(0u64)];

  let mut recursive_snark = RecursiveSNARK::new(pp, &C[0], &z0).unwrap();
  let mut IC_i = F::ZERO;

  for circuit in C {
    recursive_snark.prove_step(pp, circuit, IC_i).unwrap();

    IC_i = recursive_snark.increment_commitment(pp, circuit);
  }
  recursive_snark
    .verify(pp, recursive_snark.num_steps(), &z0, IC_i)
    .unwrap();

  recursive_snark
}

#[derive(Clone, Debug, Default)]
struct LineCircuit {
  x: u32,
}

impl<F> StepCircuit<F> for LineCircuit
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn non_deterministic_advice(&self) -> Vec<F> {
    vec![]
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    // y = 2x + 1
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(F::from(self.x as u64)))?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      let two = F::from(2u64);
      let one = F::from(1u64);
      x.get_value()
        .map(|x| x * two + one)
        .ok_or(SynthesisError::AssignmentMissing)
    })?;

    // 2x = y - 1
    cs.enforce(
      || "2x = y - 1",
      |lc| lc + x.get_variable(),
      |lc| lc + (F::from(2u64), CS::one()),
      |lc| lc + y.get_variable() - CS::one(),
    );

    Ok(z.to_vec())
  }
}

#[derive(Clone, Debug, Default)]
struct Add32Circuit {
  a: u32,
  b: u32,
}

impl<F> StepCircuit<F> for Add32Circuit
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn non_deterministic_advice(&self) -> Vec<F> {
    vec![]
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let zero = F::ZERO;
    let O = F::from(0x100000000u64);
    let ON: F = zero - O;

    let (c, of) = self.a.overflowing_add(self.b);
    let o = if of { ON } else { zero };

    // construct witness
    let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(F::from(self.a as u64)))?;
    let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(F::from(self.b as u64)))?;
    let c = AllocatedNum::alloc(cs.namespace(|| "c"), || Ok(F::from(c as u64)))?;

    // note, this is "advice"
    let o = AllocatedNum::alloc(cs.namespace(|| "o"), || Ok(o))?;
    let O = AllocatedNum::alloc(cs.namespace(|| "O"), || Ok(O))?;

    // check o * (o + O) == 0
    cs.enforce(
      || "check o * (o + O) == 0",
      |lc| lc + o.get_variable(),
      |lc| lc + o.get_variable() + O.get_variable(),
      |lc| lc,
    );

    // a + b + o = c
    cs.enforce(
      || "x + y + o = z",
      |lc| lc + a.get_variable() + b.get_variable() + o.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + c.get_variable(),
    );

    Ok(z.to_vec())
  }
}

#[derive(Clone, Debug, Default)]
struct Mul32Circuit {
  a: u32,
  b: u32,
}

impl<F> StepCircuit<F> for Mul32Circuit
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn non_deterministic_advice(&self) -> Vec<F> {
    vec![]
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let zero = F::ZERO;
    let O = F::from(0x100000000u64);
    let ON: F = zero - O;

    let (c, of) = self.a.overflowing_mul(self.b);
    let o = if of { ON } else { zero };

    // construct witness
    let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(F::from(self.a as u64)))?;
    let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(F::from(self.b as u64)))?;
    let c = AllocatedNum::alloc(cs.namespace(|| "c"), || Ok(F::from(c as u64)))?;

    // note, this is "advice"
    let o = AllocatedNum::alloc(cs.namespace(|| "o"), || Ok(o))?;
    let O = AllocatedNum::alloc(cs.namespace(|| "O"), || Ok(O))?;

    // check o * (o + O) == 0
    cs.enforce(
      || "check o * (o + O) == 0",
      |lc| lc + o.get_variable(),
      |lc| lc + o.get_variable() + O.get_variable(),
      |lc| lc,
    );

    // a * b = c - o
    cs.enforce(
      || "a * b = c - o",
      |lc| lc + a.get_variable(),
      |lc| lc + b.get_variable(),
      |lc| lc + c.get_variable() - o.get_variable(),
    );

    Ok(z.to_vec())
  }
}
