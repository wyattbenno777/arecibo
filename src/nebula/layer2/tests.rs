use crate::bellpepper::r1cs::NovaShape;
use crate::nebula::rs::{PublicParams, RecursiveSNARK};
use crate::r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
use crate::traits::snark::default_ck_hint;
use crate::traits::CurveCycleEquipped;
use crate::{
  bellpepper::shape_cs::ShapeCS,
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  nebula::{
    augmented_circuit::{AugmentedCircuit, AugmentedCircuitParams},
    rs::StepCircuit,
  },
  provider::Bn256EngineIPA,
  traits::{Dual, Engine, ROConstantsCircuit},
};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use ff::PrimeField;

use super::Layer2RS;

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

type E1 = Bn256EngineIPA;
type F = <E1 as Engine>::Scalar;

#[test]
fn test_4_4_4() {
  let circuits1 = [Add32Circuit { a: 4, b: 4 }, Add32Circuit { a: 10, b: 12 }];
  let circuits2 = [Mul32Circuit { a: 4, b: 4 }, Mul32Circuit { a: 11, b: 12 }];
  let circuits3 = [LineCircuit { x: 4 }, LineCircuit { x: 10 }];

  let pp1 = PublicParams::<E1>::setup(
    &Add32Circuit::default(),
    &*default_ck_hint(),
    &*default_ck_hint(),
  );
  let pp2 = PublicParams::<E1>::setup(
    &Mul32Circuit::default(),
    &*default_ck_hint(),
    &*default_ck_hint(),
  );
  let pp3 = PublicParams::<E1>::setup(
    &LineCircuit::default(),
    &*default_ck_hint(),
    &*default_ck_hint(),
  );

  let (rs1, rs2, rs3) = {
    (
      RSNARK(&pp1, &circuits1),
      RSNARK(&pp2, &circuits2),
      RSNARK(&pp3, &circuits3),
    )
  };

  let final_rs = Layer2RS::new((&pp1, &pp2, &pp3));
  final_rs
    .prove_step((&pp1, &pp2, &pp3), (&rs1, &rs2, &rs3))
    .unwrap();
}

fn RSNARK(pp: &PublicParams<E1>, C: &[impl StepCircuit<F>]) -> RecursiveSNARK<E1>
where
  F: PrimeField,
{
  let z0 = vec![F::from(0u64)];

  let mut recursive_snark = RecursiveSNARK::new(&pp, &C[0], &z0).unwrap();
  let mut IC_i = F::ZERO;

  for i in 0..2 {
    recursive_snark.prove_step(pp, &C[i], IC_i).unwrap();

    IC_i = recursive_snark.increment_commitment(pp, &C[i]);
  }
  recursive_snark
    .verify(pp, recursive_snark.num_steps(), &z0, IC_i)
    .unwrap();

  recursive_snark
}
