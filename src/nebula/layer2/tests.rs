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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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

type E1 = Bn256EngineIPA;
type F = <E1 as Engine>::Scalar;

#[test]
fn test_4_4_4() {
  let circuit1 = Add32Circuit { a: 4, b: 4 };
  let circuit2 = Mul32Circuit { a: 4, b: 4 };
  // Get shape
  let ro_consts_circuit = ROConstantsCircuit::<Dual<E1>>::default();

  let augmented_circuit_params = AugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS);

  let aug_circuit1: AugmentedCircuit<'_, E1, _> = AugmentedCircuit::new(
    &augmented_circuit_params,
    ro_consts_circuit.clone(),
    None,
    &circuit1,
  );
  let aug_circuit2: AugmentedCircuit<'_, E1, _> = AugmentedCircuit::new(
    &augmented_circuit_params,
    ro_consts_circuit.clone(),
    None,
    &circuit2,
  );
  let mut cs: ShapeCS<E1> = ShapeCS::new();
  let _ = aug_circuit1.synthesize(&mut cs);
  let _ = aug_circuit2.synthesize(&mut cs);

  let (r1cs_main, ck_main) = cs.r1cs_shape_and_key(&*default_ck_hint());
  let U_main = RelaxedR1CSInstance::default(&ck_main, &r1cs_main);
  let W_main = RelaxedR1CSWitness::default(&r1cs_main);

  let (U1, W1) = get_U_W::<E1>(circuit1);
  let (U2, W2) = get_U_W::<E1>(circuit2);

  // Folding
}

fn get_U_W<E>(
  circuit: impl StepCircuit<E::Scalar>,
) -> (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>)
where
  E: CurveCycleEquipped,
{
  let pp = PublicParams::<E>::setup(&circuit, &*default_ck_hint(), &*default_ck_hint());

  let z0 = vec![E::Scalar::from(2u64)];

  let mut recursive_snark = RecursiveSNARK::new(&pp, &circuit, &z0).unwrap();
  let mut IC_i = E::Scalar::ZERO;

  for _ in 0..2 {
    recursive_snark.prove_step(&pp, &circuit, IC_i).unwrap();

    IC_i = recursive_snark.increment_commitment(&pp, &circuit);
  }

  recursive_snark.U_W()
}
