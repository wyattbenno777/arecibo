use super::{public_params, L2};
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
    .in_scope(|| aggregation_node(&node_pp, &nodes_rs));
}

fn aggregation_node(node_pp: &PublicParams<E1>, nodes_rs: &[RecursiveSNARK<E1>]) {
  let final_pp = public_params(node_pp);
  let mut L2 = L2::new(&final_pp, node_pp, &nodes_rs[0]).unwrap();

  for node_rs in nodes_rs.iter() {
    L2.prove_step(&final_pp, node_pp, node_rs).unwrap();
  }

  L2.verify(&final_pp).unwrap();
}

fn node_nw(num_proofs: usize) -> (PublicParams<E1>, Vec<RecursiveSNARK<E1>>) {
  let pp = tracing::info_span!("node_pp").in_scope(|| {
    PublicParams::<E1>::setup(
      &LineCircuit::default(),
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
  });
  let circuits = [LineCircuit { x: 4 }, LineCircuit { x: 10 }];
  let proofs = (0..num_proofs)
    .map(|_| RSNARK(&pp, &circuits))
    .collect::<Vec<RecursiveSNARK<E1>>>();

  (pp, proofs)
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
