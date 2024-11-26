use crate::errors::NovaError;
use crate::gadgets::scalar_as_base;
use crate::nebula::layer2::nifs::RelaxedNIFS;
use crate::nebula::layer2::utils::RelaxedFoldingData;
use crate::nebula::rs::{PublicParams, RecursiveSNARK};
use crate::r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
use crate::traits::commitment::CommitmentTrait;
use crate::traits::snark::default_ck_hint;
use crate::traits::CurveCycleEquipped;
use crate::Commitment;
use crate::{nebula::rs::StepCircuit, provider::Bn256EngineIPA, traits::Engine};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use ff::PrimeField;
use serde::{Deserialize, Serialize};

use super::final_circuit::{FinalCircuit, FinalCircuitInputs, FinalCircuitParams};

#[derive(Debug, Clone, Deserialize, Serialize)]
struct L2<E>
where
  E: CurveCycleEquipped,
{
  r_U: RelaxedR1CSInstance<E>,
  r_W: RelaxedR1CSWitness<E>,
}

impl<E> L2<E>
where
  E: CurveCycleEquipped,
{
  /// Create a new instance of [`L2`]
  pub fn new(node_pp: &PublicParams<E>) -> L2<E> {
    let r_U = RelaxedR1CSInstance::default(
      &node_pp.ck_primary,
      &node_pp.circuit_shape_primary.r1cs_shape,
    );
    let r_W = RelaxedR1CSWitness::default(&node_pp.circuit_shape_primary.r1cs_shape);

    Self { r_U, r_W }
  }

  /// updates the provided [`L2`] by executing a step of the incremental computation
  pub fn prove_step(
    &mut self,
    node_pp: &PublicParams<E>,
    node_rs: RecursiveSNARK<E>,
  ) -> Result<(), NovaError> {
    let node_pp_digest = node_pp.digest();
    let (l_U, l_W) = node_rs.U_W();
    let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
      &node_pp.ck_primary,
      &node_pp.ro_consts_primary,
      &node_pp_digest,
      &node_pp.circuit_shape_primary.r1cs_shape,
      &self.r_U,
      &self.r_W,
      l_U,
      l_W,
    )?;

    let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

    let data_p = RelaxedFoldingData::new(self.r_U.clone(), l_U.clone(), comm_T);

    // Do calculation's outside circuit, to be passed in as advice to F'
    let E_new = self.r_U.comm_E + comm_T * r;
    let W_new = self.r_U.comm_W + l_U.comm_W * r;

    let final_circuit_inputs =
      FinalCircuitInputs::<E>::new(scalar_as_base::<E>(node_pp_digest), data_p, E_new, W_new);

    let final_circuit = FinalCircuit::new(
      &FinalCircuitParams::from(&node_pp.augmented_circuit_params),
      node_pp.ro_consts_circuit_primary.clone(),
      final_circuit_inputs,
    );

    Ok(())
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
fn test_ivc_folding() {
  let (pp, rsnark) = node();
}

fn aggregation_node(
  node_pp: &PublicParams<E1>,
  node_U: RelaxedR1CSInstance<E1>,
  node_W: RelaxedR1CSWitness<E1>,
) {
  let L2 = L2::new(node_pp);
}

fn node() -> (PublicParams<E1>, RecursiveSNARK<E1>) {
  let pp = PublicParams::<E1>::setup(
    &LineCircuit::default(),
    &*default_ck_hint(),
    &*default_ck_hint(),
  );

  let circuits = [LineCircuit { x: 4 }, LineCircuit { x: 10 }];
  let proof = RSNARK(&pp, &circuits);
  (pp, proof)
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
