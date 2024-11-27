use super::nifs::RelaxedNIFS;
use super::utils::RelaxedFoldingData;
use crate::errors::NovaError;
use crate::gadgets::scalar_as_base;
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

fn public_params<E1>(node_pp: &PublicParams<E1>) -> PublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  let circuit_params = FinalCircuitParams::from(&node_pp.augmented_circuit_params);
  let final_circuit = FinalCircuit::<E1>::new(
    &circuit_params,
    node_pp.ro_consts_circuit_primary.clone(),
    None,
  );
  PublicParams::setup(&final_circuit, &*default_ck_hint(), &*default_ck_hint())
}
#[derive(Debug, Clone, Deserialize, Serialize)]
struct L2<E>
where
  E: CurveCycleEquipped,
{
  r_U: RelaxedR1CSInstance<E>,
  r_W: RelaxedR1CSWitness<E>,
  rs: RecursiveSNARK<E>,
  IC_i: E::Scalar,
  i: usize,
}

impl<E> L2<E>
where
  E: CurveCycleEquipped,
{
  /// Create a new instance of [`L2`]
  pub fn new(
    final_pp: &PublicParams<E>,
    node_pp: &PublicParams<E>,
    node_rs: &RecursiveSNARK<E>,
  ) -> Result<L2<E>, NovaError> {
    let r_U = RelaxedR1CSInstance::default(
      &node_pp.ck_primary,
      &node_pp.circuit_shape_primary.r1cs_shape,
    );
    let r_W = RelaxedR1CSWitness::default(&node_pp.circuit_shape_primary.r1cs_shape);

    let node_pp_digest = node_pp.digest();
    let (l_U, l_W) = node_rs.U_W();
    let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
      &node_pp.ck_primary,
      &node_pp.ro_consts_primary,
      &node_pp_digest,
      &node_pp.circuit_shape_primary.r1cs_shape,
      &r_U,
      &r_W,
      l_U,
      l_W,
    )?;

    let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

    let data_p = RelaxedFoldingData::new(r_U.clone(), l_U.clone(), comm_T);

    // Do calculation's outside circuit, to be passed in as advice to F'
    let r_squared = r * r;
    let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
    let W_new = r_U.comm_W + l_U.comm_W * r;

    let final_circuit_inputs = FinalCircuitInputs::<E>::new(
      Some(scalar_as_base::<E>(node_pp_digest)),
      Some(data_p),
      Some(E_new),
      Some(W_new),
    );

    let circuit_params = FinalCircuitParams::from(&node_pp.augmented_circuit_params);

    let final_circuit = FinalCircuit::new(
      &circuit_params,
      node_pp.ro_consts_circuit_primary.clone(),
      Some(final_circuit_inputs),
    );

    let z0 = vec![E::Scalar::ZERO];
    let mut IC_i = E::Scalar::ZERO;

    let mut rs = RecursiveSNARK::new(final_pp, &final_circuit, &z0)?;

    rs.prove_step(final_pp, &final_circuit, IC_i)?;
    IC_i = rs.increment_commitment(final_pp, &final_circuit);

    Ok(Self {
      r_U,
      r_W,
      rs,
      IC_i,
      i: 0,
    })
  }

  /// updates the provided [`L2`] by executing a step of the incremental computation
  #[tracing::instrument(skip_all, name = "Layer2::prove_step")]
  pub fn prove_step(
    &mut self,
    final_pp: &PublicParams<E>,
    node_pp: &PublicParams<E>,
    node_rs: &RecursiveSNARK<E>,
  ) -> Result<(), NovaError> {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }
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
    let r_squared = r * r;
    let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
    let W_new = self.r_U.comm_W + l_U.comm_W * r;

    let final_circuit_inputs = FinalCircuitInputs::<E>::new(
      Some(scalar_as_base::<E>(node_pp_digest)),
      Some(data_p),
      Some(E_new),
      Some(W_new),
    );

    let circuit_params = FinalCircuitParams::from(&node_pp.augmented_circuit_params);

    let final_circuit = FinalCircuit::new(
      &circuit_params,
      node_pp.ro_consts_circuit_primary.clone(),
      Some(final_circuit_inputs),
    );

    self.rs.prove_step(final_pp, &final_circuit, self.IC_i)?;
    self.IC_i = self.rs.increment_commitment(final_pp, &final_circuit);
    self.r_U = r_U;
    self.r_W = r_W;

    Ok(())
  }

  /// Verifies the [`L2`] instance
  pub fn verify(&self, final_pp: &PublicParams<E>) -> Result<(), NovaError> {
    self
      .rs
      .verify(final_pp, self.rs.num_steps(), &[E::Scalar::ZERO], self.IC_i)?;

    Ok(())
  }
}

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
