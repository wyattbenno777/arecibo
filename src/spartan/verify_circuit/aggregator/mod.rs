//! Contains the data structures needed to aggregate Spartan proofs
use std::sync::Arc;

use bellpepper_core::{num::AllocatedNum, SynthesisError};
use serde::{Deserialize, Serialize};

use super::{gadgets::nonnative::ipa::EvaluationEngineGadget, ipa_prover_poseidon::batched};
use crate::bellpepper::r1cs::{NovaShape, NovaWitness};
use crate::bellpepper::solver::SatisfyingAssignment;
use crate::r1cs::{CommitmentKeyHint, R1CSShape, RelaxedR1CSWitness};
use crate::{
  bellpepper::shape_cs::ShapeCS,
  errors::NovaError,
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
  r1cs::RelaxedR1CSInstance,
  spartan::{verify_circuit::circuit::batched::SpartanVerifyCircuit, PolyEvalInstance},
  traits::{snark::RelaxedR1CSSNARKTrait, CurveCycleEquipped, Dual, Engine},
  CommitmentKey,
};
use bellpepper_core::ConstraintSystem;

#[cfg(test)]
mod tests;

/// Necessary public values needed for both proving and verifying
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<E1>
where
  E1: CurveCycleEquipped,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  r1cs_shape_primary: R1CSShape<E1>,
  ck_primary: Arc<CommitmentKey<E1>>,
  r1cs_shape_secondary: R1CSShape<Dual<E1>>,
  ck_secondary: Arc<CommitmentKey<Dual<E1>>>,
}

impl<E1> PublicParams<E1>
where
  E1: CurveCycleEquipped,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  ///  Set up builder to create `PublicParams` for a pair of circuits (one for primary curve and one for secondary curve)
  pub fn setup(
    snarks_data: &[AggregatorSNARKData<'_, E1>],
    ck_hint1: &CommitmentKeyHint<E1>,
    ck_hint2: &CommitmentKeyHint<Dual<E1>>,
  ) -> Result<Self, NovaError> {
    // Build IOP circuit and FFA circuits
    let circuits = build_circuits(snarks_data)?;

    // Constraint System for each scalar operations and base filed operations
    let mut cs_primary: ShapeCS<E1> = ShapeCS::new();
    let mut cs_secondary: ShapeCS<Dual<E1>> = ShapeCS::new();

    // Run through all circuits to build one big R1CS
    for (i, (iop_circuit, ffa_circuit)) in circuits.iter().enumerate() {
      let _ = iop_circuit.synthesize(cs_primary.namespace(|| format!("IOP {i}")));
      let _ = ffa_circuit.synthesize(cs_secondary.namespace(|| format!("FFA {i}")));
    }

    // Get primary R1CS shape and commitment key
    let (r1cs_shape_primary, ck_primary) = cs_primary.r1cs_shape_and_key(ck_hint1);
    let ck_primary = Arc::new(ck_primary);

    // Get secondary R1CS shape and commitment key
    let (r1cs_shape_secondary, ck_secondary) = cs_secondary.r1cs_shape_and_key(ck_hint2);
    let ck_secondary = Arc::new(ck_secondary);

    Ok(PublicParams {
      r1cs_shape_primary,
      ck_primary,
      r1cs_shape_secondary,
      ck_secondary,
    })
  }
}

/// Resulting SNARK after aggregating child SNARKS
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AggregatedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  snark_primary: S1,
  U_primary: RelaxedR1CSInstance<E1>,
  snark_secondary: S2,
  U_secondary: RelaxedR1CSInstance<Dual<E1>>,
}

impl<E1, S1, S2> AggregatedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
  E1::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  /// Creates prover and verifier keys for `AggregatedSNARK`
  pub fn setup(
    pp: &PublicParams<E1>,
  ) -> Result<(ProverKey<E1, S1, S2>, VerifierKey<E1, S1, S2>), NovaError> {
    // Setup prover & verifier key for curve cycle.
    let (pk_primary, vk_primary) = S1::setup(pp.ck_primary.clone(), &pp.r1cs_shape_primary)?;
    let (pk_secondary, vk_secondary) =
      S2::setup(pp.ck_secondary.clone(), &pp.r1cs_shape_secondary)?;

    let pk = ProverKey {
      pk_primary,
      pk_secondary,
    };

    let vk = VerifierKey {
      vk_primary,
      vk_secondary,
    };

    Ok((pk, vk))
  }

  /// Create a new `AggregatedSNARK`
  pub fn prove(
    pp: &PublicParams<E1>,
    pk: &ProverKey<E1, S1, S2>,
    snarks_data: &[AggregatorSNARKData<'_, E1>],
  ) -> Result<Self, NovaError> {
    let circuits = build_circuits(snarks_data)?;
    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let mut cs_secondary = SatisfyingAssignment::<Dual<E1>>::new();

    for (i, (iop_circuit, ffa_circuit)) in circuits.iter().enumerate() {
      let _ = iop_circuit.synthesize(cs_primary.namespace(|| format!("IOP {i}")));
      let _ = ffa_circuit.synthesize(cs_secondary.namespace(|| format!("FFA {i}")));
    }

    // Get instance and witness for primary curve
    let (U_primary, W_primary) =
      cs_primary.r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)?;
    let U_primary =
      RelaxedR1CSInstance::from_r1cs_instance(&*pp.ck_primary, &pp.r1cs_shape_primary, U_primary);
    let W_primary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, W_primary);

    // get final primary snark
    let snark_primary = S1::prove(
      &pp.ck_primary,
      &pk.pk_primary,
      &pp.r1cs_shape_primary,
      &U_primary,
      &W_primary,
    )?;

    // Get instance and witness for secondary curve
    let (U_secondary, W_secondary) =
      cs_secondary.r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)?;
    let U_secondary = RelaxedR1CSInstance::from_r1cs_instance(
      &*pp.ck_secondary,
      &pp.r1cs_shape_secondary,
      U_secondary,
    );
    let W_secondary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_secondary, W_secondary);

    // get final secondary snark
    let snark_secondary = S2::prove(
      &pp.ck_secondary,
      &pk.pk_secondary,
      &pp.r1cs_shape_secondary,
      &U_secondary,
      &W_secondary,
    )?;

    Ok(Self {
      snark_primary,
      U_primary,
      snark_secondary,
      U_secondary,
    })
  }

  /// Verify the AggregatedSNARK
  pub fn verify(&self, vk: &VerifierKey<E1, S1, S2>) -> Result<(), NovaError> {
    let _ = self.snark_primary.verify(&vk.vk_primary, &self.U_primary);
    self
      .snark_secondary
      .verify(&vk.vk_secondary, &self.U_secondary)
  }
}

/// Prover Key for Aggregation proving system

#[derive(Clone, Debug)]
pub struct ProverKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  pk_primary: S1::ProverKey,
  pk_secondary: S2::ProverKey,
}

#[derive(Debug, Clone, Serialize)]
#[serde(bound = "")]

/// Verifier Key for Aggregation proving system
pub struct VerifierKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  vk_primary: S1::VerifierKey,
  vk_secondary: S2::VerifierKey,
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

#[derive(Clone)]
struct IOPCircuit<'a, E: Engine> {
  snark_data: &'a AggregatorSNARKData<'a, E>,
}

impl<'a, E: Engine> IOPCircuit<'a, E> {
  pub fn new(snark_data: &'a AggregatorSNARKData<'a, E>) -> Result<Self, NovaError> {
    Ok(Self { snark_data })
  }
}

impl<'a, E: Engine> IOPCircuit<'a, E> {
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    SpartanVerifyCircuit::synthesize(
      cs.namespace(|| "verify IOP"),
      &self.snark_data.vk,
      &self.snark_data.U,
      &self.snark_data.snark,
    )?;
    Ok(vec![])
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

impl<'a, E1> FFACircuit<'a, E1>
where
  E1: CurveCycleEquipped,
  <E1 as Engine>::GE: DlogGroup,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  fn synthesize<CS: ConstraintSystem<E1::Base>>(
    &self,
    mut cs: CS,
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
    Ok(vec![])
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
