//! This module defines the Public params for HyperNova + Nebula

use super::{
  augmented_circuit::{project_aug_circuit_size, AugmentedCircuit},
  rs::StepCircuit,
};
use crate::{
  constants::{
    BASE_CONSTRAINTS, BN_LIMB_WIDTH, BN_N_LIMBS, MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT,
    MAX_CONSTRAINTS_PER_SUMCHECK_ROUND,
  },
  cyclefold::circuit::CycleFoldCircuit,
  digest::{DigestComputer, SimpleDigestible},
  frontend::{r1cs::NovaShape, shape_cs::ShapeCS},
  r1cs::{commitment_key_size, CommitmentKeyHint},
  traits::{
    commitment::CommitmentEngineTrait, CurveCycleEquipped, Dual, Engine, ROConstants,
    ROConstantsCircuit,
  },
  AugmentedCircuitParams, CommitmentKey, R1CSWithArity,
};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The public parameters used in the HyperNova recursiveSNARK proving and verification
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// R1CS shape we are arguing about
  pub circuit_shape: R1CSWithArity<E>,
  /// The arity of the step circuit
  pub F_arity: usize,
  /// Digest of the public parameters
  #[serde(skip, default = "OnceCell::new")]
  pub digest: OnceCell<E::Scalar>,
  /// Number of sumcheck rounds used in the NIFS for the augmented circuit
  pub num_rounds: usize,
  /// RO constants for primary circuit
  pub ro_consts: ROConstants<Dual<E>>,
  /// RO constants for primary circuit
  pub ro_consts_circuit: ROConstantsCircuit<Dual<E>>,
  /// Commitment key for primary circuit
  pub ck: Arc<CommitmentKey<E>>,
  /// Parameters of big nats in circuit
  pub augmented_circuit_params: AugmentedCircuitParams,
  /// secondary commitment key
  pub ck_cyclefold: Arc<CommitmentKey<Dual<E>>>,
  /// R1CS shape of cyclefold circuit
  pub circuit_shape_cyclefold: R1CSWithArity<Dual<E>>,
}

impl<E> PublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// Builds the public parameters for the circuit `C1`.
  /// The same note for public parameter hints apply as in the case for Nova's public parameters:
  /// For some final compressing SNARKs the size of the commitment key must be larger, so we include
  /// `ck_hint_primary` and `ck_hint_cyclefold` parameters to accommodate this.
  #[tracing::instrument(skip_all, name = "HyperNova::PublicParams::setup")]
  pub fn setup(
    step_circuit: &impl StepCircuit<E::Scalar>,
    ck_hint: &CommitmentKeyHint<E>,
    ck_hint_cyclefold: &CommitmentKeyHint<Dual<E>>,
  ) -> Self {
    // This value is used to validate inputs to API
    let F_arity = step_circuit.arity();

    // Get the round constants used in the poseidon hash function and poseidon hash function circuit
    let ro_consts = ROConstants::<Dual<E>>::default();
    let ro_consts_circuit = ROConstantsCircuit::<Dual<E>>::default();

    // Get the structure for the AugmentedCircuit and corresponding commitment key
    let num_rounds = project_aug_circuit_size::<E>(
      BASE_CONSTRAINTS,
      MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT,
      MAX_CONSTRAINTS_PER_SUMCHECK_ROUND,
      step_circuit,
    );
    let augmented_circuit_params = AugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS);
    let circuit: AugmentedCircuit<'_, E, _> = AugmentedCircuit::new(
      &augmented_circuit_params,
      ro_consts_circuit.clone(),
      None,
      step_circuit,
      num_rounds,
    );
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (r1cs_shape, ck) = cs.r1cs_shape_and_key(ck_hint);
    let ck = Arc::new(ck);
    let circuit_shape = R1CSWithArity::new(r1cs_shape, F_arity);

    // Get the structure for the CycleFold circuit and corresponding commitment key
    let mut cs: ShapeCS<Dual<E>> = ShapeCS::new();
    let circuit_cyclefold: CycleFoldCircuit<E> = CycleFoldCircuit::default();
    let _ = circuit_cyclefold.synthesize(&mut cs);
    let (r1cs_shape_cyclefold, ck_cyclefold) = cs.r1cs_shape_and_key(ck_hint_cyclefold);
    let ck_cyclefold = Arc::new(ck_cyclefold);
    let circuit_shape_cyclefold = R1CSWithArity::new(r1cs_shape_cyclefold, 0);
    Self {
      F_arity,
      ro_consts,
      ro_consts_circuit,
      ck,
      circuit_shape,
      augmented_circuit_params,
      ck_cyclefold,
      circuit_shape_cyclefold,
      digest: OnceCell::new(),
      num_rounds,
    }
  }

  /// Calculate the digest of the public parameters.
  pub fn digest(&self) -> E::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure in retrieving digest")
  }
}

/// Public params trait.
/// This trait defines the limited behavior of the public parameters used in the HyperNova recursiveSNARK proving and verification.
/// The behavior of the public parameters has no logic and simply returns the values of the public parameters.
pub trait PublicParamsTrait<E>
where
  E: CurveCycleEquipped,
  Self: Send + Sync,
{
  /// R1CS shape we are arguing about
  fn circuit_shape(&self) -> &R1CSWithArity<E>;
  /// The arity of the step circuit
  fn F_arity(&self) -> usize;
  /// Number of sumcheck rounds used in the NIFS for the augmented circuit
  fn num_rounds(&self) -> usize;
  /// RO constants for primary circuit
  fn ro_consts(&self) -> &ROConstants<Dual<E>>;
  /// RO constants for primary circuit
  fn ro_consts_circuit(&self) -> &ROConstantsCircuit<Dual<E>>;
  /// Commitment key for primary circuit
  fn ck(&self) -> &Arc<CommitmentKey<E>>;
  /// Parameters of big nats in circuit
  fn augmented_circuit_params(&self) -> &AugmentedCircuitParams;
  /// secondary commitment key
  fn ck_cyclefold(&self) -> &Arc<CommitmentKey<Dual<E>>>;
  /// R1CS shape of cyclefold circuit
  fn circuit_shape_cyclefold(&self) -> &R1CSWithArity<Dual<E>>;
  /// Calculate the digest of the public parameters.
  fn digest(&self) -> E::Scalar;
}

impl<E> PublicParamsTrait<E> for PublicParams<E>
where
  E: CurveCycleEquipped,
{
  fn circuit_shape(&self) -> &R1CSWithArity<E> {
    &self.circuit_shape
  }
  fn F_arity(&self) -> usize {
    self.F_arity
  }
  fn num_rounds(&self) -> usize {
    self.num_rounds
  }
  fn ro_consts(&self) -> &ROConstants<Dual<E>> {
    &self.ro_consts
  }
  fn ro_consts_circuit(&self) -> &ROConstantsCircuit<Dual<E>> {
    &self.ro_consts_circuit
  }
  fn ck(&self) -> &Arc<CommitmentKey<E>> {
    &self.ck
  }
  fn augmented_circuit_params(&self) -> &AugmentedCircuitParams {
    &self.augmented_circuit_params
  }
  fn ck_cyclefold(&self) -> &Arc<CommitmentKey<Dual<E>>> {
    &self.ck_cyclefold
  }
  fn circuit_shape_cyclefold(&self) -> &R1CSWithArity<Dual<E>> {
    &self.circuit_shape_cyclefold
  }

  fn digest(&self) -> E::Scalar {
    self.digest()
  }
}

impl<E> SimpleDigestible for PublicParams<E> where E: CurveCycleEquipped {}

/// Split public params
pub type SplitPublicParams<'a, E> = (
  &'a AuxPublicParams<E>,
  &'a R1CSPublicParams<E>,
  <E as Engine>::Scalar,
);

impl<E> PublicParamsTrait<E> for SplitPublicParams<'_, E>
where
  E: CurveCycleEquipped,
{
  fn circuit_shape(&self) -> &R1CSWithArity<E> {
    &self.1.circuit_shape
  }
  fn F_arity(&self) -> usize {
    self.1.F_arity
  }
  fn num_rounds(&self) -> usize {
    self.1.num_rounds
  }
  fn ro_consts(&self) -> &ROConstants<Dual<E>> {
    &self.0.ro_consts
  }
  fn ro_consts_circuit(&self) -> &ROConstantsCircuit<Dual<E>> {
    &self.0.ro_consts_circuit
  }
  fn ck(&self) -> &Arc<CommitmentKey<E>> {
    &self.0.ck
  }
  fn augmented_circuit_params(&self) -> &AugmentedCircuitParams {
    &self.0.augmented_circuit_params
  }
  fn ck_cyclefold(&self) -> &Arc<CommitmentKey<Dual<E>>> {
    &self.0.ck_cyclefold
  }
  fn circuit_shape_cyclefold(&self) -> &R1CSWithArity<Dual<E>> {
    &self.0.circuit_shape_cyclefold
  }

  fn digest(&self) -> E::Scalar {
    self.2
  }
}

impl<E> SimpleDigestible for SplitPublicParams<'_, E> where E: CurveCycleEquipped {}

/// Circuit specific public parameters for the circuit
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSPublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// R1CS shape we are arguing about
  pub circuit_shape: R1CSWithArity<E>,
  /// The arity of the step circuit
  pub F_arity: usize,
  /// Number of sumcheck rounds used in the NIFS for the augmented circuit
  pub num_rounds: usize,
}

impl<E> R1CSPublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// Builds circuit specific public parameters for the circuit
  #[tracing::instrument(skip_all, name = "HyperNova::PublicParams::setup")]
  pub fn setup(
    step_circuit: &impl StepCircuit<E::Scalar>,
    ro_consts_circuit: &ROConstantsCircuit<Dual<E>>,
    augmented_circuit_params: &AugmentedCircuitParams,
  ) -> Self {
    // This value is used to validate inputs to API
    let F_arity = step_circuit.arity();

    // Get the structure for the AugmentedCircuit and corresponding commitment key
    let num_rounds = project_aug_circuit_size::<E>(
      BASE_CONSTRAINTS,
      MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT,
      MAX_CONSTRAINTS_PER_SUMCHECK_ROUND,
      step_circuit,
    );
    let circuit: AugmentedCircuit<'_, E, _> = AugmentedCircuit::new(
      augmented_circuit_params,
      ro_consts_circuit.clone(),
      None,
      step_circuit,
      num_rounds,
    );
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let r1cs_shape = cs.r1cs_shape();
    let circuit_shape = R1CSWithArity::new(r1cs_shape, F_arity);
    Self {
      F_arity,
      circuit_shape,
      num_rounds,
    }
  }
}

impl<E> SimpleDigestible for R1CSPublicParams<E> where E: CurveCycleEquipped {}

/// Auxillary public parameters for ck, ro and cyclefold circuit
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AuxPublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// Commitment key for primary circuit
  pub ck: Arc<CommitmentKey<E>>,
  /// RO constants for primary circuit
  pub ro_consts: ROConstants<Dual<E>>,
  /// RO constants for primary circuit
  pub ro_consts_circuit: ROConstantsCircuit<Dual<E>>,
  /// Parameters of big nats in circuit
  pub augmented_circuit_params: AugmentedCircuitParams,
  /// secondary commitment key
  pub ck_cyclefold: Arc<CommitmentKey<Dual<E>>>,
  /// R1CS shape of cyclefold circuit
  pub circuit_shape_cyclefold: R1CSWithArity<Dual<E>>,
}

impl<E> AuxPublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// Builds auxillary public parameters for the circuit, which includes the
  /// commitment key, RO constants and cyclefold circuit
  pub fn setup(
    circuit_params: &[&R1CSWithArity<E>],
    ro_consts_circuit: ROConstantsCircuit<Dual<E>>,
    augmented_circuit_params: AugmentedCircuitParams,
    ck_hint: &CommitmentKeyHint<E>,
    ck_hint_cyclefold: &CommitmentKeyHint<Dual<E>>,
  ) -> Self {
    // Get the round constants used in the poseidon hash function and poseidon hash function circuit
    let ro_consts = ROConstants::<Dual<E>>::default();

    // Get the structure for the CycleFold circuit and corresponding commitment key
    let mut cs: ShapeCS<Dual<E>> = ShapeCS::new();
    let circuit_cyclefold: CycleFoldCircuit<E> = CycleFoldCircuit::default();
    let _ = circuit_cyclefold.synthesize(&mut cs);
    let (r1cs_shape_cyclefold, ck_cyclefold) = cs.r1cs_shape_and_key(ck_hint_cyclefold);
    let ck_cyclefold = Arc::new(ck_cyclefold);
    let circuit_shape_cyclefold = R1CSWithArity::new(r1cs_shape_cyclefold, 0);
    let ck = Self::compute_ck(circuit_params, ck_hint);
    Self {
      ck: Arc::new(ck),
      ro_consts,
      ro_consts_circuit,
      augmented_circuit_params,
      ck_cyclefold,
      circuit_shape_cyclefold,
    }
  }

  /// Compute primary and secondary commitment keys sized to handle the largest of the circuits in the provided
  /// `R1CSWithArity`.
  pub fn compute_ck(
    circuit_params: &[&R1CSWithArity<E>],
    ck_hint1: &CommitmentKeyHint<E>,
  ) -> CommitmentKey<E>
  where
    E: CurveCycleEquipped,
  {
    let size_primary = circuit_params
      .iter()
      .map(|circuit| commitment_key_size(&circuit.r1cs_shape, ck_hint1))
      .max()
      .unwrap();
    E::CE::setup(b"ck", size_primary)
  }
}

impl<E> SimpleDigestible for AuxPublicParams<E> where E: CurveCycleEquipped {}
