//! IVC scheme with Hypernova
//!
//! This module implements a SNARK that proves the correct execution of an incremental computation
use std::sync::Arc;

use crate::constants::{
  BASE_CONSTRAINTS, MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT, MAX_CONSTRAINTS_PER_SUMCHECK_ROUND,
};
use crate::cyclefold::util::{absorb_primary_relaxed_r1cs, FoldingData};
use crate::digest::SimpleDigestible;
use crate::frontend::num::AllocatedNum;
use crate::frontend::{ConstraintSystem, SynthesisError};
use crate::hypernova::augmented_circuit::{project_aug_circuit_size, AugmentedCircuit};
use crate::nebula::traits::RecursiveSNARKFieldsTrait;
use crate::traits::commitment::CommitmentEngineTrait;
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_FE_IN_EMULATED_POINT, NUM_HASH_BITS},
  cyclefold::circuit::CycleFoldCircuit,
  errors::NovaError,
  frontend::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  gadgets::scalar_as_base,
  r1cs::{CommitmentKeyHint, R1CSInstance, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROConstantsCircuit, ROTrait},
  CommitmentKey, DigestComputer, ROConstants,
};
use crate::{AugmentedCircuitParams, Commitment, R1CSWithArity};
use ff::Field;
use ff::PrimeField;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

use super::StepCircuit;

/// The public parameters used in the CycleFold recursive SNARK proof and verification
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// The arity of the step circuit
  F_arity: usize,
  /// RO constants for primary circuit
  ro_consts: ROConstants<Dual<E>>,
  /// RO constants for primary circuit
  ro_consts_circuit: ROConstantsCircuit<Dual<E>>,
  /// Commitment key for primary circuit
  ck: Arc<CommitmentKey<E>>,
  /// R1CS shape we are arguing about
  circuit_shape: R1CSWithArity<E>,
  /// Parameters of big nats in circuit
  augmented_circuit_params: AugmentedCircuitParams,
  /// secondary commitment key
  ck_cyclefold: Arc<CommitmentKey<Dual<E>>>,
  /// R1CS shape of cyclefold circuit
  circuit_shape_cyclefold: R1CSWithArity<Dual<E>>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E> PublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// Builds the public parameters for the circuit `C1`.
  /// The same note for public parameter hints apply as in the case for Nova's public parameters:
  /// For some final compressing SNARKs the size of the commitment key must be larger, so we include
  /// `ck_hint_primary` and `ck_hint_cyclefold` parameters to accommodate this.
  #[tracing::instrument(skip_all, name = "nebula::PublicParams::setup")]
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
    let (r1cs_shape, ck) = cs.r1cs_shape(ck_hint);
    let ck = Arc::new(ck);
    let circuit_shape = R1CSWithArity::new(r1cs_shape, F_arity);

    // Get the structure for the CycleFold circuit and corresponding commitment key
    let mut cs: ShapeCS<Dual<E>> = ShapeCS::new();
    let circuit_cyclefold: CycleFoldCircuit<E> = CycleFoldCircuit::default();
    let _ = circuit_cyclefold.synthesize(&mut cs);
    let (r1cs_shape_cyclefold, ck_cyclefold) = cs.r1cs_shape(ck_hint_cyclefold);
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

impl<E1> SimpleDigestible for PublicParams<E1> where E1: CurveCycleEquipped {}
