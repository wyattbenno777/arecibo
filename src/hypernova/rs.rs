//! IVC scheme with Hypernova
//!
//! This module implements a SNARK that proves the correct execution of an incremental computation
use crate::{
  constants::{
    BASE_CONSTRAINTS, BN_LIMB_WIDTH, BN_N_LIMBS, DEFAULT_ABSORBS,
    MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT, MAX_CONSTRAINTS_PER_SUMCHECK_ROUND, NUM_HASH_BITS,
  },
  cyclefold::circuit::CycleFoldCircuit,
  digest::SimpleDigestible,
  errors::NovaError,
  frontend::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
    test_cs::TestConstraintSystem,
    ConstraintSystem, SynthesisError,
  },
  gadgets::scalar_as_base,
  hypernova::{
    augmented_circuit::{project_aug_circuit_size, AugmentedCircuit, AugmentedCircuitInputs},
    nifs::NIFS,
  },
  r1cs::{CommitmentKeyHint, LR1CSInstance, R1CSInstance, R1CSWitness},
  traits::{CurveCycleEquipped, Dual, Engine, ROConstantsCircuit, ROTrait},
  AugmentedCircuitParams, CommitmentKey, DigestComputer, R1CSWithArity, ROConstants,
};
use ff::Field;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
  num_rounds: usize,
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

impl<E> SimpleDigestible for PublicParams<E> where E: CurveCycleEquipped {}

/// A SNARK that proves the correct execution of an incremental computation in the CycleFold folding
/// scheme.
///
/// (U, W, u, w) -> IVC Proof
/// (i, z0, zi) -> Statement being proven
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  r_U: LR1CSInstance<E>,
  r_W: R1CSWitness<E>,
  l_u: R1CSInstance<E>,
  l_w: R1CSWitness<E>,
  z0: Vec<E::Scalar>,
  i: usize,
  zi: Vec<E::Scalar>,
}

impl<E> RecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  /// Create a new instance of RecursiveSNARK
  #[tracing::instrument(skip_all, name = "nebula::RecursiveSNARK::new")]
  pub fn new<C>(pp: &PublicParams<E>, step_circuit: &C, z0: &[E::Scalar]) -> Result<Self, NovaError>
  where
    C: StepCircuit<E::Scalar>,
  {
    if z0.len() != pp.F_arity {
      return Err(NovaError::InvalidInitialInputLength);
    }

    // Get default running primary instance and witness pair
    let r1cs = &pp.circuit_shape.r1cs_shape;
    let r_U = LR1CSInstance::default(r1cs);
    let r_W = R1CSWitness::default(r1cs);

    // Base case for F'
    //
    // Get the new instance-witness pair to be folded into running instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let inputs: AugmentedCircuitInputs<E> = AugmentedCircuitInputs::new(
      pp.digest(),
      E::Scalar::ZERO,
      z0.to_vec(),
      None,
      None,
      None,
      None,
      None,
    );
    let circuit = AugmentedCircuit::new(
      &pp.augmented_circuit_params,
      pp.ro_consts_circuit.clone(),
      Some(inputs),
      step_circuit,
      pp.num_rounds,
    );
    let zi = circuit.synthesize(&mut cs)?;
    let (l_u, l_w) = cs.r1cs_instance_and_witness(r1cs, &pp.ck)?;

    // Get z_i values out of the constraint system
    let zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;

    Ok(Self {
      r_W,
      r_U,
      l_w,
      l_u,
      z0: z0.to_vec(),
      i: 0,
      zi,
    })
  }

  /// Create a new [`RecursiveSNARK`] (or updates the provided [`RecursiveSNARK`])
  /// by executing a step of the incremental computation
  #[tracing::instrument(skip_all, name = "nebula::RecursiveSNARK::prove_step")]
  pub fn prove_step<C>(&mut self, pp: &PublicParams<E>, step_circuit: &C) -> Result<(), NovaError>
  where
    C: StepCircuit<E::Scalar>,
  {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // Parse Πi (self) as ((Ui, Wi), (ui, wi)) and then:
    //
    // 1. compute (Ui+1,Wi+1,T) ← NIFS.P(pk,(Ui,Wi),(ui,wi)),
    let (nifs, (r_U, r_W)) = NIFS::prove(
      &pp.circuit_shape.r1cs_shape,
      &pp.ro_consts,
      &pp.digest(),
      (&self.r_U, &self.r_W),
      (&self.l_u, &self.l_w),
    )?;

    // 2. compute (ui+1, wi+1) ← trace(F ′, (vk, Ui, ui, (i, z0, zi), ωi, T )),
    let mut cs = SatisfyingAssignment::<E>::new();
    let inputs: AugmentedCircuitInputs<E> = AugmentedCircuitInputs::new(
      pp.digest(),
      E::Scalar::from(self.i as u64),
      self.z0.to_vec(),
      Some(self.zi.clone()),
      Some(nifs),
      Some(self.r_U.clone()),
      Some(self.l_u.clone()),
      Some(r_U.comm_W),
    );
    let circuit = AugmentedCircuit::new(
      &pp.augmented_circuit_params,
      pp.ro_consts_circuit.clone(),
      Some(inputs),
      step_circuit,
      pp.num_rounds,
    );
    let zi = circuit.synthesize(&mut cs)?;
    let (l_u, l_w) = cs.r1cs_instance_and_witness(&pp.circuit_shape.r1cs_shape, &pp.ck)?;

    // 3. output Πi+1 ← ((Ui+1, Wi+1), (ui+1, wi+1)).
    self.r_U = r_U;
    self.r_W = r_W;
    self.l_u = l_u;
    self.l_w = l_w;

    // Update statement being proven
    self.zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;
    self.i += 1;

    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  #[tracing::instrument(skip_all, name = "nebula::RecursiveSNARK::verify")]
  pub fn verify(
    &self,
    pp: &PublicParams<E>,
    num_steps: usize,
    z0: &[E::Scalar],
  ) -> Result<Vec<E::Scalar>, NovaError> {
    // Basic checks for IVC proof
    // //////
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;
    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;
    // check if the initial inputs match
    let is_inputs_not_match = self.z0 != z0;
    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outputs = self.r_U.X.len() != 2;
    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outputs
    {
      return Err(NovaError::ProofVerifyError);
    }

    // Hash check
    //
    // Verify the hashes equal the public IO for the final primary instance
    let mut ro = <Dual<E> as Engine>::RO::new(pp.ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp.digest());
    ro.absorb(E::Scalar::from(num_steps as u64));
    for e in z0 {
      ro.absorb(*e);
    }
    for e in &self.zi {
      ro.absorb(*e);
    }
    self.r_U.absorb_in_ro(&mut ro);
    let hash = ro.squeeze(NUM_HASH_BITS);
    if scalar_as_base::<Dual<E>>(hash) != self.l_u.X[0] {
      return Err(NovaError::ProofVerifyError);
    }

    // Verify the satisfiability of running relaxed instances, and the final primary instance.
    let (res_r_U, res_l_u) = rayon::join(
      || {
        pp.circuit_shape
          .r1cs_shape
          .is_sat_linearized(&pp.ck, &self.r_U, &self.r_W)
      },
      || {
        pp.circuit_shape
          .r1cs_shape
          .is_sat(&pp.ck, &self.l_u, &self.l_w)
      },
    );
    res_r_U?;
    res_l_u?;

    Ok(self.zi.to_vec())
  }
}

#[allow(dead_code)]
fn debug_step<E, SC>(circuit: AugmentedCircuit<'_, E, SC>) -> Result<(), NovaError>
where
  E: CurveCycleEquipped,
  SC: StepCircuit<E::Scalar>,
{
  let mut cs = TestConstraintSystem::<E::Scalar>::new();
  circuit
    .synthesize(&mut cs)
    .map_err(|_| NovaError::from(SynthesisError::AssignmentMissing))?;
  let is_sat = cs.is_satisfied();
  if !is_sat {
    assert!(is_sat);
  }
  Ok(())
}

#[cfg(test)]
mod test {
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    hypernova::StepCircuit,
    provider::Bn256EngineIPA,
    traits::{snark::default_ck_hint, CurveCycleEquipped},
    NovaError,
  };
  use ff::PrimeField;
  use std::marker::PhantomData;

  use super::RecursiveSNARK;

  #[derive(Clone)]
  struct SquareCircuit<F> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> StepCircuit<F> for SquareCircuit<F> {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      Ok(vec![x_sq])
    }
  }

  fn test_rs_with<E: CurveCycleEquipped>() -> Result<(), NovaError> {
    let circuit = SquareCircuit::<E::Scalar> { _p: PhantomData };
    let pp = super::PublicParams::<E>::setup(&circuit, &*default_ck_hint(), &*default_ck_hint());
    let z0 = vec![E::Scalar::from(2u64)];
    let mut recursive_snark = RecursiveSNARK::new(&pp, &circuit, &z0).unwrap();
    for i in 0..10 {
      recursive_snark.prove_step(&pp, &circuit)?;
      recursive_snark.verify(&pp, i + 1, &z0).unwrap();
    }
    Ok(())
  }

  #[test]
  fn test_rs() -> Result<(), NovaError> {
    test_rs_with::<Bn256EngineIPA>()
  }
}
