//! IVC scheme with Hypernova
//!
//! This module implements a SNARK that proves the correct execution of an incremental computation.

use crate::{
  constants::{
    BASE_CONSTRAINTS, BN_LIMB_WIDTH, BN_N_LIMBS, DEFAULT_ABSORBS,
    MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT, MAX_CONSTRAINTS_PER_SUMCHECK_ROUND, NUM_HASH_BITS,
  },
  cyclefold::{circuit::CycleFoldCircuit, util::FoldingData},
  digest::SimpleDigestible,
  errors::NovaError,
  frontend::{
    num::AllocatedNum,
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
    ConstraintSystem, SynthesisError,
  },
  gadgets::scalar_as_base,
  hypernova::{
    augmented_circuit::{project_aug_circuit_size, AugmentedCircuit, AugmentedCircuitInputs},
    nifs::NIFS,
  },
  r1cs::{
    split::{LR1CSInstance, SplitR1CSInstance, SplitR1CSWitness},
    CommitmentKeyHint, RelaxedR1CSInstance, RelaxedR1CSWitness,
  },
  traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROConstantsCircuit, ROTrait},
  AugmentedCircuitParams, CommitmentKey, DigestComputer, R1CSWithArity, ROConstants,
};
use ff::{Field, PrimeField};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::nebula::ic::increment_comm;

/// A type that represents the carried commitments for this commitment-carrying HyperNova IVC scheme.
pub type IncrementalCommitment<E> = (<E as Engine>::Scalar, <E as Engine>::Scalar);

/// The public parameters used in the HyperNova recursiveSNARK proving and verification
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// The arity of the step circuit
  pub F_arity: usize,
  /// RO constants for primary circuit
  pub ro_consts: ROConstants<Dual<E>>,
  /// RO constants for primary circuit
  pub ro_consts_circuit: ROConstantsCircuit<Dual<E>>,
  /// Commitment key for primary circuit
  pub ck: Arc<CommitmentKey<E>>,
  /// R1CS shape we are arguing about
  pub circuit_shape: R1CSWithArity<E>,
  /// Parameters of big nats in circuit
  pub augmented_circuit_params: AugmentedCircuitParams,
  /// secondary commitment key
  pub ck_cyclefold: Arc<CommitmentKey<Dual<E>>>,
  /// R1CS shape of cyclefold circuit
  pub circuit_shape_cyclefold: R1CSWithArity<Dual<E>>,
  /// Digest of the public parameters
  #[serde(skip, default = "OnceCell::new")]
  pub digest: OnceCell<E::Scalar>,
  /// Number of sumcheck rounds used in the NIFS for the augmented circuit
  pub num_rounds: usize,
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

/// A SNARK that proves the correct execution of an incremental computation. HyperNova IVC scheme (with CycleFold).
///
/// * (U, W, u, w) -> IVC Proof
/// * (i, z_0, z_i) -> Statement being proven
/// * Carries two commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  r_U: LR1CSInstance<E>,
  r_W: SplitR1CSWitness<E>,
  l_u: SplitR1CSInstance<E>,
  l_w: SplitR1CSWitness<E>,
  r_U_cyclefold: RelaxedR1CSInstance<Dual<E>>,
  r_W_cyclefold: RelaxedR1CSWitness<Dual<E>>,
  z_0: Vec<E::Scalar>,
  i: usize,
  z_i: Vec<E::Scalar>,
  prev_ic: IncrementalCommitment<E>,
}

impl<E> RecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  /// Create a new instance of [`RecursiveSNARK`]
  #[tracing::instrument(skip_all, name = "HyperNova::RecursiveSNARK::new")]
  pub fn new<C>(
    pp: &PublicParams<E>,
    step_circuit: &C,
    z_0: &[E::Scalar],
  ) -> Result<Self, NovaError>
  where
    C: StepCircuit<E::Scalar>,
  {
    if z_0.len() != pp.F_arity {
      return Err(NovaError::InvalidInitialInputLength);
    }

    // Get default running primary instance and witness pair
    let r1cs = &pp.circuit_shape.r1cs_shape;
    let r_U = LR1CSInstance::default(r1cs);
    let r_W = SplitR1CSWitness::default(r1cs);

    // Base case for F'
    //
    // Get the new instance-witness pair to be folded into running instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let inputs: AugmentedCircuitInputs<E> = AugmentedCircuitInputs::new(
      pp.digest(),
      E::Scalar::ZERO,
      z_0.to_vec(),
      None,
      None,
      None,
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
    let z_i = circuit.synthesize(&mut cs)?;
    let (l_u, l_w) = cs.split_r1cs_instance_and_witness(r1cs, &pp.ck)?;

    // Get z_i values out of the constraint system
    let z_i = z_i
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;

    // Get the running CycleFold instance and witness pair
    let r1cs_cyclefold = &pp.circuit_shape_cyclefold.r1cs_shape;
    let r_U_cyclefold = RelaxedR1CSInstance::default(&*pp.ck_cyclefold, r1cs_cyclefold);
    let r_W_cyclefold = RelaxedR1CSWitness::default(r1cs_cyclefold);

    Ok(Self {
      r_W,
      r_U,
      l_w,
      l_u,
      r_W_cyclefold,
      r_U_cyclefold,
      z_0: z_0.to_vec(),
      i: 0,
      z_i,
      prev_ic: (E::Scalar::ZERO, E::Scalar::ZERO),
    })
  }

  /// Create a new [`RecursiveSNARK`] (or updates the provided [`RecursiveSNARK`])
  /// by executing a step of the incremental computation
  #[tracing::instrument(skip_all, name = "HyperNova::RecursiveSNARK::prove_step")]
  pub fn prove_step<C>(
    &mut self,
    pp: &PublicParams<E>,
    step_circuit: &C,
    ic: IncrementalCommitment<E>,
  ) -> Result<(), NovaError>
  where
    C: StepCircuit<E::Scalar>,
  {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // Parse u_i.C_W as (C_ωi−1 , C_aux_i−1). Abort if C_i != hash(C_i−1, C_ωi−1)
    self.ic_check(pp, ic)?;

    // Parse Πi (self) as ((Ui, Wi), (ui, wi)) and then:
    //
    // 1. compute (Ui+1,Wi+1,T) ← NIFS.P(pk,(Ui,Wi),(ui,wi)),
    let (nifs, (r_U, r_W), (r_U_cyclefold, r_W_cyclefold)) = NIFS::prove(
      (
        &pp.circuit_shape.r1cs_shape,
        &pp.circuit_shape_cyclefold.r1cs_shape,
      ),
      &pp.ck_cyclefold,
      &pp.ro_consts,
      &pp.digest(),
      (&self.r_U, &self.r_W),
      (&self.l_u, &self.l_w),
      (&self.r_U_cyclefold, &self.r_W_cyclefold),
    )?;

    // 2. compute (ui+1, wi+1) ← trace(F ′, (vk, Ui, ui, (i, z_0, z_i), ωi, T )),
    let mut cs = SatisfyingAssignment::<E>::new();
    let cyclefold_data = FoldingData::new(
      self.r_U_cyclefold.clone(),
      nifs.cyclefold_nifs.l_u.clone(),
      nifs.cyclefold_nifs.comm_T,
    );
    let inputs: AugmentedCircuitInputs<E> = AugmentedCircuitInputs::new(
      pp.digest(),
      E::Scalar::from(self.i as u64),
      self.z_0.to_vec(),
      Some(self.z_i.clone()),
      Some(nifs),
      Some(self.r_U.clone()),
      Some(self.l_u.clone()),
      Some(r_U.comm_W),
      Some(cyclefold_data),
      Some(r_U.pre_committed.0),
      Some(r_U.pre_committed.1),
    );
    let circuit = AugmentedCircuit::new(
      &pp.augmented_circuit_params,
      pp.ro_consts_circuit.clone(),
      Some(inputs),
      step_circuit,
      pp.num_rounds,
    );
    let z_i = circuit.synthesize(&mut cs)?;
    let (l_u, l_w) = cs.split_r1cs_instance_and_witness(&pp.circuit_shape.r1cs_shape, &pp.ck)?;

    // 3. output Πi+1 ← ((Ui+1, Wi+1), (ui+1, wi+1)).
    self.r_U = r_U;
    self.r_W = r_W;
    self.l_u = l_u;
    self.l_w = l_w;
    self.r_U_cyclefold = r_U_cyclefold;
    self.r_W_cyclefold = r_W_cyclefold;

    // Update statement being proven
    self.z_i = z_i
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;
    self.i += 1;
    self.prev_ic = ic;
    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  #[tracing::instrument(skip_all, name = "HyperNova::RecursiveSNARK::verify")]
  pub fn verify(
    &self,
    pp: &PublicParams<E>,
    num_steps: usize,
    z_0: &[E::Scalar],
    ic: IncrementalCommitment<E>,
  ) -> Result<Vec<E::Scalar>, NovaError> {
    // Basic checks for IVC proof
    // //////////////////////////
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;
    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;
    // check if the initial inputs match
    let is_inputs_not_match = self.z_0 != z_0;
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
    // //////////
    // 1. Compute H(pp, i, z_0, z_i, r_U)
    let mut ro = <Dual<E> as Engine>::RO::new(pp.ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp.digest());
    ro.absorb(E::Scalar::from(num_steps as u64));
    for e in z_0 {
      ro.absorb(*e);
    }
    for e in &self.z_i {
      ro.absorb(*e);
    }
    self.r_U.absorb_in_ro(&mut ro);
    let hash = ro.squeeze(NUM_HASH_BITS);
    // //////////////////////////////////
    // 2. Compute H(pp, i, r_U_cyclefold)
    let mut ro = <Dual<E> as Engine>::RO::new(pp.ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp.digest());
    ro.absorb(E::Scalar::from(num_steps as u64));
    self.r_U_cyclefold.absorb_in_ro(&mut ro);
    let hash_cyclefold = ro.squeeze(NUM_HASH_BITS);
    // ////////////////////////////////////////////
    // 3. Check if H(pp, i, z_0, z_i, r_U) = l_u.X[0] && H(pp, i, r_U_cyclefold) = l_u.X[1]
    if scalar_as_base::<Dual<E>>(hash) != self.l_u.aux.X[0]
      || scalar_as_base::<Dual<E>>(hash_cyclefold) != self.l_u.aux.X[1]
    {
      return Err(NovaError::ProofVerifyError);
    }

    // Verify the satisfiability of running relaxed instances, and the final primary instance.
    let (res_r_U, (res_l_u, res_r_U_cyclefold)) = rayon::join(
      || {
        pp.circuit_shape
          .r1cs_shape
          .is_sat_linearized(&pp.ck, &self.r_U, &self.r_W)
      },
      || {
        rayon::join(
          || {
            pp.circuit_shape
              .r1cs_shape
              .is_sat_split(&pp.ck, &self.l_u, &self.l_w)
          },
          || {
            pp.circuit_shape_cyclefold.r1cs_shape.is_sat_relaxed(
              &pp.ck_cyclefold,
              &self.r_U_cyclefold,
              &self.r_W_cyclefold,
            )
          },
        )
      },
    );
    res_r_U?;
    res_l_u?;
    res_r_U_cyclefold?;

    // Parse u_i.C_W as (C_ωi−1 , C_aux_i−1 ). Then check that C_i = hash(C_i−1 , C_ωi−1)
    self.ic_check(pp, ic)?;
    Ok(self.z_i.to_vec())
  }
}

impl<E> RecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  fn ic_check(&self, pp: &PublicParams<E>, ic: IncrementalCommitment<E>) -> Result<(), NovaError> {
    let expected_ic = increment_comm::<E>(&pp.ro_consts, self.prev_ic, self.l_u.pre_committed);
    if expected_ic != ic {
      return Err(NovaError::InvalidIC);
    }
    Ok(())
  }
}

/// Step circuit used for Hypernova
pub trait StepCircuit<F: PrimeField>: Send + Sync + Clone {
  /// Arity of the circuit. This is needed to build the public parameters
  fn arity(&self) -> usize;

  /// Synthesize the circuit
  fn synthesize<CS>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
  where
    CS: ConstraintSystem<F>;

  /// Get non-deterministic advice for the circuit
  fn advice(&self) -> (Vec<F>, Vec<F>) {
    (vec![], vec![])
  }
}

#[cfg(test)]
mod tests {
  use super::{IncrementalCommitment, RecursiveSNARK};
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    hypernova::{nebula::ic::increment_ic, rs::StepCircuit},
    provider::Bn256EngineIPA,
    traits::{snark::default_ck_hint, CurveCycleEquipped, Engine},
    NovaError,
  };
  use ff::PrimeField;
  use std::marker::PhantomData;

  type E = Bn256EngineIPA;
  type F = <E as Engine>::Scalar;

  #[test]
  fn test_rs() -> Result<(), NovaError> {
    let circuit = SquareCircuit::<F>::default();
    test_rs_with::<E>(&circuit)
  }

  #[test]
  fn test_pow_rs() -> Result<(), NovaError> {
    let circuit = PowCircuit::<F>::default();
    test_rs_with::<E>(&circuit)
  }

  fn test_rs_with<E: CurveCycleEquipped>(
    circuit: &impl StepCircuit<E::Scalar>,
  ) -> Result<(), NovaError> {
    run_circuit::<E>(circuit)
  }

  fn run_circuit<E: CurveCycleEquipped>(c: &impl StepCircuit<E::Scalar>) -> Result<(), NovaError> {
    let pp = super::PublicParams::<E>::setup(c, &*default_ck_hint(), &*default_ck_hint());
    let z_0 = vec![E::Scalar::from(2u64)];
    let mut ic = IncrementalCommitment::<E>::default();
    let mut recursive_snark = RecursiveSNARK::new(&pp, c, &z_0)?;
    for i in 0..10 {
      recursive_snark.prove_step(&pp, c, ic)?;
      let (advice_0, advice_1) = c.advice();
      ic = increment_ic::<E>(&pp.ck, &pp.ro_consts, ic, (&advice_0, &advice_1));
      recursive_snark.verify(&pp, i + 1, &z_0, ic)?;
    }
    Ok(())
  }

  #[derive(Clone, Default)]
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

  #[derive(Clone, Default)]
  pub struct PowCircuit<F>
  where
    F: PrimeField,
  {
    _field: PhantomData<F>,
  }

  impl<F> StepCircuit<F> for PowCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let mut x = z[0].clone();
      let mut y = x.clone();
      for i in 0..10_000 {
        y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
        x = y.clone();
      }
      Ok(vec![y])
    }
  }
}
