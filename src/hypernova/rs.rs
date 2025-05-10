//! IVC scheme with Hypernova
//!
//! This module implements a SNARK that proves the correct execution of an incremental computation.

use super::{nebula::ic::increment_comm, pp::PublicParamsTrait};
use crate::{
  constants::{DEFAULT_ABSORBS, NUM_HASH_BITS},
  errors::NovaError,
  frontend::{
    num::AllocatedNum, r1cs::NovaWitness, solver::SatisfyingAssignment, test_cs::TestConstraintSystem, ConstraintSystem, SynthesisError
  },
  gadgets::scalar_as_base,
  hypernova::{
    augmented_circuit::{AugmentedCircuit, AugmentedCircuitInputs},
    nifs::NIFS,
  },
  r1cs::{
    split::{LR1CSInstance, SplitR1CSInstance, SplitR1CSWitness},
    RelaxedR1CSInstance, RelaxedR1CSWitness,
  },
  traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROTrait},
  ROConstants,
};
use ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};

/// A type that represents the carried commitments for this commitment-carrying HyperNova IVC scheme.
pub type IncrementalCommitment<E> = (<E as Engine>::Scalar, <E as Engine>::Scalar);

/// A SNARK that proves the correct execution of an incremental computation. HyperNova IVC scheme (with CycleFold).
///
/// * (U, W, u, w) -> IVC Proof
/// * (i, z_0, z_i) -> Statement being proven
///
/// # Note
///
/// * Carries two commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  pub(crate) r_U: LR1CSInstance<E>,
  pub(crate) r_W: SplitR1CSWitness<E>,
  pub(crate) l_u: SplitR1CSInstance<E>,
  pub(crate) l_w: SplitR1CSWitness<E>,
  pub(crate) r_U_cyclefold: RelaxedR1CSInstance<Dual<E>>,
  pub(crate) r_W_cyclefold: RelaxedR1CSWitness<Dual<E>>,
  pub(crate) z_0: Vec<E::Scalar>,
  pub(crate) i: usize,
  pub(crate) z_i: Vec<E::Scalar>,
  pub(crate) prev_ic: IncrementalCommitment<E>,
}

impl<E> RecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  /// Create a new instance of [`RecursiveSNARK`]
  #[tracing::instrument(skip_all, name = "HyperNova::RecursiveSNARK::new")]
  pub fn new<C>(
    pp: &impl PublicParamsTrait<E>,
    step_circuit: &C,
    z_0: &[E::Scalar],
  ) -> Result<Self, NovaError>
  where
    C: StepCircuit<E::Scalar>,
  {
    if z_0.len() != pp.F_arity() {
      return Err(NovaError::InvalidInitialInputLength);
    }

    // --- Get running instance, witness pairs ---
    //
    // 1. Get default running primary instance and witness pair
    let r1cs = &pp.circuit_shape().r1cs_shape;
    let r_U = LR1CSInstance::default(r1cs);
    let r_W = SplitR1CSWitness::default(r1cs);
    // 2. Get the running CycleFold instance and witness pair
    let r1cs_cyclefold = &pp.circuit_shape_cyclefold().r1cs_shape;
    let r_U_cyclefold = RelaxedR1CSInstance::default(&**pp.ck_cyclefold(), r1cs_cyclefold);
    let r_W_cyclefold = RelaxedR1CSWitness::default(r1cs_cyclefold);

    // --- Base case for F' ---
    //
    // Get the new instance-witness pair to be folded into running instance
    let ((l_u, l_w), z_i) = Self::synthesize_aug_circuit_base_case(pp, z_0, step_circuit)?;
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
    pp: &impl PublicParamsTrait<E>,
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

    // 1. Parse u_i.C_W as (C_ωi−1 , C_aux_i−1).
    // 2. Abort if C_i != hash(C_i−1, C_ωi−1)
    self.ic_check(pp, ic)?;

    // 1. Parse Π_i (self) as ((U_i, W_i), (u_i, w_i)).
    // 2. compute (U_i+1,W_i+1) ← NIFS.P(pk, (U_i, W_i), (u_i, w_i)),
    let (nifs, (r_U, r_W), (r_U_cyclefold, r_W_cyclefold)) = self.nifs(pp)?;

    // Compute (u_i+1, w_i+1) ← trace(F', (vk, Ui, ui, (i, z_0, z_i), ωi)),
    let ((l_u, l_w), z_i) = self.synthesize_aug_circuit(pp, nifs, &r_U, step_circuit)?;

    // Update statement being proven
    self.z_i = z_i;
    self.i += 1;
    self.prev_ic = ic;

    // Output Πi+1 ← ((Ui+1, Wi+1), (ui+1, wi+1)).
    self.r_U = r_U;
    self.r_W = r_W;
    self.l_u = l_u;
    self.l_w = l_w;
    self.r_U_cyclefold = r_U_cyclefold;
    self.r_W_cyclefold = r_W_cyclefold;
    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  #[tracing::instrument(skip_all, name = "HyperNova::RecursiveSNARK::verify")]
  pub fn verify(
    &self,
    pp: &impl PublicParamsTrait<E>,
    num_steps: usize,
    z_0: &[E::Scalar],
    ic: IncrementalCommitment<E>,
  ) -> Result<Vec<E::Scalar>, NovaError> {
    // --- Basic checks for IVC proof ---
    //
    // 1. number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;
    // 2. check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;
    // 3. check if the initial inputs match
    let is_inputs_not_match = self.z_0 != z_0;
    // 4. check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outputs = self.r_U.X.len() != 2;
    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outputs
    {
      return Err(NovaError::ProofVerifyError);
    }

    // --- Hash check ---
    //
    // 1. Compute H(pp, i, z_0, z_i, r_U, prev_ic)
    // 2. Compute H(pp, i, r_U_cyclefold)
    Self::hash_check(
      pp.ro_consts(),
      pp.digest(),
      num_steps,
      z_0,
      &self.z_i,
      &self.r_U,
      self.prev_ic,
      &self.l_u,
      &self.r_U_cyclefold,
    )?;

    // Verify the satisfiability of running relaxed instances, and the final primary instance.
    let (res_r_U, (res_l_u, res_r_U_cyclefold)) = rayon::join(
      || {
        pp.circuit_shape()
          .r1cs_shape
          .is_sat_linearized(pp.ck(), &self.r_U, &self.r_W)
      },
      || {
        rayon::join(
          || {
            pp.circuit_shape()
              .r1cs_shape
              .is_sat_split(pp.ck(), &self.l_u, &self.l_w)
          },
          || {
            pp.circuit_shape_cyclefold().r1cs_shape.is_sat_relaxed(
              pp.ck_cyclefold(),
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

    // 1. Parse u_i.C_W as (C_ωi−1 , C_aux_i−1 ).
    // 2. Then check that C_i = hash(C_i−1 , C_ωi−1)
    self.ic_check(pp, ic)?;
    Ok(self.z_i.to_vec())
  }
}

impl<E> RecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  fn ic_check(
    &self,
    pp: &impl PublicParamsTrait<E>,
    ic: IncrementalCommitment<E>,
  ) -> Result<(), NovaError> {
    let expected_ic = increment_comm::<E>(pp.ro_consts(), self.prev_ic, self.l_u.pre_committed);
    if expected_ic != ic {
      return Err(NovaError::InvalidIC);
    }
    Ok(())
  }

  fn nifs(
    &self,
    pp: &impl PublicParamsTrait<E>,
  ) -> Result<
    (
      NIFS<E>,
      (LR1CSInstance<E>, SplitR1CSWitness<E>),
      (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>),
    ),
    NovaError,
  > {
    NIFS::prove(
      (
        &pp.circuit_shape().r1cs_shape,
        &pp.circuit_shape_cyclefold().r1cs_shape,
      ),
      pp.ck_cyclefold(),
      pp.ro_consts(),
      &pp.digest(),
      (&self.r_U, &self.r_W),
      (&self.l_u, &self.l_w),
      (&self.r_U_cyclefold, &self.r_W_cyclefold),
    )
  }

  fn synthesize_aug_circuit_with_inputs(
    pp: &impl PublicParamsTrait<E>,
    inputs: AugmentedCircuitInputs<E>,
    step_circuit: &impl StepCircuit<E::Scalar>,
  ) -> Result<((SplitR1CSInstance<E>, SplitR1CSWitness<E>), Vec<E::Scalar>), NovaError> {
    let mut cs = SatisfyingAssignment::<E>::new();
    let circuit = AugmentedCircuit::new(
      pp.augmented_circuit_params(),
      pp.ro_consts_circuit().clone(),
      Some(inputs),
      step_circuit,
      pp.num_rounds(),
    );
    let z_i = circuit.synthesize(&mut cs)?;
    let z_i = z_i
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;
    let (u, w) = cs.split_r1cs_instance_and_witness(&pp.circuit_shape().r1cs_shape, pp.ck())?;
    Ok(((u, w), z_i))
  }

  fn synthesize_aug_circuit(
    &self,
    pp: &impl PublicParamsTrait<E>,
    nifs: NIFS<E>,
    r_U: &LR1CSInstance<E>,
    step_circuit: &impl StepCircuit<E::Scalar>,
  ) -> Result<((SplitR1CSInstance<E>, SplitR1CSWitness<E>), Vec<E::Scalar>), NovaError> {
    let inputs: AugmentedCircuitInputs<E> = AugmentedCircuitInputs::new(
      pp.digest(),
      E::Scalar::from(self.i as u64),
      self.z_0.to_vec(),
      Some(self.z_i.clone()),
      Some(nifs),
      Some(self.r_U.clone()),
      Some(self.l_u.clone()),
      Some(r_U.comm_W),
      Some(self.r_U_cyclefold.clone()),
      Some(r_U.pre_committed),
      Some(self.prev_ic),
    );
    Self::synthesize_aug_circuit_with_inputs(pp, inputs, step_circuit)
  }

  fn synthesize_aug_circuit_base_case(
    pp: &impl PublicParamsTrait<E>,
    z_0: &[E::Scalar],
    step_circuit: &impl StepCircuit<E::Scalar>,
  ) -> Result<((SplitR1CSInstance<E>, SplitR1CSWitness<E>), Vec<E::Scalar>), NovaError> {
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
    Self::synthesize_aug_circuit_with_inputs(pp, inputs, step_circuit)
  }

  pub(crate) fn hash_check(
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: E::Scalar,
    num_steps: usize,
    z_0: &[E::Scalar],
    z_i: &[E::Scalar],
    r_U: &LR1CSInstance<E>,
    prev_ic: IncrementalCommitment<E>,
    l_u: &SplitR1CSInstance<E>,
    r_U_cyclefold: &RelaxedR1CSInstance<Dual<E>>,
  ) -> Result<(), NovaError> {
    // --- Hash check ---
    //
    // 1. Compute H(pp, i, z_0, z_i, r_U, prev_ic)
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp_digest);
    ro.absorb(E::Scalar::from(num_steps as u64));
    for e in z_0 {
      ro.absorb(*e);
    }
    for e in z_i {
      ro.absorb(*e);
    }
    r_U.absorb_in_ro(&mut ro);
    ro.absorb(prev_ic.0);
    ro.absorb(prev_ic.1);
    let hash = ro.squeeze(NUM_HASH_BITS);
    // 2. Compute H(pp, i, r_U_cyclefold)
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp_digest);
    ro.absorb(E::Scalar::from(num_steps as u64));
    r_U_cyclefold.absorb_in_ro(&mut ro);
    let hash_cyclefold = ro.squeeze(NUM_HASH_BITS);
    // 3. Check if H(pp, i, z_0, z_i, r_U) = l_u.X[0] && H(pp, i, r_U_cyclefold) = l_u.X[1]
    if scalar_as_base::<Dual<E>>(hash) != l_u.aux.X[0]
      || scalar_as_base::<Dual<E>>(hash_cyclefold) != l_u.aux.X[1]
    {
      return Err(NovaError::ProofVerifyError);
    }
    Ok(())
  }

  /// Get the number of steps executed by the recursiveSNARK
  pub fn num_steps(&self) -> usize {
    self.i
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
mod tests {
  use super::{IncrementalCommitment, RecursiveSNARK};
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    hypernova::{nebula::ic::increment_ic, pp::PublicParams, rs::StepCircuit},
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
    let pp = PublicParams::<E>::setup(c, &*default_ck_hint(), &*default_ck_hint());
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
