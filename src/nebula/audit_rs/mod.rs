//! Special [`RecursiveSNARK`] API to handle carrying the audit (scan) commitments for memory consitency checks
//!
//! Main change is it handles carrying two commitments, one for IS and the other for FS

use std::sync::Arc;

use super::{
  augmented_circuit::AugmentedCircuitParams,
  ic::IC,
  nifs::{PrimaryNIFS, PrimaryRelaxedNIFS, NIFS},
  traits::impl_rs_fields_trait,
};
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_FE_IN_EMULATED_POINT, NUM_HASH_BITS},
  cyclefold::{
    circuit::CycleFoldCircuit,
    util::{absorb_primary_relaxed_r1cs, FoldingData},
  },
  errors::NovaError,
  frontend::{
    num::AllocatedNum,
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
    ConstraintSystem, SynthesisError,
  },
  gadgets::scalar_as_base,
  nebula::traits::RecursiveSNARKFieldsTrait,
  r1cs::{CommitmentKeyHint, R1CSInstance, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    commitment::CommitmentEngineTrait, AbsorbInROTrait, CurveCycleEquipped, Dual, Engine,
    ROConstantsCircuit, ROTrait,
  },
  Commitment, CommitmentKey, DigestComputer, R1CSWithArity, ROConstants, SimpleDigestible,
};
use augmented_circuit::{AugmentedCircuit, AugmentedCircuitInputs};
use ff::{Field, PrimeField};
use once_cell::sync::OnceCell;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

mod augmented_circuit;

/// The public parameters used in the CycleFold recursive SNARK proof and verification
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AuditPublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  /// Arity of step circuit
  pub F_arity_primary: usize,
  /// RO constants for primary circuit
  pub ro_consts: ROConstants<Dual<E1>>,
  /// RO constants for primary circuit
  pub ro_consts_circuit: ROConstantsCircuit<Dual<E1>>,
  /// Commitment key for primary circuit
  pub ck_primary: Arc<CommitmentKey<E1>>,
  /// R1CS shape we are arguing about
  pub circuit_shape_primary: R1CSWithArity<E1>,
  /// Parameters of big nats in circuit
  pub augmented_circuit_params: AugmentedCircuitParams,
  /// secondary commitment key
  pub ck_cyclefold: Arc<CommitmentKey<Dual<E1>>>,
  /// R1CS shape of cyclefold circuit
  pub circuit_shape_cyclefold: R1CSWithArity<Dual<E1>>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E1::Scalar>,
}

impl<E1> AuditPublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  /// Builds the public parameters for the circuit `C1`.
  /// The same note for public parameter hints apply as in the case for Nova's public parameters:
  /// For some final compressing SNARKs the size of the commitment key must be larger, so we include
  /// `ck_hint_primary` and `ck_hint_cyclefold` parameters to accommodate this.
  #[tracing::instrument(skip_all, name = "nebula::PublicParams::setup")]
  pub fn setup<C1: AuditStepCircuit<E1::Scalar>>(
    c_primary: &C1,
    ck_hint_primary: &CommitmentKeyHint<E1>,
    ck_hint_cyclefold: &CommitmentKeyHint<Dual<E1>>,
  ) -> Self {
    // This value is used to validate inputs to API
    let F_arity_primary = c_primary.arity();

    // Get the round constants used in the poseidon hash function and poseidon hash function circuit
    let ro_consts = ROConstants::<Dual<E1>>::default();
    let ro_consts_circuit = ROConstantsCircuit::<Dual<E1>>::default();

    // Get the structure for the AugmentedCircuit and corresponding commitment key
    let augmented_circuit_params = AugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS);
    let circuit_primary: AugmentedCircuit<'_, E1, C1> = AugmentedCircuit::new(
      &augmented_circuit_params,
      ro_consts_circuit.clone(),
      None,
      c_primary,
    );
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    let (r1cs_shape_primary, ck_primary) = cs.r1cs_shape_and_key(ck_hint_primary);
    let ck_primary = Arc::new(ck_primary);
    let circuit_shape_primary = R1CSWithArity::new(r1cs_shape_primary, F_arity_primary);

    // Get the structure for the CycleFold circuit and corresponding commitment key
    let mut cs: ShapeCS<Dual<E1>> = ShapeCS::new();
    let circuit_cyclefold: CycleFoldCircuit<E1> = CycleFoldCircuit::default();
    let _ = circuit_cyclefold.synthesize(&mut cs);
    let (r1cs_shape_cyclefold, ck_cyclefold) = cs.r1cs_shape_and_key(ck_hint_cyclefold);
    let ck_cyclefold = Arc::new(ck_cyclefold);
    let circuit_shape_cyclefold = R1CSWithArity::new(r1cs_shape_cyclefold, 0);

    Self {
      F_arity_primary,
      ro_consts,
      ro_consts_circuit,
      ck_primary,
      circuit_shape_primary,
      augmented_circuit_params,
      ck_cyclefold,
      circuit_shape_cyclefold,
      digest: OnceCell::new(),
    }
  }

  /// Calculate the digest of the public parameters.
  pub fn digest(&self) -> E1::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure in retrieving digest")
  }

  /// Return reference to commitment key
  pub fn ck(&self) -> &Arc<CommitmentKey<E1>> {
    &self.ck_primary
  }

  /// Returns the number of constraints in the primary and cyclefold circuits
  pub const fn num_constraints(&self) -> (usize, usize) {
    (
      self.circuit_shape_primary.r1cs_shape.num_cons,
      self.circuit_shape_cyclefold.r1cs_shape.num_cons,
    )
  }

  /// Returns the number of variables in the primary and cyclefold circuits
  pub const fn num_variables(&self) -> (usize, usize) {
    (
      self.circuit_shape_primary.r1cs_shape.num_vars,
      self.circuit_shape_cyclefold.r1cs_shape.num_vars,
    )
  }

  /// Break up into shape, ck and digest for layer 2
  pub fn into_shape_ck_digest(self) -> (R1CSWithArity<E1>, Arc<CommitmentKey<E1>>, E1::Scalar) {
    let digest = self.digest();
    (self.circuit_shape_primary, self.ck_primary, digest)
  }
}

impl<E1> SimpleDigestible for AuditPublicParams<E1> where E1: CurveCycleEquipped {}

/// A SNARK that proves the correct execution of an incremental computation in the CycleFold folding
/// scheme.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AuditRecursiveSNARK<E1>
where
  E1: CurveCycleEquipped,
{
  // Input
  z0: Vec<E1::Scalar>,

  // primary circuit data
  r_W_primary: RelaxedR1CSWitness<E1>,
  pub(crate) r_U_primary: RelaxedR1CSInstance<E1>,
  l_w_primary: R1CSWitness<E1>,
  pub(crate) l_u_primary: R1CSInstance<E1>,

  // Number of recursive steps proven
  i: usize,

  // incremental commitment to memory advice
  pub(in crate::nebula) prev_IC: (E1::Scalar, E1::Scalar),

  // commitment to non-deterministic advice
  pub(in crate::nebula) comm_omega_prev: (Commitment<E1>, Commitment<E1>), // supposed to be contained in self.l_u_primary // corresponds to comm_W in self.l_u_primary

  // cyclefold circuit data
  r_W_cyclefold: RelaxedR1CSWitness<Dual<E1>>,
  pub(in crate::nebula) r_U_cyclefold: RelaxedR1CSInstance<Dual<E1>>,

  // outputs
  pub(in crate::nebula) zi: Vec<E1::Scalar>,

  // makes Nova simulataable
  r_i: E1::Scalar,
}

impl<E1> AuditRecursiveSNARK<E1>
where
  E1: CurveCycleEquipped,
{
  /// Create a new instance of AuditRecursiveSNARK
  #[tracing::instrument(skip_all, name = "nebula::AuditRecursiveSNARK::new")]
  pub fn new<C>(
    pp: &AuditPublicParams<E1>,
    step_circuit: &C,
    z0: &[E1::Scalar],
  ) -> Result<Self, NovaError>
  where
    C: AuditStepCircuit<E1::Scalar>,
  {
    if z0.len() != pp.F_arity_primary {
      return Err(NovaError::InvalidInitialInputLength);
    }

    // Get default running primary instance and witness pair
    let r1cs_primary = &pp.circuit_shape_primary.r1cs_shape;
    let r_U_primary = RelaxedR1CSInstance::default(&*pp.ck_primary, r1cs_primary);
    let r_W_primary = RelaxedR1CSWitness::default(r1cs_primary);

    // Base case for F'
    //
    // Get the new instance-witness pair to be folded into running instance
    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let r_i = E1::Scalar::random(&mut OsRng);
    let inputs_primary: AugmentedCircuitInputs<E1> = AugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      <Dual<E1> as Engine>::Base::from(0u64),
      z0.to_vec(),
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      r_i,
    );
    let circuit_primary = AugmentedCircuit::new(
      &pp.augmented_circuit_params,
      pp.ro_consts_circuit.clone(),
      Some(inputs_primary),
      step_circuit,
    );
    let zi = circuit_primary.synthesize(&mut cs_primary)?;
    let (l_u_primary, l_w_primary) =
      cs_primary.r1cs_instance_and_witness(r1cs_primary, &pp.ck_primary)?;

    // Get z_i values out of the Constraint System
    let zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;

    // Get the running CycleFold instance and witness pair
    let r1cs_cyclefold = &pp.circuit_shape_cyclefold.r1cs_shape;
    let r_U_cyclefold = RelaxedR1CSInstance::default(&*pp.ck_cyclefold, r1cs_cyclefold);
    let r_W_cyclefold = RelaxedR1CSWitness::default(r1cs_cyclefold);

    Ok(Self {
      z0: z0.to_vec(),
      // IVC proof
      r_W_primary,
      r_U_primary,
      l_w_primary,
      l_u_primary,

      // data for statement being proven
      i: 0,
      zi,

      // incremental commitment
      prev_IC: (E1::Scalar::ZERO, E1::Scalar::ZERO),

      // commitment to non-deterministic advice
      comm_omega_prev: step_circuit.commit_w::<E1>(&pp.ck_primary), // C_ω_i−1

      // running Cyclefold instance, witness pair
      r_U_cyclefold,
      r_W_cyclefold,

      // makes Nova simulatable
      r_i,
    })
  }

  /// Create a new [`AuditRecursiveSNARK`] (or updates the provided [`AuditRecursiveSNARK`])
  /// by executing a step of the incremental computation
  #[tracing::instrument(skip_all, name = "nebula::AuditRecursiveSNARK::prove_step")]
  pub fn prove_step<C>(
    &mut self,
    pp: &AuditPublicParams<E1>,
    step_circuit: &C,
    IC_i: (E1::Scalar, E1::Scalar),
  ) -> Result<(), NovaError>
  where
    C: AuditStepCircuit<E1::Scalar>,
  {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // Abort if Ci  != hash(Ci−1, Cωi−1 )
    let intermediary_comm_IS =
      IC::<E1>::increment_comm_w(&pp.ro_consts, self.prev_IC.0, self.comm_omega_prev.0);
    let intermediary_comm_FS =
      IC::<E1>::increment_comm_w(&pp.ro_consts, self.prev_IC.1, self.comm_omega_prev.1);
    if IC_i.0 != intermediary_comm_IS || IC_i.1 != intermediary_comm_FS {
      return Err(NovaError::InvalidIC);
    }

    // Parse Πi (self) as ((Ui, Wi), (ui, wi)) and then:
    //
    // 1. compute (Ui+1,Wi+1,T) ← NIFS.P(pk,(Ui,Wi),(ui,wi)),
    let (nifs, (r_U_primary, r_W_primary), (r_U_cyclefold, r_W_cyclefold), _, U_secondary_temp) =
      NIFS::<E1>::prove(
        (&pp.ck_primary, &pp.ck_cyclefold),
        &pp.ro_consts,
        &pp.digest(),
        (
          &pp.circuit_shape_primary.r1cs_shape,
          &pp.circuit_shape_cyclefold.r1cs_shape,
        ),
        (&self.r_U_primary, &self.r_W_primary),
        (&self.l_u_primary, &self.l_w_primary),
        (&self.r_U_cyclefold, &self.r_W_cyclefold),
      )?;

    // Get advice to pass into verifier circuit
    let E_new = r_U_primary.comm_E;
    let W_new = r_U_primary.comm_W;
    let data_p = FoldingData::new(
      self.r_U_primary.clone(),
      self.l_u_primary.clone(),
      nifs.nifs_primary.comm_T,
    );
    let data_c_E = FoldingData::new(
      self.r_U_cyclefold.clone(),
      nifs.l_u_cyclefold_E,
      nifs.comm_T1,
    );
    let data_c_W = FoldingData::new(U_secondary_temp, nifs.l_u_cyclefold_W, nifs.comm_T2);

    // 2. compute (ui+1, wi+1) ← trace(F ′, (vk, Ui, ui, (i, z0, zi), ωi, T )),
    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let r_next = E1::Scalar::random(&mut OsRng);
    let inputs_primary: AugmentedCircuitInputs<E1> = AugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      <Dual<E1> as Engine>::Base::from(self.i as u64),
      self.z0.clone(),
      Some(self.zi.clone()),
      Some(data_p),
      Some(data_c_E),
      Some(data_c_W),
      Some(E_new),
      Some(W_new),
      Some(self.prev_IC),
      Some(self.comm_omega_prev),
      Some(self.r_i),
      r_next,
    );
    let circuit_primary: AugmentedCircuit<'_, E1, C> = AugmentedCircuit::new(
      &pp.augmented_circuit_params,
      pp.ro_consts_circuit.clone(),
      Some(inputs_primary),
      step_circuit,
    );
    let zi = circuit_primary.synthesize(&mut cs_primary)?;
    let (l_u_primary, l_w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.circuit_shape_primary.r1cs_shape, &pp.ck_primary)
      .map_err(|_| NovaError::UnSat)?;

    // Get z_i values out of the Constraint System
    self.zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<_>, _>>()?;

    // 3. output Πi+1 ← ((Ui+1, Wi+1), (ui+1, wi+1)).
    self.r_U_primary = r_U_primary;
    self.r_W_primary = r_W_primary;
    self.l_u_primary = l_u_primary;
    self.l_w_primary = l_w_primary;

    // Update running CycleFold instance and witness pair
    self.r_U_cyclefold = r_U_cyclefold;
    self.r_W_cyclefold = r_W_cyclefold;

    // update incremental commitments in IVC proof
    self.prev_IC = IC_i;
    self.comm_omega_prev = step_circuit.commit_w::<E1>(&pp.ck_primary);

    // Update number of steps proven
    self.i += 1;
    self.r_i = r_next;
    Ok(())
  }

  /// Verify the correctness of the `AuditRecursiveSNARK`
  #[tracing::instrument(skip_all, name = "nebula::AuditRecursiveSNARK::verify")]
  pub fn verify(
    &self,
    pp: &AuditPublicParams<E1>,
    num_steps: usize,
    z0: &[E1::Scalar],
    IC_i: (E1::Scalar, E1::Scalar),
  ) -> Result<Vec<E1::Scalar>, NovaError> {
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;

    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;

    // check if the initial inputs match
    let is_inputs_not_match = self.z0 != z0;

    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outputs = self.r_U_primary.X.len() != 2;

    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outputs
    {
      return Err(NovaError::ProofVerifyError);
    }

    // Calculate the hashes of the primary running instance and cyclefold running instance
    let (hash_primary, hash_cyclefold) = {
      let mut hasher_p = <Dual<E1> as Engine>::RO::new(
        pp.ro_consts.clone(),
        5 + 2 * pp.F_arity_primary + 2 * NUM_FE_IN_EMULATED_POINT + 3,
      );
      hasher_p.absorb(pp.digest());
      hasher_p.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0 {
        hasher_p.absorb(*e);
      }
      for e in &self.zi {
        hasher_p.absorb(*e);
      }
      absorb_primary_relaxed_r1cs::<E1, Dual<E1>>(&self.r_U_primary, &mut hasher_p);
      hasher_p.absorb(self.prev_IC.0);
      hasher_p.absorb(self.prev_IC.1);
      hasher_p.absorb(self.r_i);
      let hash_primary = hasher_p.squeeze(NUM_HASH_BITS);

      let mut hasher_c = <Dual<E1> as Engine>::RO::new(
        pp.ro_consts.clone(),
        1 + 1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS,
      );
      hasher_c.absorb(pp.digest());
      hasher_c.absorb(E1::Scalar::from(num_steps as u64));
      self.r_U_cyclefold.absorb_in_ro(&mut hasher_c);
      hasher_c.absorb(self.r_i);
      let hash_cyclefold = hasher_c.squeeze(NUM_HASH_BITS);
      (hash_primary, hash_cyclefold)
    };

    // Verify the hashes equal the public IO for the final primary instance
    if scalar_as_base::<Dual<E1>>(hash_primary) != self.l_u_primary.X[0]
      || scalar_as_base::<Dual<E1>>(hash_cyclefold) != self.l_u_primary.X[1]
    {
      return Err(NovaError::ProofVerifyError);
    }

    // Verify the satisfiability of running relaxed instances, and the final primary instance.
    let (res_r_primary, (res_l_primary, res_r_cyclefold)) = rayon::join(
      || {
        pp.circuit_shape_primary.r1cs_shape.is_sat_relaxed(
          &pp.ck_primary,
          &self.r_U_primary,
          &self.r_W_primary,
        )
      },
      || {
        rayon::join(
          || {
            pp.circuit_shape_primary.r1cs_shape.is_sat(
              &pp.ck_primary,
              &self.l_u_primary,
              &self.l_w_primary,
            )
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
    res_r_primary?;
    res_l_primary?;
    res_r_cyclefold?;

    // Abort if C_i  != hash(C_i−1, C_ω_i−1)
    let intermediary_comm_IS =
      IC::<E1>::increment_comm_w(&pp.ro_consts, self.prev_IC.0, self.comm_omega_prev.0);
    let intermediary_comm_FS =
      IC::<E1>::increment_comm_w(&pp.ro_consts, self.prev_IC.1, self.comm_omega_prev.1);
    if IC_i.0 != intermediary_comm_IS || IC_i.1 != intermediary_comm_FS {
      return Err(NovaError::InvalidIC);
    }

    Ok(self.zi.to_vec())
  }

  /// Increment the incremental commitment with the new non-deterministic witness from the circuit
  #[tracing::instrument(
    skip_all,
    name = "nebula::RecursiveSNARK::increment_commitment",
    level = "debug"
  )]
  pub fn increment_commitment<C>(
    &self,
    pp: &AuditPublicParams<E1>,
    step_circuit: &C,
  ) -> (E1::Scalar, E1::Scalar)
  where
    C: AuditStepCircuit<E1::Scalar>,
  {
    (
      IC::<E1>::commit(
        &pp.ck_primary,
        &pp.ro_consts,
        self.prev_IC.0,
        step_circuit.IS_advice(),
      ),
      IC::<E1>::commit(
        &pp.ck_primary,
        &pp.ro_consts,
        self.prev_IC.1,
        step_circuit.FS_advice(),
      ),
    )
  }

  /// The number of steps which have been executed thus far.
  pub fn num_steps(&self) -> usize {
    self.i
  }

  /// Get relaxed instance witness pair for primary circuit
  pub fn U_W(&self) -> (&RelaxedR1CSInstance<E1>, &RelaxedR1CSWitness<E1>) {
    (&self.r_U_primary, &self.r_W_primary)
  }

  /// Get the secondary curve part of the running instance
  pub fn secondary_rs_part(
    &self,
  ) -> (
    &RelaxedR1CSInstance<Dual<E1>>,
    &RelaxedR1CSWitness<Dual<E1>>,
  ) {
    (&self.r_U_cyclefold, &self.r_W_cyclefold)
  }

  /// Get the secondary curve part of the running instance
  pub fn secondary_rs_part_derandomized(
    &self,
    pp: &AuditPublicParams<E1>,
  ) -> (
    RelaxedR1CSInstance<Dual<E1>>,
    RelaxedR1CSWitness<Dual<E1>>,
    <Dual<E1> as Engine>::Scalar,
    <Dual<E1> as Engine>::Scalar,
  ) {
    let (derandom_W, wit_blind, err_blind) = self.r_W_cyclefold.derandomize();
    let derandom_U = self.r_U_cyclefold.derandomize(
      &<Dual<E1> as Engine>::CE::derand_key(&pp.ck_cyclefold),
      &wit_blind,
      &err_blind,
    );
    (derandom_U, derandom_W, wit_blind, err_blind)
  }

  /// Get primary & secondayr relaxed instance witness pair
  pub fn primary_secondary_U_W(
    &self,
  ) -> (
    &RelaxedR1CSInstance<E1>,
    &RelaxedR1CSWitness<E1>,
    &RelaxedR1CSInstance<Dual<E1>>,
    &RelaxedR1CSWitness<Dual<E1>>,
  ) {
    (
      &self.r_U_primary,
      &self.r_W_primary,
      &self.r_U_cyclefold,
      &self.r_W_cyclefold,
    )
  }

  /// Do NIFS.P on the IVC proof before we send it of for compression
  pub(crate) fn fold_ivc_compression_step(
    &self,
    pp: &AuditPublicParams<E1>,
  ) -> Result<
    (
      RelaxedR1CSInstance<E1>,
      RelaxedR1CSWitness<E1>,
      PrimaryNIFS<E1>,
      PrimaryRelaxedNIFS<E1>,
      E1::Scalar,
      E1::Scalar,
      RelaxedR1CSInstance<E1>,
    ),
    NovaError,
  > {
    let (nifs, (U_f, W_f), _) = PrimaryNIFS::prove(
      &*pp.ck_primary,
      &pp.ro_consts,
      &pp.digest(),
      &pp.circuit_shape_primary.r1cs_shape,
      (&self.r_U_primary, &self.r_W_primary),
      (&self.l_u_primary, &self.l_w_primary),
    )?;

    // Fold random instance and witness
    let (random_U, random_W) = pp
      .circuit_shape_primary
      .r1cs_shape
      .sample_random_instance_witness(&pp.ck_primary)?;
    let (nifs_r, (U, W), _) = PrimaryRelaxedNIFS::prove(
      &*pp.ck_primary,
      &pp.ro_consts,
      &pp.digest(),
      &pp.circuit_shape_primary.r1cs_shape,
      (&U_f, &W_f),
      (&random_U, &random_W),
    )?;

    let (derandom_W, wit_blind, err_blind) = W.derandomize();
    let derandom_U = U.derandomize(&E1::CE::derand_key(&pp.ck_primary), &wit_blind, &err_blind);
    Ok((
      derandom_U, derandom_W, nifs, nifs_r, wit_blind, err_blind, random_U,
    ))
  }
}

impl_rs_fields_trait!(AuditRecursiveSNARK);

/// A helper trait for a step of the incremental computation (i.e., circuit for F)
pub trait AuditStepCircuit<F: PrimeField>: Send + Sync + Clone {
  /// Return the number of inputs or outputs of each step
  /// (this method is called only at circuit synthesis time)
  /// `synthesize` and `output` methods are expected to take as
  /// input a vector of size equal to arity and output a vector of size equal to arity
  fn arity(&self) -> usize;

  /// Sythesize the circuit for a computation step and return variable
  /// that corresponds to the output of the step `z_{i+1}`
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>;

  /// Get the non-deterministic advice we will commit to
  fn IS_advice(&self) -> Vec<F>;

  /// Get the non-deterministic advice we will commit to
  fn FS_advice(&self) -> Vec<F>;

  /// Produce a commitment to the non_deterministic advice
  fn commit_w<E>(&self, ck: &CommitmentKey<E>) -> (Commitment<E>, Commitment<E>)
  where
    E: Engine<Scalar = F>,
  {
    (
      E::CE::commit(ck, &self.IS_advice(), &E::Scalar::ZERO),
      E::CE::commit(ck, &self.FS_advice(), &E::Scalar::ZERO),
    )
  }
}

#[cfg(test)]
mod test {
  use super::{AuditPublicParams, AuditRecursiveSNARK, AuditStepCircuit};
  use crate::{
    errors::NovaError,
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    provider::Bn256EngineIPA,
    traits::{snark::default_ck_hint, CurveCycleEquipped},
  };
  use ff::{Field, PrimeField};
  use std::marker::PhantomData;

  #[derive(Clone)]
  struct SquareCircuit<F> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> AuditStepCircuit<F> for SquareCircuit<F> {
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

    fn IS_advice(&self) -> Vec<F> {
      [F::from(1), F::from(2), F::from(3)].to_vec()
    }

    fn FS_advice(&self) -> Vec<F> {
      [F::from(4), F::from(5), F::from(6)].to_vec()
    }
  }

  fn test_trivial_cyclefold_prove_verify_with<E: CurveCycleEquipped>() -> Result<(), NovaError> {
    let primary_circuit = SquareCircuit::<E::Scalar> { _p: PhantomData };

    let pp =
      AuditPublicParams::<E>::setup(&primary_circuit, &*default_ck_hint(), &*default_ck_hint());

    let z0 = vec![E::Scalar::from(2u64)];

    let mut recursive_snark = AuditRecursiveSNARK::new(&pp, &primary_circuit, &z0).unwrap();
    let mut IC_i = (E::Scalar::ZERO, E::Scalar::ZERO);

    for i in 0..10 {
      recursive_snark.prove_step(&pp, &primary_circuit, IC_i)?;

      // TODO: figure out if i should put this in the rs API?
      IC_i = recursive_snark.increment_commitment(&pp, &primary_circuit);

      recursive_snark.verify(&pp, i + 1, &z0, IC_i).unwrap();
    }

    Ok(())
  }

  #[test]
  fn test_cyclefold_prove_verify() -> Result<(), NovaError> {
    test_trivial_cyclefold_prove_verify_with::<Bn256EngineIPA>()
  }
}
