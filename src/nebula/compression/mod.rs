//! Implements components to enable the compression-step for IVC proofs

use super::{
  ic::IC,
  nifs::{CycleFoldRelaxedNIFS, PrimaryNIFS, PrimaryRelaxedNIFS},
  traits::{Layer1PPTrait, Layer1RSTrait},
};
use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_FE_IN_EMULATED_POINT, NUM_HASH_BITS},
  cyclefold::util::absorb_primary_relaxed_r1cs,
  errors::NovaError,
  gadgets::scalar_as_base,
  nebula::traits::RecursiveSNARKFieldsTrait,
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{
    commitment::CommitmentEngineTrait,
    snark::{BatchedRelaxedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROTrait, TranscriptEngineTrait,
  },
  Commitment, DerandKey,
};
use ff::Field;
use serde::{Deserialize, Serialize};

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug, Clone)]
pub struct ProverKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  primary: S1::ProverKey,
  secondary: S2::ProverKey,
}

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug, Clone)]
pub struct VerifierKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  primary: S1::VerifierKey,
  secondary: S2::VerifierKey,
  dk_primary: DerandKey<E>,
  dk_secondary: DerandKey<Dual<E>>,
}

/// A SNARK that proves the knowledge of a valid Nebula proof
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  snark_primary: S1,
  snark_secondary: S2,
  nifs_F: PrimaryNIFS<E>,
  nifs_ops: PrimaryNIFS<E>,
  nifs_scan: PrimaryNIFS<E>,
  r_U: Vec<RelaxedR1CSInstance<E>>,
  l_u: Vec<R1CSInstance<E>>,
  r_U_secondary: Vec<RelaxedR1CSInstance<Dual<E>>>,
  nebula_instance: NebulaInstance<E>,
  F_zi: Vec<E::Scalar>,
  scan_zi: Vec<E::Scalar>,
  ops_zi: Vec<E::Scalar>,

  // F data
  num_steps_F: usize,
  prev_IC_F: E::Scalar,
  comm_omega_prev_F: Commitment<E>,
  nifs_random_F: PrimaryRelaxedNIFS<E>,
  wit_blind_F: E::Scalar,
  err_blind_F: E::Scalar,
  U_random_F: RelaxedR1CSInstance<E>,
  r_i_F: E::Scalar,

  // ops data
  num_steps_ops: usize,
  prev_IC_ops: E::Scalar,
  comm_omega_prev_ops: Commitment<E>,
  nifs_random_ops: PrimaryRelaxedNIFS<E>,
  wit_blind_ops: E::Scalar,
  err_blind_ops: E::Scalar,
  U_random_ops: RelaxedR1CSInstance<E>,
  r_i_ops: E::Scalar,

  // scan data
  num_steps_scan: usize,
  prev_IC_scan: (E::Scalar, E::Scalar),
  comm_omega_prev_scan: (Commitment<E>, Commitment<E>),
  nifs_random_scan: PrimaryRelaxedNIFS<E>,
  wit_blind_scan: E::Scalar,
  err_blind_scan: E::Scalar,
  U_random_scan: RelaxedR1CSInstance<E>,
  r_i_scan: E::Scalar,

  // CycleFold data
  nifs_1_secondary: CycleFoldRelaxedNIFS<E>,
  nifs_2_secondary: CycleFoldRelaxedNIFS<E>,
  nifs_final_secondary: CycleFoldRelaxedNIFS<E>,
  U_random_secondary: RelaxedR1CSInstance<Dual<E>>,
  derandom_U_secondary: RelaxedR1CSInstance<Dual<E>>,
  wit_blind_secondary: <Dual<E> as Engine>::Scalar,
  err_blind_secondary: <Dual<E> as Engine>::Scalar,
}

impl<E, S1, S2> CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// Creates prover and verifier keys for [`CompressedSNARK`]
  pub fn setup(
    pp: &impl Layer1PPTrait<E>,
  ) -> Result<(ProverKey<E, S1, S2>, VerifierKey<E, S1, S2>), NovaError> {
    let (pk_primary, vk_primary) = S1::setup(pp.biggest_ck().clone(), pp.primary_r1cs_shapes())?;
    let (pk_secondary, vk_secondary) =
      S2::setup(pp.ck_secondary().clone(), pp.cyclefold_r1cs_shape())?;
    let prover_key = ProverKey {
      primary: pk_primary,
      secondary: pk_secondary,
    };
    let verifier_key = VerifierKey {
      primary: vk_primary,
      secondary: vk_secondary,
      dk_primary: E::CE::derand_key(pp.biggest_ck()),
      dk_secondary: <Dual<E> as Engine>::CE::derand_key(pp.ck_secondary()),
    };
    Ok((prover_key, verifier_key))
  }

  /// Create a new [`CompressedSNARK`]
  pub fn prove(
    pp: &impl Layer1PPTrait<E>,
    pk: &ProverKey<E, S1, S2>,
    rs: &impl Layer1RSTrait<E>,
    nebula_instance: NebulaInstance<E>,
  ) -> Result<Self, NovaError> {
    let r_U = rs.r_U_clone();
    let l_u = rs.l_u_clone();

    // Primary SNARK
    //
    // Fold's (U, W, u, w) into (U', W') and runs the folded instance witness pair though Spartan
    let (U_F, W_F, nifs_F, nifs_random_F, wit_blind_F, err_blind_F, U_random_F) =
      rs.F().fold_ivc_compression_step(pp.F())?;
    let (U_ops, W_ops, nifs_ops, nifs_random_ops, wit_blind_ops, err_blind_ops, U_random_ops) =
      rs.ops().fold_ivc_compression_step(pp.ops())?;
    let (
      U_scan,
      W_scan,
      nifs_scan,
      nifs_random_scan,
      wit_blind_scan,
      err_blind_scan,
      U_random_scan,
    ) = rs.scan().fold_ivc_compression_step(pp.scan())?;
    let U = vec![U_F, U_ops, U_scan];
    let W = vec![W_F, W_ops, W_scan];
    let snark_primary = S1::prove(
      pp.biggest_ck(),
      &pk.primary,
      pp.primary_r1cs_shapes(),
      &U,
      &W,
    )?;

    // Secondary SNARK
    //
    // Run the CycleFold instances through Spartan
    let U_secondary = rs.r_U_secondary_clone();
    let (
      nifs_1_secondary,
      nifs_2_secondary,
      nifs_final_secondary,
      derandom_U_secondary,
      derandom_W_secondary,
      U_random_secondary,
      wit_blind_secondary,
      err_blind_secondary,
    ) = rs.fold_cyclefold_derandom(pp)?;
    let snark_secondary = S2::prove(
      pp.ck_secondary(),
      &pk.secondary,
      pp.cyclefold_r1cs_shape(),
      &derandom_U_secondary,
      &derandom_W_secondary,
    )?;

    Ok(Self {
      snark_primary,
      snark_secondary,
      nifs_F,
      nifs_ops,
      nifs_scan,
      r_U,
      l_u,
      r_U_secondary: U_secondary,
      nebula_instance,
      F_zi: rs.F().zi.clone(),
      scan_zi: rs.scan().zi.clone(),
      ops_zi: rs.ops().zi.clone(),

      // F data
      num_steps_F: rs.F().num_steps(),
      prev_IC_F: rs.F().prev_IC,
      comm_omega_prev_F: rs.F().comm_omega_prev,
      nifs_random_F,
      wit_blind_F,
      err_blind_F,
      U_random_F,
      r_i_F: rs.F().r_i(),

      // ops data
      num_steps_ops: rs.ops().num_steps(),
      prev_IC_ops: rs.ops().prev_IC,
      comm_omega_prev_ops: rs.ops().comm_omega_prev,
      nifs_random_ops,
      wit_blind_ops,
      err_blind_ops,
      U_random_ops,
      r_i_ops: rs.ops().r_i(),

      // scan data
      num_steps_scan: rs.scan().num_steps(),
      prev_IC_scan: rs.scan().prev_IC,
      comm_omega_prev_scan: rs.scan().comm_omega_prev,
      nifs_random_scan,
      wit_blind_scan,
      err_blind_scan,
      U_random_scan,
      r_i_scan: rs.scan().r_i(),

      // CycleFold data
      nifs_1_secondary,
      nifs_2_secondary,
      nifs_final_secondary,
      derandom_U_secondary,
      wit_blind_secondary,
      err_blind_secondary,
      U_random_secondary,
    })
  }

  /// Verify the correctness of the [`CompressedSNARK`]
  pub fn verify(
    &self,
    pp: &impl Layer1PPTrait<E>,
    vk: &VerifierKey<E, S1, S2>,
  ) -> Result<(), NovaError> {
    // TODO: Refactor hash checks

    // F hash check
    {
      // Calculate the hashes of the primary running instance and cyclefold running instance
      let (hash_primary, hash_cyclefold) = {
        let mut hasher_p = <Dual<E> as Engine>::RO::new(
          pp.F().ro_consts.clone(),
          4 + 2 * pp.F().F_arity_primary + 2 * NUM_FE_IN_EMULATED_POINT + 3, // (digest, num_steps, prev_IC) + 2 * arity "(z0, zi)" + U
        );
        hasher_p.absorb(pp.F().digest());
        hasher_p.absorb(E::Scalar::from(self.num_steps_F as u64));
        for e in &self.nebula_instance.execution_z0 {
          hasher_p.absorb(*e);
        }
        for e in &self.F_zi {
          hasher_p.absorb(*e);
        }
        absorb_primary_relaxed_r1cs::<E, Dual<E>>(&self.r_U[0], &mut hasher_p);
        hasher_p.absorb(self.prev_IC_F);
        hasher_p.absorb(self.r_i_F);
        let hash_primary = hasher_p.squeeze(NUM_HASH_BITS);
        let mut hasher_c = <Dual<E> as Engine>::RO::new(
          pp.F().ro_consts.clone(),
          1 + 1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS,
        );
        hasher_c.absorb(pp.F().digest());
        hasher_c.absorb(E::Scalar::from(self.num_steps_F as u64));
        self.r_U_secondary[0].absorb_in_ro(&mut hasher_c);
        hasher_c.absorb(self.r_i_F);
        let hash_cyclefold = hasher_c.squeeze(NUM_HASH_BITS);
        (hash_primary, hash_cyclefold)
      };

      // Verify the hashes equal the public IO for the final primary instance
      if scalar_as_base::<Dual<E>>(hash_primary) != self.l_u[0].X[0]
        || scalar_as_base::<Dual<E>>(hash_cyclefold) != self.l_u[0].X[1]
      {
        return Err(NovaError::ProofVerifyError);
      }

      // Abort if C_i  != hash(C_i−1, C_ω_i−1)
      let intermediary_comm =
        IC::<E>::increment_comm_w(&pp.F().ro_consts, self.prev_IC_F, self.comm_omega_prev_F);
      if self.nebula_instance.IC_i != intermediary_comm {
        return Err(NovaError::InvalidIC);
      }
    }

    // ops hash check
    {
      // Calculate the hashes of the primary running instance and cyclefold running instance
      let (hash_primary, hash_cyclefold) = {
        let mut hasher_p = <Dual<E> as Engine>::RO::new(
          pp.F().ro_consts.clone(),
          4 + 2 * pp.ops().F_arity_primary + 2 * NUM_FE_IN_EMULATED_POINT + 3, // (digest, num_steps, prev_IC) + 2 * arity "(z0, zi)" + U
        );
        hasher_p.absorb(pp.ops().digest());
        hasher_p.absorb(E::Scalar::from(self.num_steps_ops as u64));
        for e in &self.nebula_instance.ops_z0 {
          hasher_p.absorb(*e);
        }
        for e in &self.ops_zi {
          hasher_p.absorb(*e);
        }
        absorb_primary_relaxed_r1cs::<E, Dual<E>>(&self.r_U[1], &mut hasher_p);
        hasher_p.absorb(self.prev_IC_ops);
        hasher_p.absorb(self.r_i_ops);
        let hash_primary = hasher_p.squeeze(NUM_HASH_BITS);
        let mut hasher_c = <Dual<E> as Engine>::RO::new(
          pp.F().ro_consts.clone(),
          1 + 1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS,
        );
        hasher_c.absorb(pp.ops().digest());
        hasher_c.absorb(E::Scalar::from(self.num_steps_ops as u64));
        self.r_U_secondary[1].absorb_in_ro(&mut hasher_c);
        hasher_c.absorb(self.r_i_ops);
        let hash_cyclefold = hasher_c.squeeze(NUM_HASH_BITS);
        (hash_primary, hash_cyclefold)
      };

      // Verify the hashes equal the public IO for the final primary instance
      if scalar_as_base::<Dual<E>>(hash_primary) != self.l_u[1].X[0]
        || scalar_as_base::<Dual<E>>(hash_cyclefold) != self.l_u[1].X[1]
      {
        return Err(NovaError::ProofVerifyError);
      }

      // Abort if C_i  != hash(C_i−1, C_ω_i−1)
      let intermediary_comm = IC::<E>::increment_comm_w(
        &pp.F().ro_consts,
        self.prev_IC_ops,
        self.comm_omega_prev_ops,
      );
      if self.nebula_instance.ops_IC_i != intermediary_comm {
        return Err(NovaError::InvalidIC);
      }
    }

    // scan hash check
    {
      // Calculate the hashes of the primary running instance and cyclefold running instance
      let (hash_primary, hash_cyclefold) = {
        let mut hasher_p = <Dual<E> as Engine>::RO::new(
          pp.F().ro_consts.clone(),
          5 + 2 * pp.scan().F_arity_primary + 2 * NUM_FE_IN_EMULATED_POINT + 3, // (digest, num_steps, prev_IC) + 2 * arity "(z0, zi)" + U
        );
        hasher_p.absorb(pp.scan().digest());
        hasher_p.absorb(E::Scalar::from(self.num_steps_scan as u64));
        for e in &self.nebula_instance.scan_z0 {
          hasher_p.absorb(*e);
        }
        for e in &self.scan_zi {
          hasher_p.absorb(*e);
        }
        absorb_primary_relaxed_r1cs::<E, Dual<E>>(&self.r_U[2], &mut hasher_p);
        hasher_p.absorb(self.prev_IC_scan.0);
        hasher_p.absorb(self.prev_IC_scan.1);
        hasher_p.absorb(self.r_i_scan);
        let hash_primary = hasher_p.squeeze(NUM_HASH_BITS);
        let mut hasher_c = <Dual<E> as Engine>::RO::new(
          pp.F().ro_consts.clone(),
          1 + 1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS,
        );
        hasher_c.absorb(pp.scan().digest());
        hasher_c.absorb(E::Scalar::from(self.num_steps_scan as u64));
        self.r_U_secondary[2].absorb_in_ro(&mut hasher_c);
        hasher_c.absorb(self.r_i_scan);
        let hash_cyclefold = hasher_c.squeeze(NUM_HASH_BITS);
        (hash_primary, hash_cyclefold)
      };

      // Verify the hashes equal the public IO for the final primary instance
      if scalar_as_base::<Dual<E>>(hash_primary) != self.l_u[2].X[0]
        || scalar_as_base::<Dual<E>>(hash_cyclefold) != self.l_u[2].X[1]
      {
        return Err(NovaError::ProofVerifyError);
      }

      // Abort if C_i  != hash(C_i−1, C_ω_i−1)
      let intermediary_comm_IS = IC::<E>::increment_comm_w(
        &pp.F().ro_consts,
        self.prev_IC_scan.0,
        self.comm_omega_prev_scan.0,
      );
      let intermediary_comm_FS = IC::<E>::increment_comm_w(
        &pp.F().ro_consts,
        self.prev_IC_scan.1,
        self.comm_omega_prev_scan.1,
      );
      if self.nebula_instance.scan_IC_i.0 != intermediary_comm_IS
        || self.nebula_instance.scan_IC_i.1 != intermediary_comm_FS
      {
        return Err(NovaError::InvalidIC);
      }
    }

    // 1. check h_IS = h_RS = h_WS = h_FS = 1 // initial values are correct
    let (init_h_is, init_h_rs, init_h_ws, init_h_fs) = {
      (
        self.nebula_instance.scan_z0[2],
        self.nebula_instance.ops_z0[3],
        self.nebula_instance.ops_z0[4],
        self.nebula_instance.scan_z0[3],
      )
    };
    if init_h_is != E::Scalar::ONE
      || init_h_rs != E::Scalar::ONE
      || init_h_ws != E::Scalar::ONE
      || init_h_fs != E::Scalar::ONE
    {
      return Err(NovaError::ProofVerifyError);
    }

    // 2. check Cn′ = Cn // commitments carried in both Πops and ΠF are the same
    if self.nebula_instance.IC_i != self.nebula_instance.ops_IC_i {
      return Err(NovaError::ProofVerifyError);
    }

    // 3. check γ and γ are derived by hashing C and C′′.
    // Get alpha and gamma
    let mut keccak = E::TE::new(b"compute MCC challenges");
    keccak.absorb(b"C_n", &self.nebula_instance.IC_i);
    keccak.absorb(b"IC_IS", &self.nebula_instance.scan_IC_i.0);
    keccak.absorb(b"IC_FS", &self.nebula_instance.scan_IC_i.1);
    let gamma = keccak.squeeze(b"gamma")?;
    let alpha = keccak.squeeze(b"alpha")?;

    if self.nebula_instance.ops_z0[0] != gamma || self.nebula_instance.ops_z0[1] != alpha {
      return Err(NovaError::ProofVerifyError);
    }

    // 4. check h_IS' · h_WS' = h_RS' · h_FS'.
    // Inputs for multiset check
    let (h_is, h_rs, h_ws, h_fs) = {
      (
        self.scan_zi[2],
        self.ops_zi[3],
        self.ops_zi[4],
        self.scan_zi[3],
      )
    };
    if h_is * h_ws != h_rs * h_fs {
      return Err(NovaError::ProofVerifyError);
    }

    // Verify Primary SNARK
    let U_F_f = self.nifs_F.verify(
      &pp.F().ro_consts,
      &pp.F().digest(),
      &self.r_U[0],
      &self.l_u[0],
    );
    let U_ops_f = self.nifs_ops.verify(
      &pp.ops().ro_consts,
      &pp.ops().digest(),
      &self.r_U[1],
      &self.l_u[1],
    );
    let U_scan_f = self.nifs_scan.verify(
      &pp.scan().ro_consts,
      &pp.scan().digest(),
      &self.r_U[2],
      &self.l_u[2],
    );
    let U_F = self.nifs_random_F.verify(
      &pp.F().ro_consts,
      &pp.F().digest(),
      &U_F_f,
      &self.U_random_F,
    );
    let U_ops = self.nifs_random_ops.verify(
      &pp.ops().ro_consts,
      &pp.ops().digest(),
      &U_ops_f,
      &self.U_random_ops,
    );
    let U_scan = self.nifs_random_scan.verify(
      &pp.scan().ro_consts,
      &pp.scan().digest(),
      &U_scan_f,
      &self.U_random_scan,
    );
    let U_F_derandom = U_F.derandomize(&vk.dk_primary, &self.wit_blind_F, &self.err_blind_F);
    let U_ops_derandom =
      U_ops.derandomize(&vk.dk_primary, &self.wit_blind_ops, &self.err_blind_ops);
    let U_scan_derandom =
      U_scan.derandomize(&vk.dk_primary, &self.wit_blind_scan, &self.err_blind_scan);
    let U = vec![U_F_derandom, U_ops_derandom, U_scan_derandom];
    self.snark_primary.verify(&vk.primary, &U)?;

    // Verify secondary SNARK
    let U_temp_1 = self.nifs_1_secondary.verify(
      pp.ro_consts(),
      &self.r_U_secondary[0],
      &self.r_U_secondary[1],
    );
    let U_temp_2 = self
      .nifs_2_secondary
      .verify(pp.ro_consts(), &U_temp_1, &self.r_U_secondary[2]);
    let U = self
      .nifs_final_secondary
      .verify(pp.ro_consts(), &U_temp_2, &self.U_random_secondary);
    let derandom_U = U.derandomize(
      &vk.dk_secondary,
      &self.wit_blind_secondary,
      &self.err_blind_secondary,
    );
    self.snark_secondary.verify(&vk.secondary, &derandom_U)?;
    Ok(())
  }
}

/// Public i/o for WASM execution proving
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NebulaInstance<E>
where
  E: CurveCycleEquipped,
{
  // execution instance
  execution_z0: Vec<E::Scalar>,
  IC_i: E::Scalar,

  // ops instance
  ops_z0: Vec<E::Scalar>,
  ops_IC_i: E::Scalar,

  // scan instance
  scan_z0: Vec<E::Scalar>,
  scan_IC_i: (E::Scalar, E::Scalar),
}

impl<E> NebulaInstance<E>
where
  E: CurveCycleEquipped,
{
  /// Create a new [`NebulaInstance`]
  pub fn new(
    execution_z0: Vec<E::Scalar>,
    IC_i: E::Scalar,
    ops_z0: Vec<E::Scalar>,
    ops_IC_i: E::Scalar,
    scan_z0: Vec<E::Scalar>,
    scan_IC_i: (E::Scalar, E::Scalar),
  ) -> Self {
    Self {
      execution_z0,
      IC_i,
      ops_z0,
      ops_IC_i,
      scan_z0,
      scan_IC_i,
    }
  }
}
