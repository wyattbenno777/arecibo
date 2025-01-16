//! Implements components to enable the compression-step for IVC proofs

use super::{
  nifs::PrimaryNIFS,
  traits::{Layer1PPTrait, Layer1RSTrait},
};
use crate::{
  errors::NovaError,
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{snark::BatchedRelaxedR1CSSNARKTrait, CurveCycleEquipped, Dual},
};
use serde::{Deserialize, Serialize};

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug, Clone)]
pub struct ProverKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: BatchedRelaxedR1CSSNARKTrait<Dual<E>>,
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
  S2: BatchedRelaxedR1CSSNARKTrait<Dual<E>>,
{
  primary: S1::VerifierKey,
  secondary: S2::VerifierKey,
}

/// A SNARK that proves the knowledge of a valid Nebula proof
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: BatchedRelaxedR1CSSNARKTrait<Dual<E>>,
{
  snark_primary: S1,
  snark_secondary: S2,
  nifs_F: PrimaryNIFS<E>,
  nifs_ops: PrimaryNIFS<E>,
  nifs_scan: PrimaryNIFS<E>,
  r_U: Vec<RelaxedR1CSInstance<E>>,
  l_u: Vec<R1CSInstance<E>>,
  r_U_secondary: Vec<RelaxedR1CSInstance<Dual<E>>>,
}

impl<E, S1, S2> CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: BatchedRelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// Creates prover and verifier keys for [`CompressedSNARK`]
  pub fn setup(
    pp: &impl Layer1PPTrait<E>,
  ) -> Result<(ProverKey<E, S1, S2>, VerifierKey<E, S1, S2>), NovaError> {
    let (pk_primary, vk_primary) = S1::setup(pp.biggest_ck().clone(), pp.primary_r1cs_shapes())?;

    let (pk_secondary, vk_secondary) =
      S2::setup(pp.ck_secondary().clone(), pp.secondary_r1cs_shapes())?;

    let prover_key = ProverKey {
      primary: pk_primary,
      secondary: pk_secondary,
    };
    let verifier_key = VerifierKey {
      primary: vk_primary,
      secondary: vk_secondary,
    };

    Ok((prover_key, verifier_key))
  }

  /// Create a new [`CompressedSNARK`]
  pub fn prove(
    pp: &impl Layer1PPTrait<E>,
    pk: &ProverKey<E, S1, S2>,
    rs: &impl Layer1RSTrait<E>,
  ) -> Result<Self, NovaError> {
    let r_U = vec![
      rs.F().r_U_primary.clone(),
      rs.ops().r_U_primary.clone(),
      rs.scan().r_U_primary.clone(),
    ];
    let l_u = vec![
      rs.F().l_u_primary.clone(),
      rs.ops().l_u_primary.clone(),
      rs.scan().l_u_primary.clone(),
    ];

    // Primary SNARK
    //
    // Fold's (U, W, u, w) into (U', W') and runs the folded instance witness pair though Spartan
    let (U_F, W_F, nifs_F) = rs.F().fold_ivc_compression_step(pp.F())?;
    let (U_ops, W_ops, nifs_ops) = rs.ops().fold_ivc_compression_step(pp.ops())?;
    let (U_scan, W_scan, nifs_scan) = rs.scan().fold_ivc_compression_step(pp.scan())?;
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

    // TODO: refactor by folding these cyclefold relaxed R1CS instance witness pairs into one relaxed R1CS instance witness pair
    let (U_F_secondary, W_F_secondary) = rs.F().secondary_rs_part();
    let (U_ops_secondary, W_ops_secondary) = rs.ops().secondary_rs_part();
    let (U_scan_secondary, W_scan_secondary) = rs.scan().secondary_rs_part();
    let U_secondary = vec![
      U_F_secondary.clone(),
      U_ops_secondary.clone(),
      U_scan_secondary.clone(),
    ];
    let W_secondary = vec![
      W_F_secondary.clone(),
      W_ops_secondary.clone(),
      W_scan_secondary.clone(),
    ];
    let snark_secondary = S2::prove(
      pp.ck_secondary(),
      &pk.secondary,
      pp.secondary_r1cs_shapes(),
      &U_secondary,
      &W_secondary,
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
    })
  }

  /// Verify the correctness of the [`CompressedSNARK`]
  pub fn verify(
    &self,
    pp: &impl Layer1PPTrait<E>,
    vk: &VerifierKey<E, S1, S2>,
  ) -> Result<(), NovaError> {
    let U_F = self.nifs_F.verify(
      &pp.F().ro_consts,
      &pp.F().digest(),
      &self.r_U[0],
      &self.l_u[0],
    );
    let U_ops = self.nifs_ops.verify(
      &pp.ops().ro_consts,
      &pp.ops().digest(),
      &self.r_U[1],
      &self.l_u[1],
    );
    let U_scan = self.nifs_scan.verify(
      &pp.scan().ro_consts,
      &pp.scan().digest(),
      &self.r_U[2],
      &self.l_u[2],
    );
    let U = vec![U_F, U_ops, U_scan];
    self.snark_primary.verify(&vk.primary, &U)?;
    self
      .snark_secondary
      .verify(&vk.secondary, &self.r_U_secondary)?;
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
