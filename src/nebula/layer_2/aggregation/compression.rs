//! Applies Spartan on top of the Layer 2 proofs.

use super::{AggregationPublicParams, AggregationRecursiveSNARK};
use crate::{
  errors::NovaError,
  nebula::nifs::PrimaryNIFS,
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{snark::BatchedRelaxedR1CSSNARKTrait, CurveCycleEquipped, Dual},
};
use serde::{Deserialize, Serialize};

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug)]
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
#[derive(Debug)]
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
  nifs_verifier: PrimaryNIFS<E>,
  r_U_F: RelaxedR1CSInstance<E>,
  r_U_scan: RelaxedR1CSInstance<E>,
  r_U_ops: RelaxedR1CSInstance<E>,
  r_U_verifier: RelaxedR1CSInstance<E>,
  l_u_verifier: R1CSInstance<E>,
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
    pp: &AggregationPublicParams<E>,
  ) -> Result<(ProverKey<E, S1, S2>, VerifierKey<E, S1, S2>), NovaError> {
    let (pk_primary, vk_primary) = S1::setup(pp.ck.clone(), pp.primary_r1cs_shapes())?;
    let (pk_secondary, vk_secondary) =
      S2::setup(pp.pp.ck_cyclefold.clone(), pp.secondary_r1cs_shapes())?;
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
    pp: &AggregationPublicParams<E>,
    pk: &ProverKey<E, S1, S2>,
    rs: &AggregationRecursiveSNARK<E>,
  ) -> Result<Self, NovaError> {
    let r_U_verifier = rs.rs.r_U_primary.clone();
    let l_u_verifier = rs.rs.l_u_primary.clone();
    // Primary SNARK
    //
    // Fold's (U, W, u, w) into (U', W') and runs the folded instance witness pair though Spartan
    let (U_verifier, W_verifier, nifs_verifier, _, _, _, _) =
      rs.rs.fold_ivc_compression_step(&pp.pp)?;
    let U = vec![
      rs.r_U_F.clone(),
      rs.r_U_ops.clone(),
      rs.r_U_scan.clone(),
      U_verifier,
    ];
    let W = vec![
      rs.r_W_F.clone(),
      rs.r_W_ops.clone(),
      rs.r_W_scan.clone(),
      W_verifier,
    ];
    let snark_primary = S1::prove(&pp.ck, &pk.primary, pp.primary_r1cs_shapes(), &U, &W)?;

    // Secondary SNARK
    //
    // Run the CycleFold instances through Spartan

    // TODO: refactor by folding these cyclefold relaxed R1CS instance witness pairs into one relaxed R1CS instance witness pair
    let (r_U_secondary_verifier, r_W_secondary_verifier) = rs.rs.secondary_rs_part();
    let U_secondary = vec![rs.r_U_cyclefold.clone(), r_U_secondary_verifier.clone()];
    let W_secondary = vec![rs.r_W_cyclefold.clone(), r_W_secondary_verifier.clone()];
    let snark_secondary = S2::prove(
      &pp.pp.ck_cyclefold,
      &pk.secondary,
      pp.secondary_r1cs_shapes(),
      &U_secondary,
      &W_secondary,
    )?;

    Ok(Self {
      snark_primary,
      snark_secondary,
      nifs_verifier,
      r_U_secondary: U_secondary,
      r_U_F: rs.r_U_F.clone(),
      r_U_scan: rs.r_U_scan.clone(),
      r_U_ops: rs.r_U_ops.clone(),
      r_U_verifier,
      l_u_verifier,
    })
  }

  /// Verify the correctness of the [`CompressedSNARK`]
  pub fn verify(
    &self,
    pp: &AggregationPublicParams<E>,
    vk: &VerifierKey<E, S1, S2>,
  ) -> Result<(), NovaError> {
    let U_verifier = self.nifs_verifier.verify(
      &pp.pp.ro_consts,
      &pp.pp.digest(),
      &self.r_U_verifier,
      &self.l_u_verifier,
    );
    let U = vec![
      self.r_U_F.clone(),
      self.r_U_ops.clone(),
      self.r_U_scan.clone(),
      U_verifier,
    ];
    self.snark_primary.verify(&vk.primary, &U)?;
    self
      .snark_secondary
      .verify(&vk.secondary, &self.r_U_secondary)?;
    Ok(())
  }
}
