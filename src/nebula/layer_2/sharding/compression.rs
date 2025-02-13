//! Applies Spartan on top of the Layer 2 proofs.

use super::{ShardingPublicParams, ShardingRecursiveSNARK};
use crate::{
  errors::NovaError,
  nebula::nifs::{CycleFoldRelaxedNIFS, PrimaryNIFS, PrimaryRelaxedNIFS},
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{
    commitment::CommitmentEngineTrait,
    snark::{BatchedRelaxedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    CurveCycleEquipped, Dual, Engine,
  },
  DerandKey,
};
use serde::{Deserialize, Serialize};

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug)]
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
#[derive(Debug)]
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

  // primary data
  nifs_verifier: PrimaryNIFS<E>,
  r_U_verifier: RelaxedR1CSInstance<E>,
  l_u_verifier: R1CSInstance<E>,
  nifs_r_verifier: PrimaryRelaxedNIFS<E>,
  random_U_verifier: RelaxedR1CSInstance<E>,
  wit_blind_verifier: E::Scalar,
  err_blind_verifier: E::Scalar,

  // F data
  r_U_F: RelaxedR1CSInstance<E>,
  nifs_r_F: PrimaryRelaxedNIFS<E>,
  random_U_F: RelaxedR1CSInstance<E>,
  wit_blind_F: E::Scalar,
  err_blind_F: E::Scalar,

  // ops data
  r_U_ops: RelaxedR1CSInstance<E>,
  nifs_r_ops: PrimaryRelaxedNIFS<E>,
  random_U_ops: RelaxedR1CSInstance<E>,
  wit_blind_ops: E::Scalar,
  err_blind_ops: E::Scalar,

  // scan data
  r_U_scan: RelaxedR1CSInstance<E>,
  nifs_r_scan: PrimaryRelaxedNIFS<E>,
  random_U_scan: RelaxedR1CSInstance<E>,
  wit_blind_scan: E::Scalar,
  err_blind_scan: E::Scalar,

  // cyclefold data
  nifs_1_secondary: CycleFoldRelaxedNIFS<E>,
  r_U_cyclefold: RelaxedR1CSInstance<Dual<E>>,
  r_U_secondary_verifier: RelaxedR1CSInstance<Dual<E>>,
  nifs_final_secondary: CycleFoldRelaxedNIFS<E>,
  U_random_secondary: RelaxedR1CSInstance<Dual<E>>,
  wit_blind_secondary: E::Base,
  err_blind_secondary: E::Base,
}

impl<E, S1, S2> CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// Creates prover and verifier keys for [`CompressedSNARK`]
  pub fn setup(
    pp: &ShardingPublicParams<E>,
  ) -> Result<(ProverKey<E, S1, S2>, VerifierKey<E, S1, S2>), NovaError> {
    let (pk_primary, vk_primary) = S1::setup(pp.ck.clone(), pp.primary_r1cs_shapes())?;
    let (pk_secondary, vk_secondary) =
      S2::setup(pp.pp.ck_cyclefold.clone(), pp.r1cs_shape_cyclefold())?;
    let prover_key = ProverKey {
      primary: pk_primary,
      secondary: pk_secondary,
    };
    let verifier_key = VerifierKey {
      primary: vk_primary,
      secondary: vk_secondary,
      dk_primary: E::CE::derand_key(pp.ck()),
      dk_secondary: <Dual<E> as Engine>::CE::derand_key(pp.ck_cyclefold()),
    };

    Ok((prover_key, verifier_key))
  }

  /// Create a new [`CompressedSNARK`]
  pub fn prove(
    pp: &ShardingPublicParams<E>,
    pk: &ProverKey<E, S1, S2>,
    rs: &ShardingRecursiveSNARK<E>,
  ) -> Result<Self, NovaError> {
    let r_U_verifier = rs.rs.r_U_primary.clone();
    let l_u_verifier = rs.rs.l_u_primary.clone();

    // Primary SNARK
    //
    // Fold's (U, W, u, w) into (U', W') and runs the folded instance witness pair though Spartan
    let (
      // rs
      U_verifier,
      W_verifier,
      nifs_verifier,
      nifs_r_verifier,
      wit_blind_verifier,
      err_blind_verifier,
      random_U_verifier,
      // F
      derandom_U_F,
      derandom_W_F,
      nifs_r_F,
      wit_blind_F,
      err_blind_F,
      random_U_F,
      // ops
      derandom_U_ops,
      derandom_W_ops,
      nifs_r_ops,
      wit_blind_ops,
      err_blind_ops,
      random_U_ops,
      // scan
      derandom_U_scan,
      derandom_W_scan,
      nifs_r_scan,
      wit_blind_scan,
      err_blind_scan,
      random_U_scan,
    ) = rs.fold_derandom(pp)?;
    let U = vec![derandom_U_F, derandom_U_ops, derandom_U_scan, U_verifier];
    let W = vec![derandom_W_F, derandom_W_ops, derandom_W_scan, W_verifier];
    let snark_primary = S1::prove(&pp.ck, &pk.primary, pp.primary_r1cs_shapes(), &U, &W)?;

    // Secondary SNARK
    //
    // Run the CycleFold instances through Spartan
    let (
      U_secondary,
      W_secondary,
      nifs_1_secondary,
      nifs_final_secondary,
      U_random_secondary,
      wit_blind_secondary,
      err_blind_secondary,
    ) = rs.fold_derandom_secondary(pp)?;
    let snark_secondary = S2::prove(
      &pp.pp.ck_cyclefold,
      &pk.secondary,
      pp.r1cs_shape_cyclefold(),
      &U_secondary,
      &W_secondary,
    )?;

    Ok(Self {
      snark_primary,
      snark_secondary,
      // primary data
      nifs_verifier,
      nifs_r_verifier,
      r_U_verifier,
      l_u_verifier,
      random_U_verifier,
      wit_blind_verifier,
      err_blind_verifier,

      // F data
      r_U_F: rs.r_U_F.clone(),
      nifs_r_F,
      wit_blind_F,
      err_blind_F,
      random_U_F,

      // ops
      r_U_ops: rs.r_U_ops.clone(),
      nifs_r_ops,
      wit_blind_ops,
      err_blind_ops,
      random_U_ops,

      // scan
      r_U_scan: rs.r_U_scan.clone(),
      nifs_r_scan,
      wit_blind_scan,
      err_blind_scan,
      random_U_scan,

      // cyclefold data
      nifs_1_secondary,
      r_U_secondary_verifier: rs.rs.r_U_cyclefold.clone(),
      r_U_cyclefold: rs.r_U_cyclefold.clone(),
      nifs_final_secondary,
      U_random_secondary,
      wit_blind_secondary,
      err_blind_secondary,
    })
  }

  /// Verify the correctness of the [`CompressedSNARK`]
  pub fn verify(
    &self,
    pp: &ShardingPublicParams<E>,
    vk: &VerifierKey<E, S1, S2>,
  ) -> Result<(), NovaError> {
    // Primary SNARK
    let U_f = self.nifs_verifier.verify(
      &pp.pp.ro_consts,
      &pp.pp.digest(),
      &self.r_U_verifier,
      &self.l_u_verifier,
    );
    let U = self.nifs_r_verifier.verify(
      &pp.pp.ro_consts,
      &pp.pp.digest(),
      &U_f,
      &self.random_U_verifier,
    );
    let derandom_U = U.derandomize(
      &vk.dk_primary,
      &self.wit_blind_verifier,
      &self.err_blind_verifier,
    );

    // Check F
    let U_F = self.nifs_r_F.verify(
      &pp.pp.ro_consts,
      &pp.digest_F,
      &self.r_U_F,
      &self.random_U_F,
    );
    let derandom_U_F = U_F.derandomize(&vk.dk_primary, &self.wit_blind_F, &self.err_blind_F);

    // Check ops
    let U_ops = self.nifs_r_ops.verify(
      &pp.pp.ro_consts,
      &pp.digest_ops,
      &self.r_U_ops,
      &self.random_U_ops,
    );
    let derandom_U_ops =
      U_ops.derandomize(&vk.dk_primary, &self.wit_blind_ops, &self.err_blind_ops);

    // Check scan
    let U_scan = self.nifs_r_scan.verify(
      &pp.pp.ro_consts,
      &pp.digest_scan,
      &self.r_U_scan,
      &self.random_U_scan,
    );
    let derandom_U_scan =
      U_scan.derandomize(&vk.dk_primary, &self.wit_blind_scan, &self.err_blind_scan);

    let U = vec![derandom_U_F, derandom_U_ops, derandom_U_scan, derandom_U];
    self.snark_primary.verify(&vk.primary, &U)?;

    // Secondary SNARK
    let U_temp_1_secondary = self.nifs_1_secondary.verify(
      &pp.pp.ro_consts,
      &self.r_U_secondary_verifier,
      &self.r_U_cyclefold,
    );
    let U_secondary = self.nifs_final_secondary.verify(
      &pp.pp.ro_consts,
      &U_temp_1_secondary,
      &self.U_random_secondary,
    );
    let derandom_U_secondary = U_secondary.derandomize(
      &vk.dk_secondary,
      &self.wit_blind_secondary,
      &self.err_blind_secondary,
    );
    self
      .snark_secondary
      .verify(&vk.secondary, &derandom_U_secondary)?;
    Ok(())
  }
}
