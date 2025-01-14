//! Implements components to enable the compression-step for IVC proofs

use super::{
  nifs::PrimaryNIFS,
  traits::{Layer1PPTrait, Layer1RSTrait},
};
use crate::{
  errors::NovaError,
  traits::{snark::BatchedRelaxedR1CSSNARKTrait, CurveCycleEquipped, Dual},
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

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
}

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug)]
pub struct ProverKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: BatchedRelaxedR1CSSNARKTrait<Dual<E>>,
{
  _engine: PhantomData<E>,
  primary: S1::ProverKey,
  secondary: S2::ProverKey,
}

impl<E, S1, S2> CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: BatchedRelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// Create a new [`CompressedSNARK`]
  pub fn prove(
    pp: &impl Layer1PPTrait<E>,
    pk: &ProverKey<E, S1, S2>,
    rs: &impl Layer1RSTrait<E>,
  ) -> Result<Self, NovaError> {
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
    let snark_secondary = {
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
      let secondary_shapes = vec![
        &pp.F().circuit_shape_cyclefold.r1cs_shape,
        &pp.ops().circuit_shape_cyclefold.r1cs_shape,
        &pp.scan().circuit_shape_cyclefold.r1cs_shape,
      ];
      S2::prove(
        &pp.F().ck_cyclefold,
        &pk.secondary,
        secondary_shapes,
        &U_secondary,
        &W_secondary,
      )?
    };

    Ok(Self {
      snark_primary,
      snark_secondary,
      nifs_F,
      nifs_ops,
      nifs_scan,
    })
  }
}
