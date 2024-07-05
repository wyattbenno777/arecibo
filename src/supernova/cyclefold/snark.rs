//! This module defines a final compressing SNARK for supernova proofs (with CycleFold)
use super::{PublicParams, RecursiveSNARK};
use crate::supernova::error::SuperNovaError;
use crate::{
  constants::NUM_HASH_BITS,
  r1cs::{R1CSInstance, RelaxedR1CSWitness},
  traits::{
    snark::{BatchedRelaxedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROTrait,
  },
};
use crate::{errors::NovaError, scalar_as_base, RelaxedR1CSInstance, NIFS};

use ff::PrimeField;
use serde::{Deserialize, Serialize};

/// A type that holds the prover key for `CompressedSNARK`
#[derive(Debug)]
pub struct ProverKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  pk_primary: S1::ProverKey,
  pk_cyclefold: S2::ProverKey,
}

/// A type that holds the verifier key for `CompressedSNARK`
#[derive(Debug)]
pub struct VerifierKey<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  vk_primary: S1::VerifierKey,
  vk_cyclefold: S2::VerifierKey,
}

/// A SNARK that proves the knowledge of a valid `RecursiveSNARK`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  r_U_primary: Vec<RelaxedR1CSInstance<E1>>,
  r_W_snark_primary: S1,

  r_U_cyclefold: RelaxedR1CSInstance<Dual<E1>>,
  l_u_cyclefold: R1CSInstance<Dual<E1>>,
  nifs_cyclefold: NIFS<Dual<E1>>,
  f_W_snark_cyclefold: S2,

  num_steps: usize,
  program_counter: E1::Scalar,

  zn_primary: Vec<E1::Scalar>,
  zn_cyclefold: Vec<<Dual<E1> as Engine>::Scalar>,
}

impl<E1, S1, S2> CompressedSNARK<E1, S1, S2>
where
  E1: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  /// Creates prover and verifier keys for `CompressedSNARK`
  pub fn setup(
    pp: &PublicParams<E1>,
  ) -> Result<(ProverKey<E1, S1, S2>, VerifierKey<E1, S1, S2>), SuperNovaError> {
    let (pk_primary, vk_primary) = S1::setup(pp.ck_primary.clone(), pp.primary_r1cs_shapes())?;

    let (pk_cyclefold, vk_cyclefold) = S2::setup(
      pp.ck_cyclefold.clone(),
      &pp.circuit_shape_cyclefold.r1cs_shape,
    )?;

    let prover_key = ProverKey {
      pk_primary,
      pk_cyclefold,
    };
    let verifier_key = VerifierKey {
      vk_primary,
      vk_cyclefold,
    };

    Ok((prover_key, verifier_key))
  }
}
