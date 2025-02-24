//! This module provides the components needed to compress the HyperNova IVC proofs with Spartan.

use std::marker::PhantomData;

use super::rs::{PublicParams, RecursiveSNARK};
use crate::{
  traits::{
    commitment::CommitmentEngineTrait,
    snark::{LinearizedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    CurveCycleEquipped, Dual, Engine,
  },
  DerandKey, NovaError,
};

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug, Clone)]
pub struct ProverKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
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
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  primary: S1::VerifierKey,
  secondary: S2::VerifierKey,
  dk_primary: DerandKey<E>,
  dk_secondary: DerandKey<Dual<E>>,
}

/// A SNARK that proves the knowledge of a valid HyperNova [`RecursiveSNARK`]
pub struct CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  snark: S1,
  snark_cyclefold: S2,
  _engine: PhantomData<E>,
}

impl<E, S1, S2> CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  fn setup(
    pp: &PublicParams<E>,
  ) -> Result<(ProverKey<E, S1, S2>, VerifierKey<E, S1, S2>), NovaError> {
    let (pk_primary, vk_primary) = S1::setup(pp.ck.clone(), &pp.circuit_shape.r1cs_shape)?;
    let (pk_secondary, vk_secondary) = S2::setup(
      pp.ck_cyclefold.clone(),
      &pp.circuit_shape_cyclefold.r1cs_shape,
    )?;
    let prover_key = ProverKey {
      primary: pk_primary,
      secondary: pk_secondary,
    };
    let verifier_key = VerifierKey {
      primary: vk_primary,
      secondary: vk_secondary,
      dk_primary: E::CE::derand_key(&pp.ck),
      dk_secondary: <Dual<E> as Engine>::CE::derand_key(&pp.ck_cyclefold),
    };
    Ok((prover_key, verifier_key))
  }

  fn prove(
    pp: &PublicParams<E>,
    pk: &ProverKey<E, S1, S2>,
    rs: &RecursiveSNARK<E>,
  ) -> Result<Self, NovaError> {
    // --- Fold r_U and l_u ---
    // --- Derand the commitments to the witness ---
    // --- Apply Spartan on primary and secondary curve for instance witness pairs (new_r_U, new_r_W) & (r_U_cyclefold, r_W_cyclefold) ---
    todo!()
  }
  fn verify(&self, vk: &VerifierKey<E, S1, S2>) -> Result<(), NovaError> {
    Ok(())
  }
}

#[cfg(test)]
mod tests {}
