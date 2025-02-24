//! This module provides the components needed to compress the HyperNova IVC proofs with Spartan.

use std::marker::PhantomData;

use super::rs::{PublicParams, RecursiveSNARK};
use crate::{
  hypernova::nifs::PartialNIFS,
  r1cs::{
    split::{LR1CSInstance, SplitR1CSWitness},
    RelaxedR1CSInstance, RelaxedR1CSWitness,
  },
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
  nifs: PartialNIFS<E>,
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
    // Fold r_U and l_u & derand the commitments to the witness
    let (nifs, (U, W), (U_cyclefold, W_cyclefold)) = Self::derand_nifs(pp, rs)?;

    // Apply Spartan on primary and secondary curve for instance witness pairs
    // (new_U, new_W) & (r_U_cyclefold, r_W_cyclefold)
    let (snark, snark_cyclefold) = rayon::join(
      || S1::prove(&pp.ck, &pk.primary, &pp.circuit_shape.r1cs_shape, &U, &W),
      || {
        S2::prove(
          &pp.ck_cyclefold,
          &pk.secondary,
          &pp.circuit_shape_cyclefold.r1cs_shape,
          &U_cyclefold,
          &W_cyclefold,
        )
      },
    );
    Ok(Self {
      snark: snark?,
      snark_cyclefold: snark_cyclefold?,
      nifs,
    })
  }
  fn verify(&self, vk: &VerifierKey<E, S1, S2>) -> Result<(), NovaError> {
    Ok(())
  }
}

impl<E, S1, S2> CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  fn derand_nifs(
    pp: &PublicParams<E>,
    rs: &RecursiveSNARK<E>,
  ) -> Result<
    (
      PartialNIFS<E>,
      (LR1CSInstance<E>, SplitR1CSWitness<E>),
      (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>),
    ),
    NovaError,
  > {
    // Fold r_U and l_u
    let (nifs, (U, W), _) = PartialNIFS::prove(
      &pp.circuit_shape.r1cs_shape,
      &pp.ro_consts,
      &pp.digest(),
      (&rs.r_U, &rs.r_W),
      (&rs.l_u, &rs.l_w),
    )?;

    // --- Derand the commitments to the witness ---
    // (U, W)
    let (derand_W, wit_blind) = W.derandomize();
    let derand_U = U.derandomize(&E::CE::derand_key(&pp.ck), &wit_blind);
    // (U_cyclefold, W_cyclefold)
    let (derand_W_cyclefold, wit_blind_cyclefold, err_blind_cyclefold) =
      rs.r_W_cyclefold.derandomize();
    let derand_U_cyclefold = rs.r_U_cyclefold.derandomize(
      &<Dual<E> as Engine>::CE::derand_key(&pp.ck_cyclefold),
      &wit_blind_cyclefold,
      &err_blind_cyclefold,
    );
    Ok((
      nifs,
      (derand_U, derand_W),
      (derand_U_cyclefold, derand_W_cyclefold),
    ))
  }
}

#[cfg(test)]
mod tests {}
