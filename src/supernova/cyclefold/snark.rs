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

  f_U_cyclefold: RelaxedR1CSInstance<Dual<E1>>,
  f_W_snark_cyclefold: S2,

  num_steps: usize,
  program_counter: E1::Scalar,

  zn_primary: Vec<E1::Scalar>,
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

  /// Create a new `CompressedSNARK`
  pub fn prove(
    pp: &PublicParams<E1>,
    pk: &ProverKey<E1, S1, S2>,
    recursive_snark: &RecursiveSNARK<E1>,
  ) -> Result<Self, SuperNovaError> {
    // Prepare the list of primary Relaxed R1CS instances (a default instance is provided for
    // uninitialized circuits)
    let r_U_primary = recursive_snark
      .r_U_primary
      .iter()
      .enumerate()
      .map(|(idx, r_U)| {
        r_U
          .clone()
          .unwrap_or_else(|| RelaxedR1CSInstance::default(&*pp.ck_primary, &pp[idx].r1cs_shape))
      })
      .collect::<Vec<_>>();

    // Prepare the list of primary relaxed R1CS witnesses (a default witness is provided for
    // uninitialized circuits)
    let r_W_primary: Vec<RelaxedR1CSWitness<E1>> = recursive_snark
      .r_W_primary
      .iter()
      .enumerate()
      .map(|(idx, r_W)| {
        r_W
          .clone()
          .unwrap_or_else(|| RelaxedR1CSWitness::default(&pp[idx].r1cs_shape))
      })
      .collect::<Vec<_>>();

    // Generate a primary SNARK proof for the list of primary circuits
    let r_W_snark_primary = S1::prove(
      &pp.ck_primary,
      &pk.pk_primary,
      pp.primary_r1cs_shapes(),
      &r_U_primary,
      &r_W_primary,
    )?;

    // Generate a cyclefold SNARK proof for the cyclefold circuit
    let f_W_snark_cyclefold = S2::prove(
      &pp.ck_cyclefold,
      &pk.pk_cyclefold,
      &pp.circuit_shape_cyclefold.r1cs_shape,
      &recursive_snark.r_U_cyclefold,
      &recursive_snark.r_W_cyclefold,
    )?;

    let compressed_snark = Self {
      r_U_primary,
      r_W_snark_primary,

      f_U_cyclefold: recursive_snark.r_U_cyclefold.clone(),
      f_W_snark_cyclefold,

      num_steps: recursive_snark.i,
      program_counter: recursive_snark.program_counter,

      zn_primary: recursive_snark.zi_primary.clone(),
    };

    Ok(compressed_snark)
  }

  /// Verify the correctness of the `CompressedSNARK`
  pub fn verify(
    &self,
    vk: &VerifierKey<E1, S1, S2>,
    z0_primary: &[E1::Scalar],
  ) -> Result<Vec<E1::Scalar>, SuperNovaError> {
    // let last_circuit_idx = field_as_usize(self.program_counter);

    // let num_field_primary_ro = 3 // params_next, i_new, program_counter_new
    // + 2 * pp[last_circuit_idx].F_arity // zo, z1
    // + (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // # 1 * (7 + [X0, X1]*#num_limb)

    // // secondary circuit
    // // NOTE: This count ensure the number of witnesses sent by the prover must equal the number of
    // // NIVC circuits
    // let num_field_secondary_ro = 2 // params_next, i_new
    // + 2 * pp.circuit_shape_secondary.F_arity // zo, z1
    // + pp.circuit_shapes.len() * (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // #num_augment

    // // Compute the primary and secondary hashes given the digest, program counter, instances, and
    // // witnesses provided by the prover
    let (hash_primary, hash_cyclefold) = {
      //   let mut hasher =
      //     <Dual<E1> as Engine>::RO::new(pp.ro_consts_secondary.clone(), num_field_primary_ro);

      //   hasher.absorb(pp.digest());
      //   hasher.absorb(E1::Scalar::from(self.num_steps as u64));
      //   hasher.absorb(self.program_counter);

      //   for e in z0_primary {
      //     hasher.absorb(*e);
      //   }

      //   for e in &self.zn_primary {
      //     hasher.absorb(*e);
      //   }

      //   self.r_U_secondary.absorb_in_ro(&mut hasher);

      //   let mut hasher2 =
      //     <E1 as Engine>::RO::new(pp.ro_consts_primary.clone(), num_field_secondary_ro);

      //   hasher2.absorb(scalar_as_base::<E1>(pp.digest()));
      //   hasher2.absorb(<Dual<E1> as Engine>::Scalar::from(self.num_steps as u64));

      //   for e in z0_secondary {
      //     hasher2.absorb(*e);
      //   }

      //   for e in &self.zn_secondary {
      //     hasher2.absorb(*e);
      //   }

      //   self.r_U_primary.iter().for_each(|U| {
      //     U.absorb_in_ro(&mut hasher2);
      //   });

      //   (
      //     hasher.squeeze(NUM_HASH_BITS),
      //     hasher2.squeeze(NUM_HASH_BITS),
      //   )
    };

    // // Compare the computed hashes with the public IO of the last invocation of `prove_step`
    // if hash_primary != self.l_u_secondary.X[0] {
    //   return Err(NovaError::ProofVerifyError.into());
    // }

    // if hash_secondary != scalar_as_base::<Dual<E1>>(self.l_u_secondary.X[1]) {
    //   return Err(NovaError::ProofVerifyError.into());
    // }

    // Verify the primary SNARK
    let res_primary = self
      .r_W_snark_primary
      .verify(&vk.vk_primary, &self.r_U_primary);

    // Verify the cyclefold SNARK
    let res_cyclefold = self
      .f_W_snark_cyclefold
      .verify(&vk.vk_cyclefold, &self.f_U_cyclefold);

    res_primary?;

    res_cyclefold?;

    Ok(self.zn_primary.clone())
  }
}
