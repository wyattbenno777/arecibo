//! Implements components to enable the compression-step for IVC proofs
pub mod circuit;

use crate::onchain::{
  compressed::circuit::VerifierCircuit,
  gadgets::{FoldGadget, KZGProof}, utils::to_scalar_coordinates,
};
use crate::{
  errors::NovaError,
  frontend::groth16::{
    self, create_random_proof, generate_random_parameters, verify_proof, Parameters,
    Proof as Groth16Proof,
  },
  nebula::{
    nifs::NIFS,
    rs::{PublicParams, RecursiveSNARK},
  },
  onchain::eth::ToEth,
  provider::{
    hyperkzg::EvaluationEngine,
    kzg_commitment::{KZGProverKey, KZGVerifierKey, UVKZGCommitment},
    Bn256EngineKZG,
  },
  traits::{
    evaluation::EvaluationEngineTrait, Engine, ROConstants,
  },
  Commitment,
};
use group::Curve;
use halo2curves::bn256::{Bn256, Fr};
use rand::RngCore;
use serde::{Deserialize, Serialize};


/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Clone, Serialize, Deserialize)]
pub struct CompressedPK {
  /// Groth16 proving key
  pub groth16_pk: Parameters<Bn256EngineKZG>,
  /// KZG proving key
  pub kzg_pk: KZGProverKey<Bn256>,
}

/// A type that holds the verifier key for [`CompressedSNARK`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVK {
  /// Groth16 verifying key
  pub groth16_vk: groth16::VerifyingKey<Bn256EngineKZG>,
  /// Public parameters hash
  pub pp_hash: <Bn256EngineKZG as Engine>::Scalar,
  /// KZG verifying key
  pub kzg_vk: KZGVerifierKey<Bn256>,
}



/// A SNARK that proves the knowledge of a valid  proof
#[derive(Debug, Deserialize, Serialize)]
pub struct CompressedSNARK {
  /// Groth16 proof
  pub groth16_proof: Groth16Proof<Bn256EngineKZG>,
  /// Randomness
  pub rho: Fr,
  /// KZG challenges
  pub kzg_challenges: (Fr, Fr),
  /// KZG proofs
  pub kzg_proofs: (KZGProof<Bn256>, KZGProof<Bn256>),
  /// NIFS proof
  pub nifs_proof: NIFS<Bn256EngineKZG>,
  /// Commitment of the witness of incoming instance
  pub u_cmW: Commitment<Bn256EngineKZG>,
  /// Commitment of the witness of current relaxed instance
  pub U_cmW: Commitment<Bn256EngineKZG>,
  /// Commitment of the error of current relaxed instance
  pub U_cmE: Commitment<Bn256EngineKZG>,
  /// Number of steps
  pub num_steps: Fr,
  /// Initial state
  pub z_0: Vec<Fr>,
  /// Current state
  pub z_i: Vec<Fr>,
}

impl CompressedSNARK {
  /// Creates prover and verifier keys for [`Decider`]
  pub fn setup<R>(
    pp: &PublicParams<Bn256EngineKZG>,
    rng: &mut R,
    state_len: usize,
  ) -> Result<(CompressedPK, CompressedVK), NovaError>
  where
    R: RngCore,
  {
    let pp_hash = pp.digest();
    let circuit = VerifierCircuit::<Bn256EngineKZG>::default(
      &pp.circuit_shape_primary.r1cs_shape,
      &pp.circuit_shape_cyclefold.r1cs_shape,
      ROConstants::<Bn256EngineKZG>::default(),
      pp_hash,
      state_len,
      (&*pp.ck_primary, &*pp.ck_cyclefold),
    );

    let (kzg_pk, kzg_vk) = EvaluationEngine::<Bn256, Bn256EngineKZG>::setup(pp.ck_primary.clone());

    // get the Groth16 specific setup for the circuit
    let params = generate_random_parameters::<Bn256EngineKZG, _, _>(circuit, rng).unwrap();

    let pk = CompressedPK {
      // TODO: Remove vk from pk (optimisation)
      groth16_pk: params.clone(),
      kzg_pk,
    };
    let vk = CompressedVK {
      groth16_vk: params.vk,
      pp_hash,
      kzg_vk,
    };

    Ok((pk, vk))
  }

  /// Create a new [`CompressedSNARK`]
  pub fn prove<R>(
    pp: &PublicParams<Bn256EngineKZG>,
    pk: &CompressedPK,
    rs: &RecursiveSNARK<Bn256EngineKZG>,
    rng: &mut R,
  ) -> Result<Self, NovaError>
  where
    R: RngCore,
  {
    let circuit = VerifierCircuit::<Bn256EngineKZG>::new(pp, rs.clone())?;
    let rho = circuit.randomness;
    let nifs_proof = circuit.nifs_proof.clone();
    let kzg_challenges = circuit.kzg_challenges;

    let kzg_proofs = (
      KZGProof::prove(&pk.kzg_pk, kzg_challenges.0, &circuit.W_i1.W[..])?,
      KZGProof::prove(&pk.kzg_pk, kzg_challenges.1, &circuit.W_i1.E[..])?,
    );
    let groth16_proof = create_random_proof(circuit.clone(), &pk.groth16_pk, rng)?;
    Ok(Self {
      groth16_proof,
      rho,
      kzg_challenges,
      kzg_proofs,
      nifs_proof,
      U_cmW: rs.r_U_primary.comm_W,
      U_cmE: rs.r_U_primary.comm_E,
      u_cmW: rs.l_u_primary.comm_W,
      z_0: rs.z0.clone(),
      z_i: rs.zi.clone(),
      num_steps: Fr::from(rs.i as u64),
    })
  }

  /// Verify the correctness of the [`CompressedSNARK`]
  pub fn verify(
    &self,
    vk: CompressedVK,
  ) -> Result<(), NovaError> {
    let CompressedVK {
      groth16_vk,
      pp_hash,
      kzg_vk,
    } = vk;

    let prepared_groth16_vk = groth16::prepare_verifying_key(&groth16_vk);

    // 6.2. Fold the commitments
    let (U_cmW, U_cmE) = FoldGadget::fold_group_elements_native::<Bn256EngineKZG>(
      self.U_cmW,
      self.U_cmE,
      self.u_cmW,
      self.nifs_proof.nifs_primary.comm_T,
      self.rho,
    )?;

    let (U_cmW_x, U_cmW_y, _U_cmW_id) = to_scalar_coordinates(&U_cmW)?;
    let (U_cmE_x, U_cmE_y, _U_cmE_id) = to_scalar_coordinates(&U_cmE)?;
    let (cmT_x, cmT_y, _cmT_id) = to_scalar_coordinates(&self.nifs_proof.nifs_primary.comm_T)?;

    let public_inputs = [
      &[pp_hash],
      &[self.num_steps],
      &self.z_0[..],
      &self.z_i[..],
      &U_cmW_x[..],
      &U_cmW_y[..],
      &U_cmE_x[..],
      &U_cmE_y[..],
      &[self.kzg_challenges.0],
      &[self.kzg_challenges.1],
      &[self.kzg_proofs.0.eval],
      &[self.kzg_proofs.1.eval],
      &cmT_x[..],
      &cmT_y[..],
    ]
    .concat();

    let snark_v = verify_proof(
      &prepared_groth16_vk,
      &self.groth16_proof,
      &public_inputs[..],
    )?;

    if !snark_v {
      return Err(NovaError::ProofVerifyError);
    }

    let kzg_U_cmW = UVKZGCommitment::<Bn256>::new(U_cmW.comm.to_affine());
    let kzg_U_cmE = UVKZGCommitment::<Bn256>::new(U_cmE.comm.to_affine());

    // 7.3 Verify KZG proofs
    self
      .kzg_proofs
      .0
      .verify(&kzg_vk, &kzg_U_cmW, self.kzg_challenges.0)?;
    self
      .kzg_proofs
      .1
      .verify(&kzg_vk, &kzg_U_cmE, self.kzg_challenges.1)?;
    Ok(())
  }
}

/// Prepares solidity calldata for calling the NovaDecider contract
#[allow(clippy::too_many_arguments)]
pub fn prepare_calldata(
  function_signature_check: [u8; 4],
  proof: &CompressedSNARK,
) -> Result<Vec<u8>, NovaError> {
  Ok(
    [
      function_signature_check.to_eth(),
      proof.num_steps.to_eth(),   // i
      proof.z_0.to_eth(), // z_0
      proof.z_i.to_eth(), // z_i
      proof.U_cmW.to_eth(),
      proof.U_cmE.to_eth(),
      proof.u_cmW.to_eth(),
      proof.nifs_proof.nifs_primary.comm_T.to_eth(), // cmT
      proof.rho.to_eth(),                            // r
      proof.groth16_proof.to_eth(),                  // pA, pB, pC
      proof.kzg_challenges.0.to_eth(),               // challenge_W
      proof.kzg_challenges.1.to_eth(),               // challenge_E
      proof.kzg_proofs.0.eval.to_eth(),              // eval W
      proof.kzg_proofs.1.eval.to_eth(),              // eval E
      proof.kzg_proofs.0.proof.to_eth(),             // W kzg_proof
      proof.kzg_proofs.1.proof.to_eth(),             // E kzg_proof
    ]
    .concat(),
  )
}
