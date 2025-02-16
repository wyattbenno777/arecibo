//! Implements components to enable the compression-step for IVC proofs

use super::{
  decider_circuit::DeciderCircuit,
  gadgets::{FoldGadget, KZGProof},
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
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{evaluation::EvaluationEngineTrait, Engine, ROConstants},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::RngCore;

pub mod test;


/// A type that holds the prover key for [`Decider`]
#[derive(Clone)]
pub struct DeciderProverKey {
  pub groth16_pk: Parameters<Bn256EngineKZG>,
  pub kzg_pk: KZGProverKey<Bn256>,
}

/// A type that holds the verifier key for [`Decider`]
#[derive(Debug, Clone)]
pub struct DeciderVerifierKey {
  pub groth16_vk: groth16::VerifyingKey<Bn256EngineKZG>,
  pub pp_hash: <Bn256EngineKZG as Engine>::Scalar,
  pub kzg_vk: KZGVerifierKey<Bn256>,
}

/// A SNARK that proves the knowledge of a valid  proof
#[derive(Debug)]
pub struct Decider {
  groth16_proof: Groth16Proof<Bn256EngineKZG>,
  rho: Fr,
  kzg_challenges: (Fr, Fr),
  kzg_proofs: (KZGProof<Bn256>, KZGProof<Bn256>),
  nifs_proof: NIFS<Bn256EngineKZG>,
}

impl Decider {
  /// Creates prover and verifier keys for [`Decider`]
  pub fn setup<R>(
    pp: &PublicParams<Bn256EngineKZG>,
    rng: &mut R,
  ) -> Result<(DeciderProverKey, DeciderVerifierKey), NovaError>
  where
    R: RngCore,
  {
    let pp_hash = pp.digest();
    let circuit = DeciderCircuit::<Bn256EngineKZG>::default(
      &pp.circuit_shape_primary.r1cs_shape,
      &pp.circuit_shape_cyclefold.r1cs_shape,
      ROConstants::<Bn256EngineKZG>::default(),
      pp_hash,
      1, // TODO: Parameterize
      (&*pp.ck_primary, &*pp.ck_cyclefold),
    );

    let (kzg_pk, kzg_vk) = EvaluationEngine::<Bn256, Bn256EngineKZG>::setup(pp.ck_primary.clone());

    // get the Groth16 specific setup for the circuit
    let params = generate_random_parameters::<Bn256EngineKZG, _, _>(circuit, rng).unwrap();

    let pk = DeciderProverKey {
      groth16_pk: params.clone(),
      kzg_pk,
    };
    let vk = DeciderVerifierKey {
      groth16_vk: params.vk,
      pp_hash,
      kzg_vk,
    };

    Ok((pk, vk))
  }

  /// Create a new [`CompressedSNARK`]
  pub fn prove<R>(
    pp: &PublicParams<Bn256EngineKZG>,
    pk: &DeciderProverKey,
    rs: &RecursiveSNARK<Bn256EngineKZG>,
    rng: &mut R,
  ) -> Result<Self, NovaError>
  where
    R: RngCore,
  {
    let circuit = DeciderCircuit::<Bn256EngineKZG>::new(pp, rs.clone())?;
    let rho = circuit.randomness;
    let nifs_proof = circuit.nifs_proof.clone();
    let kzg_challenges = circuit.kzg_challenges.clone();

    let kzg_proofs = (
      KZGProof::prove(&pk.kzg_pk, kzg_challenges.0, &circuit.W_i1.W[..])?,
      KZGProof::prove(&pk.kzg_pk, kzg_challenges.1, &circuit.W_i1.E[..])?,
    );
    let groth16_proof = create_random_proof(circuit, &pk.groth16_pk, rng)?;
    Ok(Self {
      groth16_proof,
      rho,
      kzg_challenges,
      kzg_proofs,
      nifs_proof,
    })
  }

  /// Verify the correctness of the [`CompressedSNARK`]
  pub fn verify(
    &self,
    vk: DeciderVerifierKey,
    i: Fr,
    z_0: Vec<Fr>,
    z_i: Vec<Fr>,
    U_commitments: (UVKZGCommitment<Bn256>, UVKZGCommitment<Bn256>),
    u_commitments: UVKZGCommitment<Bn256>,
  ) -> Result<(), NovaError> {
    let DeciderVerifierKey {
      groth16_vk,
      pp_hash,
      kzg_vk,
    } = vk;

    let prepared_groth16_vk = groth16::prepare_verifying_key(&groth16_vk);

    // 6.2. Fold the commitments
    let (U_cmW, U_cmE) = FoldGadget::fold_group_elements_native::<Bn256>(
      U_commitments,
      u_commitments,
      self.nifs_proof.nifs_primary.comm_T.comm,
      self.rho,
    )?;

    let public_inputs = [
      &[pp_hash],
      &[i],
      &z_0[..],
      &z_i[..],
      // TODO: Pass the U commitments as inputs
      &[self.kzg_challenges.0, self.kzg_challenges.1],
      &[self.kzg_proofs.0.eval, self.kzg_proofs.1.eval],
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
    // 7.3 Verify KZG proofs
    self.kzg_proofs.0.verify(&kzg_vk, &U_cmW, self.kzg_challenges.0)?;
    self.kzg_proofs.1.verify(&kzg_vk, &U_cmE, self.kzg_challenges.1)?;
    Ok(())
  }
}

/// Prepares solidity calldata for calling the NovaDecider contract
#[allow(clippy::too_many_arguments)]
pub fn prepare_calldata(
  function_signature_check: [u8; 4],
  i: Fr,
  z_0: &Vec<Fr>,
  z_i: &Vec<Fr>,
  running_instance: &RelaxedR1CSInstance<Bn256EngineKZG>,
  incoming_instance: &R1CSInstance<Bn256EngineKZG>,
  proof: &Decider,
) -> Result<Vec<u8>, NovaError> {
  Ok(
    [
      function_signature_check.to_eth(),
      i.to_eth(),   // i
      z_0.to_eth(), // z_0
      z_i.to_eth(), // z_i
      running_instance.comm_W.to_eth(),
      running_instance.comm_E.to_eth(),
      incoming_instance.comm_W.to_eth(),
      proof.nifs_proof.nifs_primary.comm_T.to_eth(),                 // cmT
      proof.rho.to_eth(),              // r
      proof.groth16_proof.to_eth(),    // pA, pB, pC
      proof.kzg_challenges.0.to_eth(), // challenge_W, challenge_E
      proof.kzg_challenges.1.to_eth(), // challenge_W, challenge_E
      proof.kzg_proofs.0.eval.to_eth(),  // eval W
      proof.kzg_proofs.1.eval.to_eth(),  // eval E
      proof.kzg_proofs.0.proof.to_eth(), // W kzg_proof
      proof.kzg_proofs.1.proof.to_eth(), // E kzg_proof
    ]
    .concat(),
  )
}
