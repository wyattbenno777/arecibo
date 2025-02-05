//! Implements components to enable the compression-step for IVC proofs

use crate::{
  frontend::groth16::{self}, 
  provider::{hyperkzg::EvaluationEngine, kzg_commitment::{KZGCommitmentEngine, KZGProof, KZGProverKey, KZGVerifierKey}, Bn256EngineKZG}, 
  traits::Engine
};
use crate::
  errors::NovaError;
use crate::{
  nebula::
    traits::{Layer1PPTrait, Layer1RSTrait},
  traits::ROConstants,
};
use halo2curves::bn256::{Bn256, Fr};
use rand::RngCore;
use serde::Serialize;
use crate::frontend::groth16::{create_random_proof, generate_random_parameters, Parameters};
use super::decider_circuit::DeciderCircuit;
use crate::traits::evaluation::EvaluationEngineTrait;
// use crate::traits::commitment::CommitmentEngineTrait;

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Clone)]
pub struct ProverKey {
  groth16_pk: Parameters<Bn256EngineKZG>,
  kzg_pk: KZGProverKey<Bn256>,
}

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug, Clone)]
pub struct VerifierKey {
  groth16_vk: groth16::VerifyingKey<Bn256EngineKZG>,
  pp_hash: <Bn256EngineKZG as Engine>::Scalar,
  kzg_vk: KZGVerifierKey<Bn256>,
}

/// A SNARK that proves the knowledge of a valid Nebula proof
#[derive(Debug)]
pub struct Decider {
    groth16_proof: groth16::Proof<Bn256EngineKZG>,
    rho: Fr,
    kzg_challenges: Vec<Fr>,
    kzg_proofs: Vec<KZGProof<Bn256>>,
  }

impl Decider {
  /// Creates prover and verifier keys for [`Decider`]
  pub fn setup<R>(
    pp: &impl Layer1PPTrait<Bn256EngineKZG>,
    rng: &mut R,
  ) -> Result<(ProverKey, VerifierKey), NovaError> 
  where
    R: RngCore,
    {
    let pp_hash = pp.F().digest();
    let circuit = DeciderCircuit::<Bn256EngineKZG>::default(
      pp.primary_r1cs_shapes()[0],
      pp.secondary_r1cs_shapes()[0],
      ROConstants::<Bn256EngineKZG>::default(),
      pp_hash,
      2, // TODO: Set correct value
      // TODO: Set correct value
      2, // Nebula's running CommittedInstance contains 2 commitments
      (&*pp.F().ck_primary, &*pp.ck_secondary()),
    );

    let (kzg_pk, kzg_vk) =
    EvaluationEngine::<Bn256, Bn256EngineKZG>::setup(pp.F().ck_primary.clone());
    // get the Groth16 specific setup for the circuit
    let params = generate_random_parameters::<Bn256EngineKZG, _, _>(circuit, rng).unwrap();

    let pk = ProverKey {
      groth16_pk: params.clone(),
      kzg_pk,
    };
    let vk = VerifierKey {
      groth16_vk: params.vk,
      pp_hash,
      kzg_vk,
    };

    Ok((pk, vk))
  }

  /// Create a new [`CompressedSNARK`]
  pub fn prove<R>(
    pp: &impl Layer1PPTrait<Bn256EngineKZG>,
    pk: &ProverKey,
    rs: &impl Layer1RSTrait<Bn256EngineKZG>,
    rng: &mut R,
  ) -> Result<Self, NovaError> 
  where R: RngCore {
    let circuit = DeciderCircuit::<Bn256EngineKZG>::new(pp.F(), rs.F().clone())?;
    // TODO: Do we need D::Proof?
    let rho = circuit.randomness;
    let kzg_challenges = circuit.kzg_challenges.clone();

    let kzg_proofs = kzg_challenges.iter()
      .map(|c| {          
        KZGCommitmentEngine::prove_with_challenge(
            &pk.kzg_pk,
            *c,
            &circuit.W_i1.W[..]
          )
      })
      .collect::<Result<Vec<_>, _>>()?;

    let groth16_proof = create_random_proof(circuit, &pk.groth16_pk, rng)?;
    Ok(Self {
      groth16_proof,
      rho,
      kzg_challenges,
      kzg_proofs,
    })
  }

  // / Verify the correctness of the [`CompressedSNARK`]
  // pub fn verify(
  //   &self,
  //   pp: &impl Layer1PPTrait<E>,
  //   vk: &VerifierKey<E, S1, S2>,
  // ) -> Result<(), NovaError> {
  //   Ok(())
  // }
}

// /// Prepares solidity calldata for calling the NovaDecider contract
// #[allow(clippy::too_many_arguments)]
// pub fn prepare_calldata(
//   function_signature_check: [u8; 4],
//   i: Fr,
//   z_0: Vec<Fr>,
//   z_i: Vec<Fr>,
//   running_instance: &CommittedInstance<G1>,
//   incoming_instance: &CommittedInstance<G1>,
//   proof: Proof<G1, KZG<'static, Bn254>, Groth16<Bn254>>,
// ) -> Result<Vec<u8>, Error> {
//   Ok(
//     [
//       function_signature_check.to_eth(),
//       i.to_eth(),   // i
//       z_0.to_eth(), // z_0
//       z_i.to_eth(), // z_i
//       running_instance.cmW.to_eth(),
//       running_instance.cmE.to_eth(),
//       incoming_instance.cmW.to_eth(),
//       proof.cmT.to_eth(),                 // cmT
//       proof.r.to_eth(),                   // r
//       proof.snark_proof.to_eth(),         // pA, pB, pC
//       proof.kzg_challenges.to_eth(),      // challenge_W, challenge_E
//       proof.kzg_proofs[0].eval.to_eth(),  // eval W
//       proof.kzg_proofs[1].eval.to_eth(),  // eval E
//       proof.kzg_proofs[0].proof.to_eth(), // W kzg_proof
//       proof.kzg_proofs[1].proof.to_eth(), // E kzg_proof
//     ]
//     .concat(),
//   )
// }


