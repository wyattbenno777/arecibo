//! Implements components to enable the compression-step for IVC proofs

use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_FE_IN_EMULATED_POINT, NUM_HASH_BITS}, gadgets::scalar_as_base, provider::Bn256EngineKZG, traits::{Engine, TranscriptEngineTrait}
};
use crate::{cyclefold::util::absorb_primary_relaxed_r1cs, traits::ROTrait};
use crate::{
  errors::NovaError,
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{snark::BatchedRelaxedR1CSSNARKTrait, CurveCycleEquipped, Dual},
};
use crate::{
  nebula::{
    compression::CompressedSNARK,
    ic::IC,
    nifs::PrimaryNIFS,
    traits::{Layer1PPTrait, Layer1RSTrait},
  },
  traits::ROConstants,
};
use crate::{traits::AbsorbInROTrait, Commitment};
use abomonation::Abomonation;
use ff::{Field, PrimeField};
use group::WnafGroup;
use halo2curves::bn256::{Bn256, Fr, G1};
use pairing::{MultiMillerLoop, Engine as PairingEngine};
use rand::RngCore;
use revm::interpreter::instructions::host::create;
use serde::{Deserialize, Serialize};
use bellperson::{gpu::GpuName, groth16::{create_proof, generate_random_parameters, Parameters}};
use super::decider_circuit::DeciderCircuit;

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Serialize)]
#[serde(bound = "")]
pub struct Decider<S1, S2, R>
where
  S1: BatchedRelaxedR1CSSNARKTrait<Bn256EngineKZG>,
  S2: BatchedRelaxedR1CSSNARKTrait<Dual<Bn256EngineKZG>>,
  R: RngCore {}

impl<S1, S2, R> Decider<S1, S2, R>
where
  S1: BatchedRelaxedR1CSSNARKTrait<Bn256EngineKZG>,
  S2: BatchedRelaxedR1CSSNARKTrait<Dual<Bn256EngineKZG>>,
  R: RngCore  
{
  /// Creates prover and verifier keys for [`Decider`]
  pub fn setup(
    pp: &impl Layer1PPTrait<Bn256EngineKZG>,
    rng: &mut R,
  ) -> Result<Parameters<Bn256>, NovaError> 
    {
    let (nebula_pk_primary, nebula_vk_primary) =
      S1::setup(pp.biggest_ck().clone(), pp.primary_r1cs_shapes())?;

    let (nebula_pk_secondary, nebula_vk_secondary) =
      S2::setup(pp.ck_secondary().clone(), pp.secondary_r1cs_shapes())?;

    let nebula_pk = ProverKey {
      primary: nebula_pk_primary,
      secondary: nebula_pk_secondary,
    };
    let nebula_vk = VerifierKey {
      primary: nebula_vk_primary,
      secondary: nebula_vk_secondary,
    };

    let circuit = DeciderCircuit::<Bn256EngineKZG>::default(
      pp.primary_r1cs_shapes()[0],
      pp.secondary_r1cs_shapes()[0],
      ROConstants::<Bn256EngineKZG>::default(),
      pp.F().digest(),
      2, // TODO: Set correct value
      // TODO: Set correct value
      2, // Nebula's running CommittedInstance contains 2 commitments
      (&*pp.F().ck_primary, &*pp.ck_secondary()),
    );

    // get the Groth16 specific setup for the circuit
    let params = generate_random_parameters::<Bn256, _, _>(circuit, &mut rng).unwrap();

    Ok(params)
  }

  /// Create a new [`CompressedSNARK`]
  pub fn prove(
    pp: &impl Layer1PPTrait<E>,
    pk: &ProverKey<E, S1, S2>,
    rs: &impl Layer1RSTrait<E>,
  ) -> Result<Self, NovaError> {
    let circuit = DeciderCircuit::<E>::new(pp.F(), rs.F().clone())?;
    let rho = circuit.randomness;
    let kzg_challenges = circuit.kzg_challenges;
    let kzg_proofs = circuit
      .W_i1
      .W
      .iter()
      .zip(&kzg_challenges)
      .map(|(v, c)| unimplemented!())
      .collect::<Result<Vec<_>, _>>()?;

    let snark_proof = CompressedSNARK::prove(&pp, &pk, &recursive_snark);
    Ok(Self {})
  }

  /// Verify the correctness of the [`CompressedSNARK`]
  pub fn verify(
    &self,
    pp: &impl Layer1PPTrait<E>,
    vk: &VerifierKey<E, S1, S2>,
  ) -> Result<(), NovaError> {
    Ok(())
  }
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


