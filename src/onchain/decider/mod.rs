//! Implements components to enable the compression-step for IVC proofs

use super::{
  decider_circuit::DeciderCircuit,
  gadgets::{FoldGadget, KZGProof},
};
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS}, errors::NovaError, frontend::groth16::{
    self, create_random_proof, generate_random_parameters, verify_proof, Parameters,
    Proof as Groth16Proof,
  }, gadgets::{nat_to_limbs, scalar_as_base, BigNat}, nebula::{
    nifs::NIFS,
    rs::{PublicParams, RecursiveSNARK},
  }, onchain::eth::ToEth, provider::{
    hyperkzg::EvaluationEngine, kzg_commitment::{KZGProverKey, KZGVerifierKey, UVKZGCommitment}, traits::DlogGroup, Bn256EngineKZG
  }, r1cs::{R1CSInstance, RelaxedR1CSInstance}, traits::{evaluation::EvaluationEngineTrait, Dual, Engine, ROConstants}, Commitment
};
use halo2curves::bn256::{Bn256, Fr};
use num_bigint::{BigInt, Sign};
use rand::RngCore;
use group::Curve;
use ff::PrimeField;
use crate::traits::commitment::CommitmentTrait;

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
    state_len: usize,
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
      state_len,
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
    let groth16_proof = create_random_proof(circuit.clone(), &pk.groth16_pk, rng)?;
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
    U_commitments: (Commitment<Bn256EngineKZG>, Commitment<Bn256EngineKZG>),
    u_commitments: Commitment<Bn256EngineKZG>,
  ) -> Result<(), NovaError> {
    let DeciderVerifierKey {
      groth16_vk,
      pp_hash,
      kzg_vk,
    } = vk;

    let prepared_groth16_vk = groth16::prepare_verifying_key(&groth16_vk);

    // 6.2. Fold the commitments
    let (U_cmW, U_cmE) = FoldGadget::fold_group_elements_native::<Bn256EngineKZG>(
      U_commitments,
      u_commitments,
      self.nifs_proof.nifs_primary.comm_T,
      self.rho,
    )?;

    // TODO: Refactor
    let (U_cmW_x, U_cmW_y, U_cmW_id) = {
      let (x, y, id) = U_cmW.to_coordinates();
      let x_bignat = BigInt::from_bytes_le(Sign::Plus, &x.to_repr());
      let x_limbs = nat_to_limbs(&x_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
      let y_bignat = BigInt::from_bytes_le(Sign::Plus, &y.to_repr());
      let y_limbs = nat_to_limbs(&y_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
      let id_fr = Fr::from(id);
      (x_limbs, y_limbs, id_fr)
    };

    let (U_cmE_x, U_cmE_y, U_cmE_id) = {
      let (x, y, id) = U_cmE.to_coordinates();
      let x_bignat = BigInt::from_bytes_le(Sign::Plus, &x.to_repr());
      let x_limbs = nat_to_limbs(&x_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
      let y_bignat = BigInt::from_bytes_le(Sign::Plus, &y.to_repr());
      let y_limbs = nat_to_limbs(&y_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
      let id_fr = Fr::from(id);
      (x_limbs, y_limbs, id_fr)
    };

    let (cmT_x, cmT_y, cmT_id) = {
      let (x, y, id) = self.nifs_proof.nifs_primary.comm_T.to_coordinates();
      let x_bignat = BigInt::from_bytes_le(Sign::Plus, &x.to_repr());
      let x_limbs = nat_to_limbs(&x_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
      let y_bignat = BigInt::from_bytes_le(Sign::Plus, &y.to_repr());
      let y_limbs = nat_to_limbs(&y_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
      let id_fr = Fr::from(id);
      (x_limbs, y_limbs, id_fr)
    };

    let public_inputs = [
      &[pp_hash],
      &[i],
      &z_0[..],
      &z_i[..],
      &U_cmW_x[..],
      &U_cmW_y[..],
      &[U_cmW_id],
      &U_cmE_x[..],
      &U_cmE_y[..],
      &[U_cmE_id],
      &[self.kzg_challenges.0, self.kzg_challenges.1],
      &[self.kzg_proofs.0.eval, self.kzg_proofs.1.eval],
      &cmT_x[..],
      &cmT_y[..],
      &[cmT_id],
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

    // let kzg_U_cmW = UVKZGCommitment::<Bn256>::new(U_cmW.comm.to_affine());
    // let kzg_U_cmE = UVKZGCommitment::<Bn256>::new(U_cmE.comm.to_affine());

    // 7.3 Verify KZG proofs
    // self.kzg_proofs.0.verify(&kzg_vk, &kzg_U_cmW, self.kzg_challenges.0)?;
    // self.kzg_proofs.1.verify(&kzg_vk, &kzg_U_cmE, self.kzg_challenges.1)?;
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
  println!("function_signature_check: {:?}", function_signature_check.to_eth());
  println!("i: {:?}", i.to_eth());
  println!("z_0: {:?}", z_0.to_eth());
  println!("z_i: {:?}", z_i.to_eth());
  println!("running_instance.comm_W: {:?}", running_instance.comm_W.to_eth());
  println!("running_instance.comm_E: {:?}", running_instance.comm_E.to_eth());
  println!("incoming_instance.comm_W: {:?}", incoming_instance.comm_W.to_eth());
  println!("proof.nifs_proof.nifs_primary.comm_T: {:?}", proof.nifs_proof.nifs_primary.comm_T.to_eth());
  println!("proof.rho: {:?}", proof.rho.to_eth());
  println!("proof.groth16_proof a: {:?}", proof.groth16_proof.a.to_eth());
  println!("proof.groth16_proof b: {:?}", proof.groth16_proof.b.to_eth());
  println!("proof.groth16_proof c: {:?}", proof.groth16_proof.c.to_eth());
  println!("proof.kzg_challenges.0: {:?}", proof.kzg_challenges.0.to_eth());
  println!("proof.kzg_challenges.1: {:?}", proof.kzg_challenges.1.to_eth());
  println!("proof.kzg_proofs.0.eval: {:?}", proof.kzg_proofs.0.eval.to_eth());
  println!("proof.kzg_proofs.1.eval: {:?}", proof.kzg_proofs.1.eval.to_eth());
  println!("proof.kzg_proofs.0.proof: {:?}", proof.kzg_proofs.0.proof.to_eth());
  println!("proof.kzg_proofs.1.proof: {:?}", proof.kzg_proofs.1.proof.to_eth());
  Ok(
    [
      function_signature_check.to_eth(),
      i.to_eth(),   // i
      z_0.to_eth(), // z_0
      z_i.to_eth(), // z_i
      running_instance.comm_W.to_eth(),
      running_instance.comm_E.to_eth(),
      incoming_instance.comm_W.to_eth(),
      proof.nifs_proof.nifs_primary.comm_T.to_eth(),  // cmT
      proof.rho.to_eth(),              // r
      proof.groth16_proof.to_eth(),    // pA, pB, pC
      proof.kzg_challenges.0.to_eth(), // challenge_W
      proof.kzg_challenges.1.to_eth(), // challenge_E
      proof.kzg_proofs.0.eval.to_eth(),  // eval W
      proof.kzg_proofs.1.eval.to_eth(),  // eval E
      proof.kzg_proofs.0.proof.to_eth(), // W kzg_proof
      proof.kzg_proofs.1.proof.to_eth(), // E kzg_proof
    ]
    .concat(),
  )
}
