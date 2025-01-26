use ff::Field;
use ff::PrimeField;
use group::{
  // prime::PrimeCurveAffine,
  Curve,
};
use halo2curves::bn256::G2Prepared;
use halo2curves::bn256::{Bn256, G1Affine, G2Affine, Gt};
use pairing::Engine;
use pairing::{MillerLoopResult, MultiMillerLoop};
use rand_core::RngCore;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};

/// A verification key in the Groth16 SNARK.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VerifyingKey {
  pub alpha_g1: G1Affine,
  pub beta_g2: G2Affine,
  pub gamma_g2: G2Affine,
  pub delta_g2: G2Affine,
  pub gamma_abc_g1: Vec<G1Affine>,
}

/// The prover key for the Groth16 zkSNARK.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProvingKey {
  pub vk: VerifyingKey,
  pub beta_g1: G1Affine,
  pub delta_g1: G1Affine,
  pub a_query: Vec<G1Affine>,
  pub b_g1_query: Vec<G1Affine>,
  pub b_g2_query: Vec<G2Affine>,
  pub h_query: Vec<G1Affine>,
  pub l_query: Vec<G1Affine>,
  pub _dummy: G2Affine,
}

/// A Groth16 proof.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Proof {
  pub a: G1Affine,
  pub b: G2Affine,
  pub c: G1Affine,
}

/// Generates a circuit-specific Groth16 setup that produces the ProvingKey and VerifyingKey.
pub fn generate_parameters<R: RngCore>(
  rng: &mut R,
  num_public_inputs: usize,
) -> (ProvingKey, VerifyingKey) {
  let alpha_scalar = <Bn256 as Engine>::Fr::random(&mut *rng);
  let beta_scalar = <Bn256 as Engine>::Fr::random(&mut *rng);
  let gamma_scalar = <Bn256 as Engine>::Fr::random(&mut *rng);
  let delta_scalar = <Bn256 as Engine>::Fr::random(&mut *rng);

  let alpha_g1 = G1Affine::generator().mul(alpha_scalar).to_affine();
  let beta_g2 = G2Affine::generator().mul(beta_scalar).to_affine();
  let gamma_g2 = G2Affine::generator().mul(gamma_scalar).to_affine();
  let delta_g2 = G2Affine::generator().mul(delta_scalar).to_affine();

  let mut gamma_abc_g1 = Vec::with_capacity(num_public_inputs + 1);
  for _ in 0..(num_public_inputs + 1) {
    let scalar = <Bn256 as Engine>::Fr::random(&mut *rng);
    let g = G1Affine::generator().mul(scalar).to_affine();
    gamma_abc_g1.push(g);
  }

  let vk = VerifyingKey {
    alpha_g1,
    beta_g2,
    gamma_g2,
    delta_g2,
    gamma_abc_g1,
  };

  let beta_g1 = G1Affine::generator().mul(beta_scalar).to_affine();
  let delta_g1 = G1Affine::generator().mul(delta_scalar).to_affine();

  let mut a_query = vec![];
  let mut b_g1_query = vec![];
  let mut b_g2_query = vec![];
  let mut h_query = vec![];
  let mut l_query = vec![];

  for _ in 0..4 {
    let rand_scalar = <Bn256 as Engine>::Fr::random(&mut *rng);

    let g1 = G1Affine::generator().mul(rand_scalar).to_affine();
    a_query.push(g1);

    let g1b = G1Affine::generator().mul(rand_scalar).to_affine();
    b_g1_query.push(g1b);

    let g2b = G2Affine::generator().mul(rand_scalar).to_affine();
    b_g2_query.push(g2b);

    let g1h = G1Affine::generator().mul(rand_scalar).to_affine();
    h_query.push(g1h);

    let g1l = G1Affine::generator().mul(rand_scalar).to_affine();
    l_query.push(g1l);
  }

  let dummy_scalar = <Bn256 as Engine>::Fr::random(&mut *rng);
  let dummy_g2 = G2Affine::generator().mul(dummy_scalar).to_affine();

  let pk = ProvingKey {
    vk: vk.clone(),
    beta_g1,
    delta_g1,
    a_query,
    b_g1_query,
    b_g2_query,
    h_query,
    l_query,
    _dummy: dummy_g2,
  };

  (pk, vk)
}

/// Generates a Groth16 proof given a ProvingKey, witness data, and an R1CS instance.
pub fn create_proof<R: RngCore>(pk: &ProvingKey, _constraints: &[()], rng: &mut R) -> Proof {
  let mut random_scalar = <Bn256 as Engine>::Fr::random(&mut *rng);

  let ephemeral_a = pk.vk.alpha_g1.mul(random_scalar).to_affine();

  let ephemeral_b = pk.vk.beta_g2.mul(random_scalar).to_affine();

  random_scalar = <Bn256 as Engine>::Fr::random(&mut *rng);
  let ephemeral_c = pk.vk.gamma_abc_g1[0].mul(random_scalar).to_affine();

  Proof {
    a: ephemeral_a,
    b: ephemeral_b,
    c: ephemeral_c,
  }
}

/// Verifies a Groth16 proof.
pub fn verify_proof(
  vk: &VerifyingKey,
  public_inputs: &[<Bn256 as Engine>::Fr],
  proof: &Proof,
) -> bool {
  if public_inputs.len() + 1 != vk.gamma_abc_g1.len() {
    return false;
  }

  let mut acc = vk.gamma_abc_g1[0].mul(public_inputs[0]).to_affine();
  for (val, g) in public_inputs.iter().zip(vk.gamma_abc_g1.iter().skip(1)) {
    let tmp = g.mul(val).to_affine();
    acc = acc.add(&tmp).to_affine();
  }

  let pairing_inputs = [
    (&proof.a, &G2Prepared::from(proof.b)),
    (&vk.alpha_g1, &G2Prepared::from(vk.beta_g2)),
    (&acc, &G2Prepared::from(vk.gamma_g2)),
    (&proof.c, &G2Prepared::from(vk.delta_g2)),
  ];

  // TODO: Check if this is correct
  let result = Bn256::multi_miller_loop(&pairing_inputs).final_exponentiation();
  result == Gt::identity()
}
