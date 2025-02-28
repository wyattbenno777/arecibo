use crate::{
  constants::NUM_HASH_BITS,
  cyclefold::{
    gadgets::emulated::AllocatedEmulRelaxedR1CSInstance, util::absorb_primary_commitment,
  },
  errors::PCSError,
  frontend::{
    domain::EvaluationDomain, gpu::GpuName, groth16::aggregate::poly::DensePolynomial,
    num::AllocatedNum, ConstraintSystem, SynthesisError,
  },
  gadgets::{le_bits_to_num, scalar_as_base},
  provider::{kzg_commitment::{KZGProverKey, KZGVerifierKey, UVKZGCommitment},  traits::DlogGroup},
  r1cs::RelaxedR1CSInstance,
  traits::{
    CurveCycleEquipped, Dual, Engine, ROCircuitTrait, ROConstants, ROConstantsCircuit, ROTrait,
  }, 
};
use ec_gpu_gen::threadpool::Worker;
use group::Curve;
use pairing::Engine as PairingEngine;
use serde::{Deserialize, Serialize};

/// Gadget that computes the KZG challenges.
/// It also offers the rust native implementation compatible with the gadget.
pub struct KZGChallengesGadget {}

impl KZGChallengesGadget {
  /// Compute the KZG challenges natively.
  pub fn get_challenges_native<E: CurveCycleEquipped>(
    U_i: RelaxedR1CSInstance<E>,
  ) -> (E::Scalar, E::Scalar) {
    let ro_consts = ROConstants::<Dual<E>>::default();
    let mut ro: <Dual<E> as Engine>::RO = <Dual<E> as Engine>::RO::new(ro_consts.clone(), 9);
    absorb_primary_commitment::<E, Dual<E>>(&U_i.comm_W, &mut ro);
    let rw = ro.squeeze(NUM_HASH_BITS);
    let mut ro: <Dual<E> as Engine>::RO = <Dual<E> as Engine>::RO::new(ro_consts.clone(), 9);
    absorb_primary_commitment::<E, Dual<E>>(&U_i.comm_E, &mut ro);
    let re = ro.squeeze(NUM_HASH_BITS);
    let rw = scalar_as_base::<Dual<E>>(rw);
    let re = scalar_as_base::<Dual<E>>(re);
    (rw, re)
  }

  /// Compute the KZG challenges in-circuit.
  pub fn get_challenges_gadget<CS, E: CurveCycleEquipped>(
    cs: &mut CS,
    U_i: AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
  ) -> Result<(AllocatedNum<E::Scalar>, AllocatedNum<E::Scalar>), SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ROConstantsCircuit::<Dual<E>>::default(), 9);

    U_i
      .comm_W
      .absorb_in_ro(cs.namespace(|| "absorb_W"), &mut ro)?;
    let rw = ro.squeeze(cs.namespace(|| "squeeze_W"), NUM_HASH_BITS)?;
    let alloc_rw = le_bits_to_num(cs.namespace(|| "bits_to_num rw"), &rw)?;

    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ROConstantsCircuit::<Dual<E>>::default(), 9);

    U_i
      .comm_E
      .absorb_in_ro(cs.namespace(|| "absorb_E"), &mut ro)?;
    let re = ro.squeeze(cs.namespace(|| "squeeze_E"), NUM_HASH_BITS)?;
    let alloc_re = le_bits_to_num(cs.namespace(|| "bits_to_num re"), &re)?;

    Ok((alloc_rw, alloc_re))
  }
}

/// A KZG proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KZGProof<E: PairingEngine> {
  /// The proof.
  pub proof: E::G1,
  /// The evaluation.
  pub eval: E::Fr,
}

impl<E: PairingEngine> KZGProof<E> {
  /// Create a KZG proof.
  pub fn prove(
    pk: &KZGProverKey<E>,
    challenge: E::Fr,
    v: &[E::Fr],
  ) -> Result<KZGProof<E>, PCSError>
  where
    E::G1: DlogGroup<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
    E::Fr: GpuName,
  {
    let mut domain =
      EvaluationDomain::from_coeffs(v.to_vec()).expect("Failed to creat
e evaluation domain");

    let worker = Worker::new();
    domain.ifft(&worker, &mut None).expect("FFT failed");

    let polynomial = DensePolynomial::from_coeffs(domain.into_coeffs())
;
    if polynomial.degree() >= pk.powers_of_g().len() {
      return Err(PCSError::LengthError);
    }

    let divisor = DensePolynomial::from_coeffs(vec![-challenge, E::Fr::from(1)]);
    let (witness_poly, remainder_poly) = polynomial.quot_rem(&divisor);

    let eval = if remainder_poly.is_zero() {
      E::Fr::from(0)
    } else {
      remainder_poly.coeffs()[0]
    };

    let proof = E::G1::vartime_multiscalar_mul(
      witness_poly.coeffs(),
      &pk.powers_of_g()[..witness_poly.coeffs().len()],
    );

    Ok(KZGProof { proof, eval })
  }

  /// Verify a KZG proof.
  pub fn verify(
    &self,
    vk: &KZGVerifierKey<E>,
    commitment: &UVKZGCommitment<E>,
    challenge: E::Fr,
  ) -> Result<(), PCSError> {
    // Verify that `proof.eval` is the evaluation at `challenge` of the polynomial committed inside `commitment`.
    let cm = E::G1::from(commitment.0);
    let inner = cm - vk.g * self.eval;
    let lhs = E::pairing(&inner.to_affine(), &vk.h);
    let inner = E::G2::from(vk.beta_h) - vk.h * challenge;
    let rhs = E::pairing(&self.proof.to_affine(), &inner.to_affine());
    if lhs != rhs {
      return Err(PCSError::InvalidPCS);
    }
    Ok(())
  }
}

