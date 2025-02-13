#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use crate::constants::NUM_HASH_BITS;
use crate::cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance;
use crate::cyclefold::util::absorb_primary_commitment;
use crate::errors::PCSError;
use crate::frontend::domain::EvaluationDomain;
use crate::frontend::gpu::GpuName;
use crate::frontend::groth16::aggregate::poly::DensePolynomial;
use crate::frontend::num::AllocatedNum;
use crate::frontend::{ConstraintSystem, SynthesisError};
use crate::gadgets::{le_bits_to_num, scalar_as_base};
use crate::provider::kzg_commitment::KZGProverKey;
use crate::provider::traits::DlogGroup;
use crate::r1cs::RelaxedR1CSInstance;
use crate::traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROCircuitTrait, ROConstants, ROConstantsCircuit, ROTrait};
use crate::Commitment;
use ec_gpu_gen::threadpool::Worker;
use ff::PrimeField;
use domain::{AllocatedEvaluations, AllocatedRadix2Domain};
use pairing::Engine as PairingEngine;
use serde::{Deserialize, Serialize};
use super::utils::nth_root_of_unity;

pub mod domain;

/// Gadget that computes the KZG challenges.
/// It also offers the rust native implementation compatible with the gadget.
pub struct KZGChallengesGadget {}

impl KZGChallengesGadget {
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

  pub fn get_challenges_gadget<CS, E: CurveCycleEquipped>(
    cs: &mut CS,
    U_i: AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
  ) -> Result<(AllocatedNum<E::Scalar>, AllocatedNum<E::Scalar>), SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      ROConstantsCircuit::<Dual<E>>::default(),
      9,
    );

    U_i.comm_W.absorb_in_ro(cs.namespace(|| "absorb_W"), &mut ro)?;
    let rw = ro.squeeze(cs.namespace(|| "squeeze_W"), NUM_HASH_BITS)?;
    let alloc_rw = le_bits_to_num(cs.namespace(|| "bits_to_num"), &rw)?;

    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      ROConstantsCircuit::<Dual<E>>::default(),
      9,
    );

    U_i.comm_E.absorb_in_ro(cs.namespace(|| "absorb_E"), &mut ro)?;
    let re = ro.squeeze(cs.namespace(|| "squeeze_E"), NUM_HASH_BITS)?;
    let alloc_re = le_bits_to_num(cs.namespace(|| "bits_to_num"), &re)?;

    // Trivial constraint to ensure that the variables are used 
    cs.enforce(
      || "Trivial constraint",
      |lc| lc + U_i.x0.get_variable() + U_i.x1.get_variable() + U_i.u.get_variable(),
      |lc| lc,
      |lc| lc,
    );
    Ok((alloc_rw, alloc_re))
  }
}

/// Gadget that interpolates the polynomial from the given vector and returns
/// its evaluation at the given point.
/// It also offers the rust native implementation compatible with the gadget.
pub struct EvalGadget {}

impl EvalGadget {
  pub fn evaluate_native<F: PrimeField + GpuName>(mut v: Vec<F>, point: F) -> F {
    // v.resize(v.len().next_power_of_two(), F::ZERO);
    let mut domain = EvaluationDomain::from_coeffs(v).expect("Failed to create evaluation domain");

    let worker = Worker::new(); 
    domain.ifft(&worker, &mut None).expect("FFT failed");

    // Evaluate the polynomial at the given point
    let eval = domain.evaluate_at(point);
    println!("challenge: {:?}", point);
    println!("eval native: {:?}", eval);
    eval
  }

  pub fn evaluate_gadget<CS, E: CurveCycleEquipped>(
    mut cs: CS,
    mut v: Vec<AllocatedNum<E::Scalar>>,
    point: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
    E::Scalar: GpuName,
  {
    let alloc_one = AllocatedNum::alloc(&mut cs, || Ok(E::Scalar::from(1)))?;

    let native_v = v.iter().map(|x| x.get_value().unwrap_or(E::Scalar::from(0))).collect::<Vec<_>>();
    let mut domain = EvaluationDomain::from_coeffs(native_v).expect("Failed to create evaluation domain");
    let omega = domain.omega;
    let n = domain.into_coeffs().len();
    // TODO: Check if nth_root_of_unity is faster than EvaluationDomain::from_coeffs
    // let omega_1 = nth_root_of_unity::<E::Scalar>(n).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    v.resize(n, AllocatedNum::alloc(&mut cs, || Ok(E::Scalar::from(0)))?);
    println!("Allocated one: {:?}", alloc_one.get_value());

    let log2_v = usize::BITS - v.len().leading_zeros() - 1;
    let alloc_domain = AllocatedRadix2Domain::new(&mut cs, omega, log2_v as u64, alloc_one)?;

    let alloc_evaluations = AllocatedEvaluations::from_vec_and_domain(v, alloc_domain, true);
    let eval = alloc_evaluations.interpolate_and_evaluate(&mut cs, point)?;
    println!("2. Eval gadget: {:?}", eval.get_value());
    Ok(eval)
  }
}

pub struct DeciderNovaGadget {}

impl DeciderNovaGadget {
  pub fn fold_group_elements_native<E: CurveCycleEquipped>(
    U_commitments: (Commitment<E>, Commitment<E>),
    u_commitments: Commitment<E>,
    // cmT: Commitment<E>,
    r: E::Scalar,
  ) -> Result<(Commitment<E>, Commitment<E>), SynthesisError> {
    let U_cmW = U_commitments.0;
    let U_cmE = U_commitments.1;
    let u_cmW = u_commitments;
    // *comm_E_1 + *comm_T * *r;
    let cmW = U_cmW + u_cmW * r;
    let cmE = U_cmE; // + cmT * r;

    Ok((cmW, cmE))
  }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KZGProof<E: PairingEngine> {
  pub proof: E::G1,
  pub eval: E::Fr,
}

impl<E: PairingEngine> KZGProof<E> {
  pub fn prove_with_challenge(
    params: &KZGProverKey<E>,
    challenge: E::Fr,
    v: &[E::Fr],
  ) -> Result<KZGProof<E>, PCSError>
  where
    E::G1: DlogGroup<ScalarExt = E::Fr, AffineExt = E::G1Affine>,
    E::Fr: GpuName,
  {
    let mut domain = EvaluationDomain::from_coeffs(v.to_vec())
      .expect("Failed to create evaluation domain");

    let worker = Worker::new(); 
    domain.ifft(&worker, &mut None).expect("FFT failed");

    let polynomial = DensePolynomial::from_coeffs(domain.into_coeffs());
    if polynomial.degree() >= params.powers_of_g().len() {
      return Err(PCSError::LengthError);
    }

    let divisor = DensePolynomial::from_coeffs(vec![-challenge, E::Fr::from(1)]);
    let (witness_poly, remainder_poly) = polynomial.quot_rem(&divisor);

    let eval = if remainder_poly.is_zero() {
      E::Fr::from(0)
    } else {
      remainder_poly.coeffs()[0]
    };

    println!("challenge 1: {:?}", challenge);
    println!("eval 1: {:?}", eval);

    if witness_poly.degree() >= params.powers_of_g().len() {
      return Err(PCSError::LengthError);
    }

    let proof = E::G1::vartime_multiscalar_mul(
      &witness_poly.coeffs(),
      &params.powers_of_g()[..witness_poly.coeffs().len()],
    );

    Ok(KZGProof {
      proof,
      eval,
    })
  }
}