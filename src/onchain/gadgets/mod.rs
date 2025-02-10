#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use crate::{
  constants::NUM_HASH_BITS, cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance, frontend::{
    domain::EvaluationDomain, gpu::GpuName, num::AllocatedNum, ConstraintSystem, SynthesisError,
  }, gadgets::le_bits_to_num, r1cs::RelaxedR1CSInstance, traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, ROCircuitTrait, ROTrait}, Commitment
};
use ec_gpu_gen::threadpool::Worker;
use ff::PrimeField;

/// Gadget that computes the KZG challenges.
/// It also offers the rust native implementation compatible with the gadget.
pub struct KZGChallengesGadget {}

impl KZGChallengesGadget {
  pub fn get_challenges_native<E: CurveCycleEquipped>(
    ro_1: &mut E::RO,
    ro_2: &mut E::RO,
    U_i: RelaxedR1CSInstance<E>,
  ) -> (E::Scalar, E::Scalar) {
    U_i.comm_W.absorb_in_ro(ro_1);
    let rw = ro_1.squeeze(NUM_HASH_BITS); 

    U_i.comm_E.absorb_in_ro(ro_2);
    let re = ro_2.squeeze(NUM_HASH_BITS);

    (rw, re)
  }

  pub fn get_challenges_gadget<CS, RO, E: CurveCycleEquipped>(
    cs: &mut CS,
    ro_1: &mut RO,
    ro_2: &mut RO,
    U_i: AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
  ) -> Result<(AllocatedNum<E::Scalar>, AllocatedNum<E::Scalar>), SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
    RO: ROCircuitTrait<E::Scalar>,
  {
    U_i.comm_W.absorb_in_ro(cs.namespace(|| "absorb_W"), ro_1)?;
    let rw = ro_1.squeeze(cs.namespace(|| "squeeze_W"), NUM_HASH_BITS)?;
    let alloc_rw = le_bits_to_num(cs.namespace(|| "bits_to_num"), &rw)?;

    U_i.comm_E.absorb_in_ro(cs.namespace(|| "absorb_E"), ro_2)?;
    let re = ro_2.squeeze(cs.namespace(|| "squeeze_E"), NUM_HASH_BITS)?;
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
    v.resize(v.len().next_power_of_two(), F::ZERO);
    // Create an evaluation domain from the coefficients
    let mut domain = EvaluationDomain::from_coeffs(v).expect("Failed to create evaluation domain");

    // Perform FFT to transform the polynomial into evaluation form
    let worker = Worker::new(); // Assuming you have a worker for parallel computation
    domain.fft(&worker, &mut None).expect("FFT failed");

    // Evaluate the polynomial at the given point
    let eval = domain.evaluate_at(point);

    eval
  }

  pub fn evaluate_gadget<CS, E: CurveCycleEquipped>(
    mut cs: CS,
    v: &Vec<AllocatedNum<E::Scalar>>,
    point: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
    E::Scalar: GpuName,
  {
    // Convert AllocatedNum to native field elements
    let mut native_v: Vec<E::Scalar> = v
      .iter()
      .map(|num| num.get_value().unwrap_or(E::Scalar::from(0)))
      .collect();

    // Resize to the next power of two
    native_v.resize(native_v.len().next_power_of_two(), E::Scalar::from(0));

    // Create an evaluation domain from the coefficients
    let mut domain =
      EvaluationDomain::from_coeffs(native_v).expect("Failed to create evaluation domain");

    // Perform FFT to transform the polynomial into evaluation form
    let worker = Worker::new();
    domain.fft(&worker, &mut None).expect("FFT failed");

    // Evaluate the polynomial at the given point
    let point_value = point.get_value().unwrap_or(E::Scalar::from(0));
    let eval = domain.evaluate_at(point_value);

    // Convert the result back to AllocatedNum
    AllocatedNum::alloc(&mut cs, || Ok(eval))
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