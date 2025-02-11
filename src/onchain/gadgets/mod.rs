#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use crate::constants::NUM_HASH_BITS;
use crate::cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance;
use crate::cyclefold::util::absorb_primary_commitment;
use crate::frontend::domain::EvaluationDomain;
use crate::frontend::gpu::GpuName;
use crate::frontend::num::AllocatedNum;
use crate::frontend::{ConstraintSystem, SynthesisError};
use crate::gadgets::{le_bits_to_num, scalar_as_base};
use crate::r1cs::RelaxedR1CSInstance;
use crate::traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROCircuitTrait, ROConstants, ROConstantsCircuit, ROTrait};
use crate::Commitment;
use ec_gpu_gen::threadpool::Worker;
use ff::PrimeField;

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