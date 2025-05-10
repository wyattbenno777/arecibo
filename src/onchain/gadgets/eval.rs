use crate::frontend::domain::EvaluationDomain;
use crate::frontend::gpu::GpuName;
use crate::frontend::num::AllocatedNum;
use crate::frontend::{ConstraintSystem, SynthesisError};
use crate::traits::CurveCycleEquipped;
use ec_gpu_gen::threadpool::Worker;
use ff::PrimeField;
use crate::onchain::gadgets::domain::{AllocatedEvaluations, AllocatedRadix2Domain};

/// Gadget that interpolates the polynomial from the given vector and returns
/// its evaluation at the given point.
/// It also offers the rust native implementation compatible with the gadget.
pub struct EvalGadget {}

impl EvalGadget {
  /// Evaluate a polynomial natively.
  pub fn evaluate_native<F: PrimeField + GpuName>(v: Vec<F>, point: F) -> F {
    let mut domain = EvaluationDomain::from_coeffs(v).expect("Failed to create evaluation domain");

    let worker = Worker::new(); 
    domain.ifft(&worker, &mut None).expect("FFT failed");

    // Evaluate the polynomial at the given point
    domain.evaluate_at(point)
  }

  /// Evaluate a polynomial in circuit.
  pub fn evaluate_gadget<CS, E: CurveCycleEquipped>(
    mut cs: CS,
    mut v: Vec<AllocatedNum<E::Scalar>>,
    point: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
    E::Scalar: GpuName,
  {
    let native_v = v.iter().map(|x| x.get_value().unwrap_or(E::Scalar::from(0))).collect::<Vec<_>>();
    let domain = EvaluationDomain::from_coeffs(native_v).expect("Failed to create evaluation domain");
    let omega = domain.omega;
    let n = domain.into_coeffs().len();
    // TODO: Check if nth_root_of_unity is faster than EvaluationDomain::from_coeffs
    // let omega_1 = nth_root_of_unity::<E::Scalar>(n).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    v.resize(n, AllocatedNum::alloc(&mut cs, || Ok(E::Scalar::from(0)))?);

    let log2_v = usize::BITS - v.len().leading_zeros() - 1;
    let alloc_domain = AllocatedRadix2Domain::new(omega, log2_v as u64);

    let alloc_evaluations = AllocatedEvaluations::from_vec_and_domain(v, alloc_domain, true);
    let eval = alloc_evaluations.interpolate_and_evaluate(&mut cs, point)?;
    Ok(eval)
  }
}
