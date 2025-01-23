#![allow(non_snake_case)]


use crate::gadgets::le_bits_to_num;
use crate::onchain::utils::{evaluate_polynomial, lagrange_interpolation, nth_root_of_unity};
use crate::{
  cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance,
  r1cs::RelaxedR1CSInstance,
  traits::{AbsorbInROTrait, CurveCycleEquipped,  ROTrait, ROCircuitTrait},
};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use radix_domain::{AllocatedEvaluations, AllocatedRadix2Domain};

pub mod radix_domain;

/// Gadget that computes the KZG challenges.
/// It also offers the rust native implementation compatible with the gadget.
pub struct KZGChallengesGadget {}

impl KZGChallengesGadget {
  pub fn get_challenges_native<E: CurveCycleEquipped>(
    ro: &mut E::RO,
    U_i: RelaxedR1CSInstance<E>,
  ) -> (E::Scalar, E::Scalar) {
    U_i.comm_W.absorb_in_ro(ro);
    let rw = ROTrait::squeeze(ro, 128); //TODO: Choose right number

    U_i.comm_E.absorb_in_ro(ro);
    let re = ROTrait::squeeze(ro, 128); //TODO: Choose right number

    (rw, re)
  }

  pub fn get_challenges_gadget<CS, E: CurveCycleEquipped>(
    mut cs: CS,
    ro: &mut E::ROCircuit,
    U_i: AllocatedEmulRelaxedR1CSInstance<E>,
  ) -> Result<(AllocatedNum<E::Base>, AllocatedNum<E::Base>), SynthesisError>
  where
    CS: ConstraintSystem<E::Base>,
  {
    U_i.comm_W.absorb_in_ro(cs.namespace(|| "absorb_W"), ro)?;
    let rw = ROCircuitTrait::squeeze(ro, cs.namespace(|| "squeeze_W"), 128)?;
    let alloc_rw = le_bits_to_num(cs.namespace(|| "bits_to_num"), &rw)?;

    U_i.comm_E.absorb_in_ro(cs.namespace(|| "absorb_E"), ro)?;
    let re = ROCircuitTrait::squeeze(ro, cs.namespace(|| "squeeze_E"), 128)?;
    let alloc_re = le_bits_to_num(cs.namespace(|| "bits_to_num"), &re)?;
    Ok((alloc_rw, alloc_re))
  }
}

/// Gadget that interpolates the polynomial from the given vector and returns
/// its evaluation at the given point.
/// It also offers the rust native implementation compatible with the gadget.
pub struct EvalGadget {}

impl EvalGadget {
  pub fn evaluate_native<F: PrimeField>(mut v: Vec<F>, point: F) -> F {
    v.resize(v.len().next_power_of_two(), F::ZERO);
    let p = lagrange_interpolation(v);
    let eval = evaluate_polynomial(p, point);
    eval
  }

  pub fn evaluate_gadget<CS, E: CurveCycleEquipped>(    
    mut cs: CS, 
    mut v: Vec<AllocatedNum<E::Base>>, 
    point: &AllocatedNum<E::Base>
  ) -> Result<AllocatedNum<E::Base>, SynthesisError> 
  where
    CS: ConstraintSystem<E::Base> 
  {
    let alloc_zero = AllocatedNum::alloc(&mut cs, || Ok(E::Base::from(0)))?;
    v.resize(v.len().next_power_of_two(), alloc_zero);
    let n = v.len() as usize;
    let gen = nth_root_of_unity::<E::Base>(n).ok_or(SynthesisError::PolynomialDegreeTooLarge)?; // TODO: Use a better error
    let alloc_one = AllocatedNum::alloc(&mut cs, || Ok(E::Base::from(1)))?;
    let log2_v = usize::BITS - v.len().leading_zeros() - 1;
    let domain = AllocatedRadix2Domain::new(&mut cs, gen, log2_v as u64, alloc_one)?;

    let alloc_evaluations = AllocatedEvaluations::from_vec_and_domain(v, domain, true);
    alloc_evaluations.interpolate_and_evaluate(&mut cs, point)
  }
}
