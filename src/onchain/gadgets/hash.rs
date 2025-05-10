use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_HASH_BITS},
  cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance,
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{le_bits_to_num, AllocatedRelaxedR1CSInstance},
  traits::{CurveCycleEquipped, Dual, Engine, ROCircuitTrait, ROConstantsCircuit},
};

/// Hash a relaxed R1CS instance.
pub fn hash_U_i<E: CurveCycleEquipped, CS: ConstraintSystem<E::Scalar>>(
  cs: &mut CS,
  U_i: &AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
  pp_hash: &AllocatedNum<E::Scalar>,
  i: &AllocatedNum<E::Scalar>,
  z_0: &Vec<AllocatedNum<E::Scalar>>,
  z_i: &Vec<AllocatedNum<E::Scalar>>,
  prev_IC: &AllocatedNum<E::Scalar>,
  r_i: &AllocatedNum<E::Scalar>,
) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
  let mut ro = <Dual<E> as Engine>::ROCircuit::new(
    ROConstantsCircuit::<Dual<E>>::default(),
    25 + z_0.len() + z_i.len(),
  );
  ro.absorb(pp_hash);
  ro.absorb(i);
  for z in z_0 {
    ro.absorb(z);
  }
  for z in z_i {
    ro.absorb(z);
  }
  U_i.absorb_in_ro(cs.namespace(|| "U_i"), &mut ro)?;
  ro.absorb(prev_IC);
  ro.absorb(r_i);
  let hash_bits_p = ro.squeeze(cs.namespace(|| "primary hash bits"), NUM_HASH_BITS)?;
  le_bits_to_num(cs.namespace(|| "bits_to_num"), &hash_bits_p)
}

/// Hash a cyclefoldcommitment.
pub fn hash_cf_U_i<E: CurveCycleEquipped, CS: ConstraintSystem<E::Scalar>>(
  cs: &mut CS,
  cf_U_i: &AllocatedRelaxedR1CSInstance<Dual<E>, BN_N_LIMBS>,
  pp_hash: &AllocatedNum<E::Scalar>,
  i: &AllocatedNum<E::Scalar>,
  r_i: &AllocatedNum<E::Scalar>,
) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
  let mut ro = <Dual<E> as Engine>::ROCircuit::new(
    ROConstantsCircuit::<Dual<E>>::default(),
    1 + 1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS, // r_i + pp + i + W + E + u + X
  );
  ro.absorb(pp_hash);
  ro.absorb(i);
  cf_U_i.absorb_in_ro(cs.namespace(|| "cf_U_i"), &mut ro)?;
  ro.absorb(r_i);
  let cf_U_i_hash_bits = ro.squeeze(cs.namespace(|| "squeeze"), NUM_HASH_BITS)?;
  le_bits_to_num(cs.namespace(|| "bits_to_num"), &cf_U_i_hash_bits)
}
