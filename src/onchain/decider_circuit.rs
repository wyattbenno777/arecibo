#![allow(non_snake_case)]
#![allow(unused_imports)]

use std::hash::Hash;

use crate::gadgets::{alloc_scalar_as_base, le_bits_to_num, scalar_as_base};
use crate::onchain::utils::{evaluate_polynomial, lagrange_interpolation};
use crate::provider::poseidon::{PoseidonROCircuit, PoseidonConstantsCircuit};
use crate::{
  cyclefold::gadgets::emulated::{
    AllocatedEmulR1CSInstance, AllocatedEmulRelaxedR1CSInstance, AllocatedEmulRelaxedR1CSWitness,
  },
  errors::NovaError,
  gadgets::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance},
  nebula::{
    nifs::NIFS,
    rs::{PublicParams, RecursiveSNARK},
  },
  onchain::verifiers::nova::NonNativeUintVar,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROConstants, ROTrait, ROCircuitTrait},
  Commitment,
};
use bellpepper_core::boolean::AllocatedBit;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use abomonation::Abomonation;
use super::gadgets::{KZGChallengesGadget, EvalGadget};

// TODO: Add ZK
pub struct DeciderCircuit<E>
where
  E: CurveCycleEquipped,
  <E::Base as PrimeField>::Repr: Abomonation,
{
  /// Constraint system of the Augmented Function circuit
  pub arith: R1CSShape<E>,
  /// R1CS of the CycleFold circuit
  pub cf_arith: R1CSShape<Dual<E>>,
  /// CycleFold PedersenParams over C2
  pub cf_pedersen_params: PublicParams<E>,
  pub ro_consts: ROConstants<Dual<E>>,
  // /// public params hash
  pub pp_hash: E::Scalar,
  pub i: usize,
  /// initial state
  pub z_0: Vec<E::Scalar>,
  /// current i-th state
  pub z_i: Vec<E::Scalar>,
  /// Folding scheme instances
  pub U_i: RelaxedR1CSInstance<E>,
  pub W_i: RelaxedR1CSWitness<E>,
  pub u_i: R1CSInstance<E>,
  pub w_i: R1CSWitness<E>,
  pub U_i1: RelaxedR1CSInstance<E>,
  pub W_i1: RelaxedR1CSWitness<E>,

  // /// Helper for folding verification
  // pub proof: D::Proof,
  // pub randomness: D::Randomness,
  /// CycleFold running instance
  pub cf_U_i: RelaxedR1CSInstance<Dual<E>>,
  pub cf_W_i: RelaxedR1CSWitness<Dual<E>>,

  /// KZG challenges
  pub kzg_challenges: Vec<E::Scalar>,
  /// KZG evaluations
  pub kzg_evaluations: Vec<E::Scalar>,
}


fn hash_instance_var<E: CurveCycleEquipped, CS: ConstraintSystem<E::Base>>(
    cs: &mut CS,
    ro: &mut E::ROCircuit,
    U_i: &AllocatedEmulRelaxedR1CSInstance<E>,
    pp_hash: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    z_0: &[AllocatedNum<E::Scalar>],
    z_i: &[AllocatedNum<E::Scalar>]
) -> Result<Vec<AllocatedBit>, SynthesisError> {
    let pp_hash_base = alloc_scalar_as_base::<E, _>(cs.namespace(|| "pp_hash_base"), pp_hash.get_value())?;
    ROCircuitTrait::absorb(ro, &pp_hash_base);
    let i_base = alloc_scalar_as_base::<E, _>(cs.namespace(|| "i_base"), i.get_value())?;
    ROCircuitTrait::absorb(ro, &i_base);
    for (idx, z) in z_0.iter().enumerate() {
        let z_base = alloc_scalar_as_base::<E, _>(cs.namespace(|| format!("z0_{}", idx)), z.get_value())?;
        ROCircuitTrait::absorb(ro, &z_base);
    }
    for (idx, z) in z_i.iter().enumerate() {
        let z_base = alloc_scalar_as_base::<E, _>(cs.namespace(|| format!("zi_{}", idx)), z.get_value())?;
        ROCircuitTrait::absorb(ro, &z_base);
    }
    U_i.absorb_in_ro(cs.namespace(|| "U_i"), ro)?;
    ROCircuitTrait::squeeze(ro, cs.namespace(|| "squeeze"), 128)
}

// fn fold_field_elements<E: CurveCycleEquipped>(
//     ro: &mut <Dual<E> as Engine>::RO,
//     pp_hash: E::Scalar,
//     U: AllocatedRelaxedR1CSInstance<E, 2>,
//     U_vec: Vec<AllocatedNum<E::Scalar>>,
//     u: AllocatedR1CSInstance<E, 2>,
//     // proof: Commitment<E> // I believe Nebula doesn't have this cross term
// ) -> Result<AllocatedRelaxedR1CSInstance<E, 2>, SynthesisError> {
//     // let cmT = NonNativeUintVar::new(constants, num_absorbs);
//     unimplemented!()
// }

impl<E> DeciderCircuit<E>
where
  E: CurveCycleEquipped,
  <E::Base as PrimeField>::Repr: Abomonation,
{
  pub fn new(pp: &PublicParams<E>, rs: RecursiveSNARK<E>) -> Result<Self, NovaError> {
    let mut ro = <Dual<E> as Engine>::RO::new(
      pp.ro_consts.clone(),
      42, // TODO: Pass a right number
    );

    // compute the U_{i+1}, W_{i+1}
    // 1. compute (Ui+1,Wi+1,T) ← NIFS.P(pk,(Ui,Wi),(ui,wi)),
    let (nifs, (r_U_primary, r_W_primary), (r_U_cyclefold, r_W_cyclefold), r, U_secondary_temp) =
      NIFS::<E>::prove(
        (&pp.ck_primary, &pp.ck_cyclefold),
        &pp.ro_consts,
        &pp.digest(),
        (
          &pp.circuit_shape_primary.r1cs_shape,
          &pp.circuit_shape_cyclefold.r1cs_shape,
        ),
        (&rs.r_U_primary, &rs.r_W_primary),
        (&rs.l_u_primary, &rs.l_w_primary),
        (&rs.r_U_cyclefold, &rs.r_W_cyclefold),
      )?;

    let (rw, re) = KZGChallengesGadget::get_challenges_native(&mut ro, r_U_primary.clone());
    let rw_eval = EvalGadget::evaluate_native(r_W_primary.W, rw);
    let re_eval = EvalGadget::evaluate_native(r_W_primary.E, re);

    Ok(Self {
      arith: pp.circuit_shape_primary.r1cs_shape,
      cf_arith: pp.circuit_shape_cyclefold.r1cs_shape,
      cf_pedersen_params: pp.clone(),
      ro_consts: pp.ro_consts.clone(),
      pp_hash: pp.digest(),
      i: rs.i,
      z_0: rs.z0,
      z_i: rs.zi,
      U_i: rs.r_U_primary,
      W_i: rs.r_W_primary,
      u_i: rs.l_u_primary,
      w_i: rs.l_w_primary,
      U_i1: r_U_primary,
      W_i1: r_W_primary,
      cf_U_i: r_U_cyclefold,
      cf_W_i: r_W_cyclefold,
      kzg_challenges: vec![rw, re],
      kzg_evaluations: vec![rw_eval, re_eval],
    })
  }

  pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<(), SynthesisError> {
    // let arith = AllocatedR1CSInstance::alloc(cs, Some(self.arith))?;

    let pp_hash = AllocatedNum::alloc(cs.namespace(|| "pp_hash"), || Ok(self.pp_hash))?;

    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(E::Scalar::from(self.i as u64)))?;

    let z_0: Vec<AllocatedNum<E::Scalar>> = self
      .z_0
      .iter()
      .enumerate()
      .map(|(i, val)| AllocatedNum::alloc(cs.namespace(|| format!("z_0_{}", i)), || Ok(*val)))
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    let z_i: Vec<AllocatedNum<E::Scalar>> = self
      .z_i
      .iter()
      .enumerate()
      .map(|(i, val)| AllocatedNum::alloc(cs.namespace(|| format!("z_i_{}", i)), || Ok(*val)))
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    let u_i: AllocatedEmulR1CSInstance<E> = AllocatedEmulR1CSInstance::alloc(
      cs.namespace(|| "u_i"), Some(&self.u_i), 1, 2)?;
    let U_i = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "U_i"),
      Some(&self.U_i),
      1, // TODO: Pass a right number
      2, // TODO: Pass a right number
    )?;
    // here (U_i1, W_i1) = NIFS.P( (U_i,W_i), (u_i,w_i))
    // let U_i1_commitments = Vec::<NonNativeAffineVar<C1>>::new_input(cs.clone(), || {
    //     Ok(self.U_i1.get_commitments())
    // })?;

    let U_i1 = AllocatedEmulRelaxedR1CSInstance::alloc(cs, Some(&self.U_i1), 1, 2)?;
    let W_i1 = AllocatedEmulRelaxedR1CSWitness::alloc(cs, Some(&self.W_i1))?;

    // U_i1.get_commitments().enforce_equal(&U_i1_commitments)?;

    let cf_U_i = AllocatedRelaxedR1CSInstance::alloc(cs, Some(&self.cf_U_i), 1, 2)?;

    let kzg_challenges: Vec<AllocatedNum<E::Scalar>> = self
      .kzg_challenges
      .iter()
      .map(|x| AllocatedNum::alloc(cs.namespace(|| "kzg_challenges"), || Ok(*x)))
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    let kzg_evaluations: Vec<AllocatedNum<E::Scalar>> = self
      .kzg_evaluations
      .iter()
      .map(|x| AllocatedNum::alloc(cs.namespace(|| "kzg_evaluations"), || Ok(*x)))
      .collect::<Result<Vec<_>, SynthesisError>>()?;

      let mut ro = <Dual<E> as Engine>::RO::new(
        self.ro_consts.clone(),
        42, // TODO: Pass a right number
      );
  
    // Step 1: Enforce U_{n+1} and W_{n+1} satisfy r1cs
    // Nova has no need for this, since we are checking if an r1cs relation 
    // is sat inside r1cs thus creating another r1cs relation you would have to check is sat. 
    // In any case lets say you could do this what are you eventually going to use to prove the circuit with, 
    // since you could just use that proving mechanism on the original r1cs instance, witness pair.

    // Step 2: Check that u_n.E == 0 and un u_n.u == 1.
    // Trivial

    // Step 3: Verify the hash conditions:
    //         un.x0 == H(n, z0, zn, Un) and un.x1 == H(U_EC,n).
    let hash_bits = hash_instance_var(cs, &mut ro, &U_i, &pp_hash, &i, &z_0, &z_i)?;
    let alloc_hash = le_bits_to_num(cs.namespace(|| "bits_to_num"), &hash_bits)?;

    cs.enforce(
      || "u_i.x[0] == H(i, z_0, z_i, U_i)",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_i.x0.get_variable() - alloc_hash.get_variable(),
    );
    // Step 4: Commitments verification for U_{EC,n}.{E, W} with respect to W_{EC,n}.{E, W}.
    //         - Pedersen commitments are used for this.
    //         - This check is native in Fr because U_{EC,n}.{E, W} ∈ E2.

    // Step 5: Enforce U_{EC,n} and W_{EC,n} satisfy r1cs_{EC},
    //         the Relaxed R1CS relation of the CycleFoldCircuit.
    //         - This involves non-native operations because W_{EC,n}.{E, W} ∈ Fq.
    //         - With naive sparse matrix-vector product, this increases the number of constraints.

    // Step 6.1: Partially enforce that U_{n+1} is the correct folding of U_n and un.
    //           - Only field elements in U_{n+1} are checked, while group elements (commitments) are not.
    //           - Group elements are in E1 and are expensive non-native operations.
    // TODO: Check how this is done in Nebula. We may not need any cross term

    // Step 7.1: Check correct computation of the KZG challenges.
    //           - cE ≡ H(E.{x, y}), cW ≡ H(W.{x, y}).
    let (alloc_rw, alloc_re) = KZGChallengesGadget::get_challenges_gadget(cs, &mut ro, U_i1)?;

    cs.enforce(
      || "cW ≡ H(W.{x, y})",
      |lc| lc,
      |lc| lc,
      |lc| lc + kzg_challenges[0].get_variable() - alloc_rw.get_variable(),
    );

    cs.enforce(
      || "cE ≡ H(E.{x, y})",
      |lc| lc,
      |lc| lc,
      |lc| lc + kzg_challenges[1].get_variable() - alloc_re.get_variable(),
    );

    // Step 7.2: Verify that the KZG evaluations are correct:
    for ((v, c), e) in vec![W_i1.W]
      .iter() 
      .zip(&kzg_challenges)
      .zip(&kzg_evaluations)
    {
      let eval = EvalGadget::evaluate_gadget(cs, *v, c)?;
      cs.enforce(
        || "evalW == pW(cW)",
        |lc| lc,
        |lc| lc + eval.get_variable(),
        |lc| lc + e.get_variable(),
      );
    }

    Ok(())
  }
}
