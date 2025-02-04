#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
use crate::gadgets::le_bits_to_num;
use crate::provider::traits::DlogGroup;
use crate::traits::ROConstantsCircuit;
use crate::CommitmentKey;
use crate::{
  cyclefold::gadgets::emulated::{
    AllocatedEmulR1CSInstance, AllocatedEmulRelaxedR1CSInstance, AllocatedEmulRelaxedR1CSWitness,
  },
  errors::NovaError,
  gadgets::AllocatedRelaxedR1CSInstance,
  nebula::{
    nifs::NIFS,
    rs::{PublicParams, RecursiveSNARK},
  },
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{CurveCycleEquipped, Dual, Engine, ROTrait, ROConstants, ROCircuitTrait},
};
use bellpepper_core::boolean::AllocatedBit;
use bellperson::Circuit;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use super::gadgets::{KZGChallengesGadget, EvalGadget};

// TODO: Add ZK
pub struct DeciderCircuit<E>
where
  E: CurveCycleEquipped,
{
  /// Constraint system of the Augmented Function circuit
  pub arith: R1CSShape<E>,
  /// R1CS of the CycleFold circuit
  pub cf_arith: R1CSShape<Dual<E>>,
  pub ro_consts: ROConstants<E>,
  // /// public params hash
  pub pp_hash: E::Scalar,
  pub i: usize, // TODO: Maybe pass E::Scalar as sonobe
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
  pub nifs_proof: NIFS<E>,
  pub randomness: E::Scalar,

  /// CycleFold running instance
  pub cf_U_i: RelaxedR1CSInstance<Dual<E>>,
  pub cf_W_i: RelaxedR1CSWitness<Dual<E>>,

  /// KZG challenges
  pub kzg_challenges: Vec<E::Scalar>,
  /// KZG evaluations
  pub kzg_evaluations: Vec<E::Scalar>,
}


fn hash_U_i<E: CurveCycleEquipped, CS: ConstraintSystem<E::Scalar>>(
    cs: &mut CS,
    ro: &mut <Dual<E> as Engine>::ROCircuit,
    U_i: &AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
    pp_hash: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    z_0: &Vec<AllocatedNum<E::Scalar>>,
    z_i: &Vec<AllocatedNum<E::Scalar>>
) -> Result<Vec<AllocatedBit>, SynthesisError> {
    ROCircuitTrait::absorb(ro, pp_hash);
    ROCircuitTrait::absorb(ro, i);
    for z in z_0 {
        ROCircuitTrait::absorb(ro, &z);
    }
    for z in z_i {
        ROCircuitTrait::absorb(ro, &z);
    }
    U_i.absorb_in_ro(cs.namespace(|| "U_i"), ro)?;
    ROCircuitTrait::squeeze(ro, cs.namespace(|| "squeeze"), 128)
}

impl<E> DeciderCircuit<E>
where
  E: CurveCycleEquipped,
{
  pub fn default(
    arith: &R1CSShape<E>,
    cf_arith: &R1CSShape<Dual<E>>,
    ro_consts: ROConstants<E>,
    pp_hash: E::Scalar,
    state_len: usize,
    num_commitments: usize,
    (ck, ck_secondary): (&CommitmentKey<E>, &CommitmentKey<Dual<E>>),
  ) -> Self {
    Self {
      arith: arith.clone(),
      cf_arith: cf_arith.clone(),
      ro_consts,
      pp_hash,
      i: 0,
      z_0: vec![E::Scalar::from(0); state_len],
      z_i: vec![E::Scalar::from(0); state_len],
      U_i: RelaxedR1CSInstance::default(ck, arith),
      W_i: RelaxedR1CSWitness::default(arith),
      u_i: R1CSInstance::<E>::default(arith),
      w_i: R1CSWitness::<E>::default(arith),
      U_i1: RelaxedR1CSInstance::default(ck, arith),
      W_i1: RelaxedR1CSWitness::default(arith),
      nifs_proof: NIFS::default(cf_arith),
      randomness: E::Scalar::from(0),
      cf_U_i: RelaxedR1CSInstance::default(ck_secondary, cf_arith),
      cf_W_i: RelaxedR1CSWitness::default(cf_arith),
      kzg_challenges: vec![E::Scalar::from(0); num_commitments],
      kzg_evaluations: vec![E::Scalar::from(0); num_commitments],
    }
  }

  pub fn new(pp: &PublicParams<E>, rs: RecursiveSNARK<E>) -> Result<Self, NovaError> {
    let ro_consts = ROConstants::<E>::default(); // TODO: Not sure if this is OK
    let mut ro = <E as Engine>::RO::new(
      ROConstants::<E>::default(),
      42,
    );

    // TODO: Do I need to run an iteration for IS and FS in Nebula?
    // 1. Compute the U_{i+1}, W_{i+1}
    let (nifs, (r_U_primary, r_W_primary), (r_U_cyclefold, r_W_cyclefold), rho, _U_secondary_temp) =
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
    let rw_eval = EvalGadget::evaluate_native(r_W_primary.clone().W, rw);
    let re_eval = EvalGadget::evaluate_native(r_W_primary.clone().E, re);

    Ok(Self {
      arith: pp.circuit_shape_primary.r1cs_shape.clone(),
      cf_arith: pp.circuit_shape_cyclefold.r1cs_shape.clone(),
      ro_consts: ro_consts,
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
      randomness: rho,
      nifs_proof: nifs
    })
  }

}

impl<E> Circuit<E::Scalar> for DeciderCircuit<E> 
  where
    E: CurveCycleEquipped {
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    self,
    cs: &mut CS,
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

    let u_i: AllocatedEmulR1CSInstance<Dual<E>> = AllocatedEmulR1CSInstance::alloc(
      cs.namespace(|| "u_i"), Some(&self.u_i), 1, 2)?;
    let U_i: AllocatedEmulRelaxedR1CSInstance<Dual<E>> = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "U_i"),
      Some(&self.U_i),
      1, // TODO: Pass a right number
      2, // TODO: Pass a right number
    )?;
    // here (U_i1, W_i1) = NIFS.P( (U_i,W_i), (u_i,w_i))
    // let U_i1_commitments = Vec::<NonNativeAffineVar<C1>>::new_input(cs.clone(), || {
    //     Ok(self.U_i1.get_commitments())
    // })?;

    let U_i1: AllocatedEmulRelaxedR1CSInstance<Dual<E>> = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "U_i1"), Some(&self.U_i1), 1, 2)?;
    let W_i1: AllocatedEmulRelaxedR1CSWitness<Dual<E>> = AllocatedEmulRelaxedR1CSWitness::alloc(
      cs.namespace(|| "W_i1"), Some(&self.W_i1))?;

    // U_i1.get_commitments().enforce_equal(&U_i1_commitments)?;

    let cf_U_i: AllocatedRelaxedR1CSInstance<Dual<E>, 2> = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "cf_U_i"), Some(&self.cf_U_i), 1, 2)?;

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

      let mut ro = <Dual<E> as Engine>::ROCircuit::new(
        ROConstantsCircuit::<Dual<E>>::default(),
        42,
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
    let hash_bits = hash_U_i::<E, CS>(
      cs, 
      &mut ro, 
      &U_i, 
      &pp_hash, 
      &i, 
      &z_0, 
      &z_i)?;
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
    // let (alloc_rw, alloc_re) = KZGChallengesGadget::get_challenges_gadget(
    //   cs, 
    //   // &mut ro, 
    //   U_i1)?;

    // cs.enforce(
    //   || "cW ≡ H(W.{x, y})",
    //   |lc| lc,
    //   |lc| lc,
    //   |lc| lc + kzg_challenges[0].get_variable() - alloc_rw.get_variable(),
    // );

    // cs.enforce(
    //   || "cE ≡ H(E.{x, y})",
    //   |lc| lc,
    //   |lc| lc,
    //   |lc| lc + kzg_challenges[1].get_variable() - alloc_re.get_variable(),
    // );

    // Step 7.2: Verify that the KZG evaluations are correct:
    for ((v, c), e) in vec![W_i1.W]
      .iter() 
      .zip(&kzg_challenges)
      .zip(&kzg_evaluations)
    {
      let eval = EvalGadget::evaluate_gadget::<&mut CS, E>(cs, v.clone(), c)?;
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
