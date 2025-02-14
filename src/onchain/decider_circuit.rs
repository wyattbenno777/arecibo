#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
use ff::PrimeField;

use super::gadgets::{EvalGadget, KZGChallengesGadget};
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS, NUM_HASH_BITS},
  cyclefold::gadgets::emulated::{
    AllocatedEmulPoint, AllocatedEmulR1CSInstance, AllocatedEmulRelaxedR1CSInstance,
    AllocatedEmulRelaxedR1CSWitness,
  },
  errors::NovaError,
  frontend::{
    gpu::GpuName, num::AllocatedNum, AllocatedBit, Circuit, ConstraintSystem, Index,
    SynthesisError, Variable,
  },
  gadgets::{alloc_num_equals, le_bits_to_num, AllocatedRelaxedR1CSInstance},
  nebula::{
    nifs::NIFS,
    rs::{PublicParams, RecursiveSNARK},
  },
  provider::traits::DlogGroup,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROCircuitTrait, ROConstants,
    ROConstantsCircuit, ROTrait,
  },
  CommitmentKey,
};

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
  pub prev_IC: E::Scalar,
  pub r_i: E::Scalar,
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
  pub kzg_challenges: (E::Scalar, E::Scalar),
  /// KZG evaluations
  pub kzg_evaluations: (E::Scalar, E::Scalar),
}

fn hash_U_i<E: CurveCycleEquipped, CS: ConstraintSystem<E::Scalar>>(
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
    ro.absorb(&z);
  }
  for z in z_i {
    ro.absorb(&z);
  }
  U_i.absorb_in_ro(cs.namespace(|| "U_i"), &mut ro)?;
  ro.absorb(prev_IC);
  ro.absorb(r_i);
  let hash_bits_p = ro.squeeze(cs.namespace(|| "primary hash bits"), NUM_HASH_BITS)?;
  le_bits_to_num(cs.namespace(|| "bits_to_num"), &hash_bits_p)
}

fn hash_cf_U_i<E: CurveCycleEquipped, CS: ConstraintSystem<E::Scalar>>(
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

impl<E> DeciderCircuit<E>
where
  E: CurveCycleEquipped,
  E::Scalar: GpuName,
{
  pub fn default(
    arith: &R1CSShape<E>,
    cf_arith: &R1CSShape<Dual<E>>,
    ro_consts: ROConstants<E>,
    pp_hash: E::Scalar,
    state_len: usize,
    (ck, ck_secondary): (&CommitmentKey<E>, &CommitmentKey<Dual<E>>),
  ) -> Self {
    Self {
      arith: arith.clone(),
      cf_arith: cf_arith.clone(),
      ro_consts,
      pp_hash,
      i: 0,
      prev_IC: E::Scalar::from(0),
      r_i: E::Scalar::from(0),
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
      kzg_challenges: (E::Scalar::from(0), E::Scalar::from(0)),
      kzg_evaluations: (E::Scalar::from(0), E::Scalar::from(0)),
    }
  }

  pub fn new(pp: &PublicParams<E>, rs: RecursiveSNARK<E>) -> Result<Self, NovaError> {
    let ro_consts = ROConstants::<E>::default(); // TODO: Not sure if this is OK

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

    let (rw, re) =
      KZGChallengesGadget::get_challenges_native(r_U_primary.clone());
    let rw_eval = EvalGadget::evaluate_native(r_W_primary.clone().W, rw);
    let re_eval = EvalGadget::evaluate_native(r_W_primary.clone().E, re);

    Ok(Self {
      arith: pp.circuit_shape_primary.r1cs_shape.clone(),
      cf_arith: pp.circuit_shape_cyclefold.r1cs_shape.clone(),
      ro_consts: ro_consts,
      pp_hash: pp.digest(),
      i: rs.i,
      prev_IC: rs.prev_IC,
      r_i: rs.r_i,
      z_0: rs.z0,
      z_i: rs.zi,
      U_i: rs.r_U_primary,
      W_i: rs.r_W_primary,
      u_i: rs.l_u_primary,
      w_i: rs.l_w_primary,
      U_i1: r_U_primary,
      W_i1: r_W_primary,
      cf_U_i: rs.r_U_cyclefold,
      cf_W_i: rs.r_W_cyclefold,
      kzg_challenges: (rw, re),
      kzg_evaluations: (rw_eval, re_eval),
      randomness: rho,
      nifs_proof: nifs,
    })
  }
}

impl<E> Circuit<E::Scalar> for DeciderCircuit<E>
where
  E: CurveCycleEquipped,
  E::Scalar: GpuName,
{
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // let arith = AllocatedR1CSInstance::alloc(cs, Some(self.arith))?;

    let pp_hash = AllocatedNum::alloc(cs.namespace(|| "get pp_hash"), || Ok(self.pp_hash))?;
    pp_hash.inputize(cs.namespace(|| "pp_hash"))?;

    let i = AllocatedNum::alloc(cs.namespace(|| "get i"), || {
      Ok(E::Scalar::from(self.i as u64))
    })?;
    i.inputize(cs.namespace(|| "i"))?;

    let prev_IC = AllocatedNum::alloc(cs.namespace(|| "get prev_IC"), || Ok(self.prev_IC))?;

    let r_i = AllocatedNum::alloc(cs.namespace(|| "get r_i"), || Ok(self.r_i))?;

    let z_0: Vec<AllocatedNum<E::Scalar>> = self
      .z_0
      .iter()
      .enumerate()
      .map(|(i, val)| {
        let tmp = AllocatedNum::alloc(cs.namespace(|| format!("z_0_{}", i)), || Ok(*val))?;
        tmp.inputize(cs.namespace(|| format!("z_0_{}", i)))?;
        Ok(tmp)
      })
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    let z_i: Vec<AllocatedNum<E::Scalar>> = self
      .z_i
      .iter()
      .enumerate()
      .map(|(i, val)| {
        let tmp = AllocatedNum::alloc(cs.namespace(|| format!("z_i_{}", i)), || Ok(*val))?;
        tmp.inputize(cs.namespace(|| format!("z_i_{}", i)))?;
        Ok(tmp)
      })
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    let u_i_x0 = AllocatedNum::alloc(cs.namespace(|| "allocate x0"), || Ok(self.u_i.X[0]))?;

    let U_i: AllocatedEmulRelaxedR1CSInstance<Dual<E>> = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "U_i"),
      Some(&self.U_i),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Step 1: Enforce U_{n+1} and W_{n+1} satisfy r1cs
    // Nova has no need for this, since we are checking if an r1cs relation
    // is sat inside r1cs thus creating another r1cs relation you would have to check is sat.
    // In any case lets say you could do this what are you eventually going to use to prove the circuit with,
    // since you could just use that proving mechanism on the original r1cs instance, witness pair.

    // Step 2: Check that u_n.E == 0 and un u_n.u == 1.
    // Trivial

    // Step 3: Verify the hash conditions:
    //         un.x0 == H(n, z0, zn, Un) and un.x1 == H(U_EC,n).

    let U_i_hash = hash_U_i::<E, CS>(cs, &U_i, &pp_hash, &i, &z_0, &z_i, &prev_IC, &r_i)?;

    cs.enforce(
      || "u_i.x[0] == H(i, z_0, z_i, U_i)",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_i_x0.get_variable() - U_i_hash.get_variable(),
    );


    // TODO: Maybe use cs.alloc instead
    let u_i_x1 = AllocatedNum::alloc(cs.namespace(|| "allocate x1"), || Ok(self.u_i.X[1]))?;
    let cf_U_i: AllocatedRelaxedR1CSInstance<Dual<E>, BN_N_LIMBS> =
      AllocatedRelaxedR1CSInstance::alloc(
        cs.namespace(|| "cf_U_i"),
        Some(&self.cf_U_i),
        BN_LIMB_WIDTH,
        BN_N_LIMBS,
      )?;

    let cf_U_i_hash = hash_cf_U_i::<E, CS>(
      cs,
      &cf_U_i,
      &pp_hash,
      &i,
      &r_i,
    )?;

    cs.enforce(
      || "u_i.x[1] == H(U_EC, i)",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_i_x1.get_variable() - cf_U_i_hash.get_variable(),
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
    let kzg_alloc_rw = AllocatedNum::alloc(cs.namespace(|| "get kzg_challenges rw"), || Ok(self.kzg_challenges.0))?;
    kzg_alloc_rw.inputize(cs.namespace(|| "kzg_alloc_rw"))?;
    let kzg_alloc_re = AllocatedNum::alloc(cs.namespace(|| "get kzg_challenges re"), || Ok(self.kzg_challenges.1))?;
    kzg_alloc_re.inputize(cs.namespace(|| "kzg_alloc_re"))?;

    let U_i1: AllocatedEmulRelaxedR1CSInstance<Dual<E>> = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "U_i1"),
      Some(&self.U_i1),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let (alloc_rw, alloc_re) =
      KZGChallengesGadget::get_challenges_gadget::<CS, E>(cs, U_i1)?;

    cs.enforce(
      || "cW ≡ H(W.{x, y})",
      |lc| lc,
      |lc| lc,
      |lc| lc + kzg_alloc_rw.get_variable() - alloc_rw.get_variable(),
    );

    cs.enforce(
      || "cE ≡ H(E.{x, y})",
      |lc| lc,
      |lc| lc,
      |lc| lc + kzg_alloc_re.get_variable() - alloc_re.get_variable(),
    );


    // Step 7.2: Verify that the KZG evaluations are correct
    let kzg_alloc_rw_eval = AllocatedNum::alloc(cs.namespace(|| "get kzg_evaluations"), || Ok(self.kzg_evaluations.0))?;
    kzg_alloc_rw_eval.inputize(cs.namespace(|| "kzg_alloc_rw_eval"))?;
    let kzg_alloc_re_eval = AllocatedNum::alloc(cs.namespace(|| "get kzg_evaluations"), || Ok(self.kzg_evaluations.1))?;
    kzg_alloc_re_eval.inputize(cs.namespace(|| "kzg_alloc_re_eval"))?;

    let W_i1_W = self.W_i1.W.iter().map(|x| {
      AllocatedNum::alloc(
        cs.namespace(|| "allocate W_i1.W"),
        || Ok(*x)
      )
    }).collect::<Result<Vec<_>, _>>()?;
    
    for x in &W_i1_W {
      cs.enforce(
        || "W_i1.W",
        |lc| lc + x.get_variable(),
        |lc| lc,
        |lc| lc,
      );
    }

    let W_i1_E = self.W_i1.E.iter().map(|x| {
      AllocatedNum::alloc(
        cs.namespace(|| "allocate W_i1.E"),
        || Ok(*x)
      )
    }).collect::<Result<Vec<_>, _>>()?;

    for x in &W_i1_E {
      cs.enforce(
        || "W_i1.E",
        |lc| lc + x.get_variable(),
        |lc| lc,
        |lc| lc,
      );
    }

    let alloc_rw_eval = EvalGadget::evaluate_gadget::<&mut CS, E>(cs,W_i1_W, &kzg_alloc_rw)?;
    cs.enforce(
      || "evalW == pW(cW)",
      |lc| lc,
      |lc| lc,
      |lc| lc + kzg_alloc_rw_eval.get_variable() - alloc_rw_eval.get_variable(),
    );

    let alloc_re_eval = EvalGadget::evaluate_gadget::<&mut CS, E>(cs, W_i1_E, &kzg_alloc_re)?;
    cs.enforce(
      || "evalE == pE(cE)",
      |lc| lc,
      |lc| lc,  
      |lc| lc + kzg_alloc_re_eval.get_variable() - alloc_re_eval.get_variable(),
    );

    Ok(())
  }
}

