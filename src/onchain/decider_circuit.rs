#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
use ff::PrimeField;

use crate::constants::{BN_LIMB_WIDTH, BN_N_LIMBS};
use crate::cyclefold::gadgets::emulated::AllocatedEmulPoint;
use crate::frontend::gpu::GpuName;
use crate::frontend::{Index, Variable};
use crate::gadgets::le_bits_to_num;
use crate::provider::traits::DlogGroup;
use crate::traits::commitment::CommitmentTrait;
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
use crate::frontend::{
  AllocatedBit,
  num::AllocatedNum, 
  Circuit,
  ConstraintSystem, SynthesisError};
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

pub fn mimc<S: PrimeField>(mut xl: S, mut xr: S, constants: &[S]) -> S {
  assert_eq!(constants.len(), MIMC_ROUNDS);

  for c in constants {
      let mut tmp1 = xl;
      tmp1.add_assign(c);
      let mut tmp2 = tmp1.square();
      tmp2.mul_assign(&tmp1);
      tmp2.add_assign(&xr);
      xr = xl;
      xl = tmp2;
  }

  xl
}
pub const MIMC_ROUNDS: usize = 322;

/// This is our demo circuit for proving knowledge of the
/// preimage of a MiMC hash invocation.
#[allow(clippy::upper_case_acronyms)]
pub struct MiMCDemo<'a, S: PrimeField> {
  pub xl: Option<S>,
  pub xr: Option<S>,
  pub constants: &'a [S],
}

/// Our demo circuit implements this `Circuit` trait which
/// is used during paramgen and proving in order to
/// synthesize the constraint system.
impl<'a, S: PrimeField> Circuit<S> for MiMCDemo<'a, S> {
  fn synthesize<CS: ConstraintSystem<S>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
      assert_eq!(self.constants.len(), MIMC_ROUNDS);

      // Allocate the first component of the preimage.
      let mut xl_value = self.xl;
      let mut xl = cs.alloc(
          || "preimage xl",
          || xl_value.ok_or(SynthesisError::AssignmentMissing),
      )?;

      // Allocate the second component of the preimage.
      let mut xr_value = self.xr;
      let mut xr = cs.alloc(
          || "preimage xr",
          || xr_value.ok_or(SynthesisError::AssignmentMissing),
      )?;

      for i in 0..MIMC_ROUNDS {
          // xL, xR := xR + (xL + Ci)^3, xL
          let cs = &mut cs.namespace(|| format!("round {}", i));

          // tmp = (xL + Ci)^2
          let tmp_value = xl_value.map(|mut e| {
              e.add_assign(&self.constants[i]);
              e.square()
          });
          let tmp = cs.alloc(
              || "tmp",
              || tmp_value.ok_or(SynthesisError::AssignmentMissing),
          )?;

          cs.enforce(
              || "tmp = (xL + Ci)^2",
              |lc| lc + xl + (self.constants[i], CS::one()),
              |lc| lc + xl + (self.constants[i], CS::one()),
              |lc| lc + tmp,
          );

          // new_xL = xR + (xL + Ci)^3
          // new_xL = xR + tmp * (xL + Ci)
          // new_xL - xR = tmp * (xL + Ci)
          let new_xl_value = xl_value.map(|mut e| {
              e.add_assign(&self.constants[i]);
              e.mul_assign(&tmp_value.unwrap());
              e.add_assign(&xr_value.unwrap());
              e
          });

          let new_xl = if i == (MIMC_ROUNDS - 1) {
              // This is the last round, xL is our image and so
              // we allocate a public input.
              cs.alloc_input(
                  || "image",
                  || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
              )?
          } else {
              cs.alloc(
                  || "new_xl",
                  || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
              )?
          };

          cs.enforce(
              || "new_xL = xR + (xL + Ci)^3",
              |lc| lc + tmp,
              |lc| lc + xl + (self.constants[i], CS::one()),
              |lc| lc + new_xl - xr,
          );

          // xR = xL
          xr = xl;
          xr_value = xl_value;

          // xL = new_xL
          xl = new_xl;
          xl_value = new_xl_value;
      }

      Ok(())
  }
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

fn hash_cf_U_i<E: CurveCycleEquipped, CS: ConstraintSystem<E::Scalar>>(
    cs: &mut CS,
    ro: &mut <Dual<E> as Engine>::ROCircuit,
    cf_U_i: &AllocatedRelaxedR1CSInstance<Dual<E>, BN_N_LIMBS>,
    pp_hash: &AllocatedNum<E::Scalar>,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
    ROCircuitTrait::absorb(ro, pp_hash);
    cf_U_i.absorb_in_ro(cs.namespace(|| "cf_U_i"), ro)?;
    ROCircuitTrait::squeeze(ro, cs.namespace(|| "squeeze"), 128)
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
    let mut ro_1 = <E as Engine>::RO::new(
      ROConstants::<E>::default(),
      3,
    );
    let mut ro_2 = <E as Engine>::RO::new(
      ROConstants::<E>::default(),
      3,
    );

    // TODO: Do I need to run an iteration for IS and FS in Nebula?
    println!("DeciderCircuit::new: 1. Compute the U_i+1, W_i+1");
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

    println!("DeciderCircuit::new: 2. Compute the KZG challenges");
    let (rw, re) = KZGChallengesGadget::get_challenges_native(&mut ro_1, &mut ro_2, r_U_primary.clone());
    println!("DeciderCircuit::new: 3. Compute the KZG evaluations");
    let rw_eval = EvalGadget::evaluate_native(r_W_primary.clone().W, rw);
    println!("DeciderCircuit::new: 4. Compute the KZG evaluations");
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
    E: CurveCycleEquipped,
    E::Scalar: GpuName,
  {
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

    // We don't need to check u_i.W
    let u_i_x0 = AllocatedNum::alloc(cs.namespace(|| "allocate x0"), || {
      Ok(self.u_i.X[0])
    })?;

    let u_i_x1 = AllocatedNum::alloc(cs.namespace(|| "allocate x1"), || {
      Ok(self.u_i.X[1])
    })?;

    let U_i: AllocatedEmulRelaxedR1CSInstance<Dual<E>> = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "U_i"),
      Some(&self.U_i),
      BN_LIMB_WIDTH, BN_N_LIMBS
    )?;
    // here (U_i1, W_i1) = NIFS.P( (U_i,W_i), (u_i,w_i))
    let U_i1: AllocatedEmulRelaxedR1CSInstance<Dual<E>> = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "U_i1"), Some(&self.U_i1), BN_LIMB_WIDTH, BN_N_LIMBS)?;
    
    // We don't need to check W_i1.E
    let W_i1_W = self.W_i1.W.iter().map(|x| {
      AllocatedNum::alloc(
        cs.namespace(|| "allocate W_i1.W"),
        || Ok(*x)
      )
    }).collect::<Result<Vec<_>, _>>()?;

    // I don't think this is necessary
    // U_i1.get_commitments().enforce_equal(&U_i1_commitments)?;

    let cf_U_i: AllocatedRelaxedR1CSInstance<Dual<E>, BN_N_LIMBS> = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "cf_U_i"), Some(&self.cf_U_i), BN_LIMB_WIDTH, BN_N_LIMBS)?;

    let kzg_challenges: Vec<AllocatedNum<E::Scalar>> = self
      .kzg_challenges
      .iter()
      .map(|x| AllocatedNum::alloc(cs.namespace(|| "kzg_challenges"), || Ok(*x)))
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    println!("kzg_challenges len: {:?}", kzg_challenges.len());

    let kzg_evaluations: Vec<AllocatedNum<E::Scalar>> = self
      .kzg_evaluations
      .iter()
      .map(|x| AllocatedNum::alloc(cs.namespace(|| "kzg_evaluations"), || Ok(*x)))
      .collect::<Result<Vec<_>, SynthesisError>>()?;
  
    // Step 1: Enforce U_{n+1} and W_{n+1} satisfy r1cs
    // Nova has no need for this, since we are checking if an r1cs relation 
    // is sat inside r1cs thus creating another r1cs relation you would have to check is sat. 
    // In any case lets say you could do this what are you eventually going to use to prove the circuit with, 
    // since you could just use that proving mechanism on the original r1cs instance, witness pair.

    // Step 2: Check that u_n.E == 0 and un u_n.u == 1.
    // Trivial

    // Step 3: Verify the hash conditions:
    //         un.x0 == H(n, z0, zn, Un) and un.x1 == H(U_EC,n).
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      ROConstantsCircuit::<Dual<E>>::default(),
      23 + self.z_0.len() + self.z_i.len(),
    );
    let U_i_hash_bits = hash_U_i::<E, CS>(
      cs, 
      &mut ro, 
      &U_i, 
      &pp_hash, 
      &i, 
      &z_0, 
      &z_i)?;
    let U_i_hash = le_bits_to_num(cs.namespace(|| "bits_to_num"), &U_i_hash_bits)?;

    cs.enforce(
      || "u_i.x[0] == H(i, z_0, z_i, U_i)",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_i_x0.get_variable() - U_i_hash.get_variable(),
    );

    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      ROConstantsCircuit::<Dual<E>>::default(),
      24,
    );
    let cf_U_i_hash_bits = hash_cf_U_i::<E, CS>(
      cs, 
      &mut ro, 
      &cf_U_i, 
      &pp_hash)?;
    let cf_U_i_hash = le_bits_to_num(cs.namespace(|| "bits_to_num"), &cf_U_i_hash_bits)?;
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
    let mut ro_1 = <Dual<E> as Engine>::ROCircuit::new(
      ROConstantsCircuit::<Dual<E>>::default(),
      9,
    );
    let mut ro_2 = <Dual<E> as Engine>::ROCircuit::new(
      ROConstantsCircuit::<Dual<E>>::default(),
      9,
    );
    let (alloc_rw, alloc_re) = KZGChallengesGadget::get_challenges_gadget::<CS, _, E>(
      cs, 
      &mut ro_1, 
      &mut ro_2,
      U_i1)?;

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

    for w in &W_i1_W {
      cs.enforce(
        || "Trivial constraint",
        |lc| lc + w.get_variable(),
        |lc| lc,
        |lc| lc,
      );
    }
    // Step 7.2: Verify that the KZG evaluations are correct:
    for (c, e) in kzg_challenges
      .iter()
      .zip(&kzg_evaluations)
    {
      let eval = EvalGadget::evaluate_gadget::<&mut CS, E>(cs, &W_i1_W, c)?;
      cs.enforce(
        || "evalW == pW(cW)",
        |lc| lc,
        |lc| lc,
        |lc| lc + e.get_variable() - eval.get_variable(),
      );
    }

    Ok(())
  }
}
