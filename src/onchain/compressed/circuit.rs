//! Implements the compressedSNARK circuit.

use crate::onchain::gadgets::{EvalGadget, FoldGadget, KZGChallengesGadget};
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  cyclefold::gadgets::emulated::{
    AllocatedEmulR1CSInstance, AllocatedEmulRelaxedR1CSInstance,
  },
  errors::NovaError,
  frontend::{
    gpu::GpuName, num::AllocatedNum, Circuit, ConstraintSystem,
    SynthesisError,
  },
  gadgets::{
    AllocatedRelaxedR1CSInstance,
    emulated::AllocatedEmulPoint,
  },
  nebula::{
    nifs::NIFS,
    rs::{PublicParams, RecursiveSNARK},
  },
  onchain::gadgets::hash::{hash_U_i, hash_cf_U_i},
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    commitment::{CommitmentTrait, CommitmentEngineTrait},
    CurveCycleEquipped, Dual, Engine, ROConstants,
  },
  CommitmentKey,
};
use std::sync::Arc;

/// Verifier circuit for the CompressedSNARK.
#[derive(Debug, Clone)]
pub struct VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  /// Constraint system of the Augmented Function circuit
  pub arith: R1CSShape<E>,
  /// R1CS of the CycleFold circuit
  pub cf_arith: R1CSShape<Dual<E>>,
  /// RO constants
  pub ro_consts: ROConstants<E>,
  /// public params hash
  pub pp_hash: E::Scalar,
  /// current index
  pub i: usize,
  /// previous IC
  pub prev_IC: E::Scalar,
  /// Randomness
  pub r_i: E::Scalar,
  /// initial state
  pub z_0: Vec<E::Scalar>,
  /// current i-th state
  pub z_i: Vec<E::Scalar>,
  /// Primary relaxed R1CS instance
  pub U_i: RelaxedR1CSInstance<E>,
  /// Primary relaxed R1CS witness
  pub W_i: RelaxedR1CSWitness<E>,
  /// Primary R1CS instance
  pub u_i: R1CSInstance<E>,
  /// Primary R1CS witness
  pub w_i: R1CSWitness<E>,
  /// Next relaxed R1CS instance
  pub U_i1: RelaxedR1CSInstance<E>,
  /// Next relaxed R1CS witness
  pub W_i1: RelaxedR1CSWitness<E>,
  /// NIFS proof
  pub nifs_proof: NIFS<E>,
  /// Randomness
  pub randomness: E::Scalar,
  /// CycleFold running instance
  pub cf_U_i: RelaxedR1CSInstance<Dual<E>>,
  /// CycleFold running witness
  pub cf_W_i: RelaxedR1CSWitness<Dual<E>>,
  /// CycleFold commitment key      
  pub cf_ck: Arc<CommitmentKey<Dual<E>>>,
  /// KZG challenges
  pub kzg_challenges: (E::Scalar, E::Scalar),
  /// KZG evaluations
  pub kzg_evaluations: (E::Scalar, E::Scalar),
}

impl<E> VerifierCircuit<E>
where
  E: CurveCycleEquipped,
  <E as Engine>::Scalar: GpuName,
{
  /// Default constructor.
  pub fn default(
    arith: &R1CSShape<E>,
    cf_arith: &R1CSShape<Dual<E>>,
    ro_consts: ROConstants<E>,
    pp_hash: <E as Engine>::Scalar,
    state_len: usize,
    (ck, ck_secondary): (&CommitmentKey<E>, &CommitmentKey<Dual<E>>),
  ) -> Self {
    Self {
      arith: arith.clone(),
      cf_arith: cf_arith.clone(),
      ro_consts,
      pp_hash,
      i: 0,
      prev_IC: <E as Engine>::Scalar::from(0),
      r_i: <E as Engine>::Scalar::from(0),
      z_0: vec![<E as Engine>::Scalar::from(0); state_len],
      z_i: vec![<E as Engine>::Scalar::from(0); state_len],
      U_i: RelaxedR1CSInstance::default(ck, arith),
      W_i: RelaxedR1CSWitness::default(arith),
      u_i: R1CSInstance::<E>::default(arith),
      w_i: R1CSWitness::<E>::default(arith),
      U_i1: RelaxedR1CSInstance::default(ck, arith),
      W_i1: RelaxedR1CSWitness::default(arith),
      nifs_proof: NIFS::default(cf_arith),
      randomness: <E as Engine>::Scalar::from(0),
      cf_U_i: RelaxedR1CSInstance::default(ck_secondary, cf_arith),
      cf_W_i: RelaxedR1CSWitness::default(cf_arith),
      cf_ck: ck_secondary.clone().into(),
      kzg_challenges: (
        <E as Engine>::Scalar::from(0),
        <E as Engine>::Scalar::from(0),
      ),
      kzg_evaluations: (
        <E as Engine>::Scalar::from(0),
        <E as Engine>::Scalar::from(0),
      ),
    }
  }

  /// Constructor from public parameters and recursive SNARK.
  pub fn new(pp: &PublicParams<E>, rs: RecursiveSNARK<E>) -> Result<Self, NovaError> {
    let ro_consts = ROConstants::<E>::default(); // TODO: Not sure if this is OK

    // TODO: Do I need to run an iteration for IS and FS in Nebula?
    // 1. Compute the U_{i+1}, W_{i+1}
    let (nifs, (r_U_primary, r_W_primary), _, rho, _U_secondary_temp) =
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

    let (rw, re) = KZGChallengesGadget::get_challenges_native(r_U_primary.clone());
    let rw_eval = EvalGadget::evaluate_native(r_W_primary.clone().W, rw);
    let re_eval = EvalGadget::evaluate_native(r_W_primary.clone().E, re);

    Ok(Self {
      arith: pp.circuit_shape_primary.r1cs_shape.clone(),
      cf_arith: pp.circuit_shape_cyclefold.r1cs_shape.clone(),
      ro_consts,
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
      cf_ck: pp.ck_cyclefold.clone(),
      cf_U_i: rs.r_U_cyclefold,
      cf_W_i: rs.r_W_cyclefold,
      kzg_challenges: (rw, re),
      kzg_evaluations: (rw_eval, re_eval),
      randomness: rho,
      nifs_proof: nifs,
    })
  }
}

impl<E> Circuit<E::Scalar> for VerifierCircuit<E>
where
  E: CurveCycleEquipped,
  E::Scalar: GpuName,
{
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
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

    let U_i1: AllocatedEmulRelaxedR1CSInstance<Dual<E>> = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "U_i1"),
      Some(&self.U_i1),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let (U_i1_cmW_x, U_i1_cmW_y, _) = U_i1.comm_W.to_coordinates();

    for (i, limb) in U_i1_cmW_x.as_limbs().iter().enumerate() {
      let tmp = limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of x to num")))?;
      tmp.inputize(cs.namespace(|| format!("convert limb {i} of x to num")))?;
    }

    for (i, limb) in U_i1_cmW_y.as_limbs().iter().enumerate() {
      let tmp = limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of y to num")))?;
      tmp.inputize(cs.namespace(|| format!("convert limb {i} of y to num")))?;
    }

    let (U_i1_cmE_x, U_i1_cmE_y, _) = U_i1.comm_E.to_coordinates();
    for (i, limb) in U_i1_cmE_x.as_limbs().iter().enumerate() {
      let tmp = limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of x to num")))?;
      tmp.inputize(cs.namespace(|| format!("convert limb {i} of x to num")))?;
    }

    for (i, limb) in U_i1_cmE_y.as_limbs().iter().enumerate() {
      let tmp = limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of y to num")))?;
      tmp.inputize(cs.namespace(|| format!("convert limb {i} of y to num")))?;
    }

    // --------------------------------------------------------------------------------------------
    // Step 1: Enforce U_{n+1} and W_{n+1} satisfy r1cs
    // --------------------------------------------------------------------------------------------
    // Nova has no need for this, since we are checking if an r1cs relation
    // is sat inside r1cs thus creating another r1cs relation you would have to check is sat.

    // --------------------------------------------------------------------------------------------
    // Step 2: Check that u_n.E == 0 and un u_n.u == 1.
    // --------------------------------------------------------------------------------------------
    // Trivial, since we are using a r1cs relation.

    // --------------------------------------------------------------------------------------------
    // Step 3: Verify the hash conditions:
    //         un.x0 == H(n, z0, zn, Un)  
    //         un.x1 == H(U_EC,n).
    // --------------------------------------------------------------------------------------------
    let U_i_hash = hash_U_i::<E, CS>(cs, &U_i, &pp_hash, &i, &z_0, &z_i, &prev_IC, &r_i)?;

    cs.enforce(
      || "u_i.x[0] == H(i, z_0, z_i, U_i)",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_i_x0.get_variable() - U_i_hash.get_variable(),
    );

    let u_i_x1 = AllocatedNum::alloc(cs.namespace(|| "allocate x1"), || Ok(self.u_i.X[1]))?;
    let cf_U_i: AllocatedRelaxedR1CSInstance<Dual<E>, BN_N_LIMBS> =
      AllocatedRelaxedR1CSInstance::alloc(
        cs.namespace(|| "cf_U_i"),
        Some(&self.cf_U_i),
        BN_LIMB_WIDTH,
        BN_N_LIMBS,
      )?;

    let cf_U_i_hash = hash_cf_U_i::<E, CS>(cs, &cf_U_i, &pp_hash, &i, &r_i)?;

    cs.enforce(
      || "u_i.x[1] == H(U_EC, i)",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_i_x1.get_variable() - cf_U_i_hash.get_variable(),
    );

    // --------------------------------------------------------------------------------------------
    // Step 4: Commitments verification for U_{EC,n}.{E, W} with respect to W_{EC,n}.{E, W}.
    // --------------------------------------------------------------------------------------------
    #[cfg(not(feature = "light_onchain_prover"))]
    {
      let cf_W_i_commit = <Dual<E> as Engine>::CE::commit_gadget(
        cs,
        &*self.cf_ck,
        &self.cf_W_i.W[..],
        &self.cf_W_i.r_W,
      )?;

      // Check that Commit(cf_W_i.W) == cf_U_i.cmW
      cf_W_i_commit.check_equal(
        cs.namespace(|| "check that cf_W_i.W == cf_U_i.cmW"),
        &cf_U_i.W,
      )?;
    }

    // -------------------------------------------------------------------------------------------- 
    // Step 5: Enforce U_{EC,n} and W_{EC,n} satisfy r1cs_{EC}, the Relaxed R1CS relation of the CycleFoldCircuit.
    // --------------------------------------------------------------------------------------------

    // Step 6.1: Partially enforce that U_{n+1} is the correct folding of U_n and un.
    let u_i = AllocatedEmulR1CSInstance::alloc(
      cs.namespace(|| "u_i"),
      Some(&self.u_i),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let r = AllocatedNum::alloc(cs.namespace(|| "get r"), || Ok(self.randomness))?;
    let alloc_fold_U_i1 = FoldGadget::fold_field_elements_gadget::<_, E>(
      cs,
      pp_hash,
      U_i,
      u_i,
      self.nifs_proof.clone(),
      r,
    )?;

    cs.enforce(
      || "U_i1.u == U_i.u",
      |lc| lc,
      |lc| lc,
      |lc| lc + U_i1.u.get_variable() - alloc_fold_U_i1.u.get_variable(),
    );

    cs.enforce(
      || "U_i1.x0 == U_i.x0",
      |lc| lc,
      |lc| lc,
      |lc| lc + U_i1.x0.get_variable() - alloc_fold_U_i1.x0.get_variable(),
    );

    cs.enforce(
      || "U_i1.x1 == U_i.x1",
      |lc| lc,
      |lc| lc,
      |lc| lc + U_i1.x1.get_variable() - alloc_fold_U_i1.x1.get_variable(),
    );

    // --------------------------------------------------------------------------------------------
    // Step 7.1: Check correct computation of the KZG challenges:
    //           cE ≡ H(E.{x, y}), cW ≡ H(W.{x, y}).
    // --------------------------------------------------------------------------------------------
    let kzg_alloc_rw = AllocatedNum::alloc(cs.namespace(|| "get kzg_challenges rw"), || {
      Ok(self.kzg_challenges.0)
    })?;
    kzg_alloc_rw.inputize(cs.namespace(|| "kzg_alloc_rw"))?;
    let kzg_alloc_re = AllocatedNum::alloc(cs.namespace(|| "get kzg_challenges re"), || {
      Ok(self.kzg_challenges.1)
    })?;
    kzg_alloc_re.inputize(cs.namespace(|| "kzg_alloc_re"))?;

    let (alloc_rw, alloc_re) = KZGChallengesGadget::get_challenges_gadget::<CS, E>(cs, U_i1)?;

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

    // --------------------------------------------------------------------------------------------
    // Step 7.2: Verify that the KZG evaluations are correct
    // --------------------------------------------------------------------------------------------
    let kzg_alloc_rw_eval = AllocatedNum::alloc(cs.namespace(|| "get kzg_evaluations"), || {
      Ok(self.kzg_evaluations.0)
    })?;
    kzg_alloc_rw_eval.inputize(cs.namespace(|| "kzg_alloc_rw_eval"))?;
    let kzg_alloc_re_eval = AllocatedNum::alloc(cs.namespace(|| "get kzg_evaluations"), || {
      Ok(self.kzg_evaluations.1)
    })?;
    kzg_alloc_re_eval.inputize(cs.namespace(|| "kzg_alloc_re_eval"))?;

    let W_i1_W = self
      .W_i1
      .W
      .iter()
      .map(|x| AllocatedNum::alloc(cs.namespace(|| "allocate W_i1.W"), || Ok(*x)))
      .collect::<Result<Vec<_>, _>>()?;

    let W_i1_E = self
      .W_i1
      .E
      .iter()
      .map(|x| AllocatedNum::alloc(cs.namespace(|| "allocate W_i1.E"), || Ok(*x)))
      .collect::<Result<Vec<_>, _>>()?;

    let alloc_rw_eval = EvalGadget::evaluate_gadget::<&mut CS, E>(cs, W_i1_W, &kzg_alloc_rw)?;
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

    // ------------------------------
    // Inputize the cross term cmT
    // ------------------------------

    let alloc_cmT: AllocatedEmulPoint<<Dual<E> as Engine>::GE> = AllocatedEmulPoint::alloc(
      cs.namespace(|| "alloc_cmT"),
      Some(self.nifs_proof.nifs_primary.comm_T.to_coordinates()),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;
    let (cmT_x, cmT_y, cmT_id) = alloc_cmT.to_coordinates();
    for (i, limb) in cmT_x.as_limbs().iter().enumerate() {
      let tmp = limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of x to num")))?;
      tmp.inputize(cs.namespace(|| format!("convert limb {i} of x to num")))?;
    }

    for (i, limb) in cmT_y.as_limbs().iter().enumerate() {
      let tmp = limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of y to num")))?;
      tmp.inputize(cs.namespace(|| format!("convert limb {i} of y to num")))?;
    }

    cs.enforce(
      || "dummy constraint",
      |lc| lc + cmT_id.get_variable(),
      |lc| lc,
      |lc| lc,
    );

    Ok(())
  }
}
