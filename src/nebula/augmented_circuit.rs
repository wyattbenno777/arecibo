//! This module defines the Nova augmented circuit used for Cyclefold

use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_FE_IN_EMULATED_POINT, NUM_HASH_BITS},
  gadgets::{
    alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select,
    conditionally_select_vec, emulated::AllocatedEmulPoint, le_bits_to_num,
    AllocatedRelaxedR1CSInstance,
  },
  traits::{
    commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROCircuitTrait,
    ROConstantsCircuit,
  },
  Commitment,
};

use crate::frontend::{
  gadgets::{boolean::Boolean, num::AllocatedNum, Assignment},
  AllocatedBit, ConstraintSystem, SynthesisError,
};
use ff::Field;
use serde::{Deserialize, Serialize};

use crate::cyclefold::{
  gadgets::{emulated, AllocatedCycleFoldData},
  util::FoldingData,
};

use super::rs::StepCircuit;


/// The parameters for the [`AugmentedCircuit`]
///
/// Used for non-native arithmetic
#[derive(Clone, Debug, PartialEq, Copy, Eq, Serialize, Deserialize)]
pub struct AugmentedCircuitParams {
  /// how many bits in each limb
  pub limb_width: usize,
  /// how many limbs represent the BigNat
  pub n_limbs: usize,
}

impl AugmentedCircuitParams {
  /// Create a new instance of the [`AugmentedCircuitParams`]
  pub const fn new(limb_width: usize, n_limbs: usize) -> Self {
    Self {
      limb_width,
      n_limbs,
    }
  }
}

/// Augmented circuit inputs
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AugmentedCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  pp_digest: E1::Base,
  i: E1::Scalar,
  z0: Vec<E1::Scalar>,

  zi: Option<Vec<E1::Scalar>>,
  data_p: Option<FoldingData<E1>>,

  data_c_1: Option<FoldingData<Dual<E1>>>,
  data_c_2: Option<FoldingData<Dual<E1>>>,

  E_new: Option<Commitment<E1>>,
  W_new: Option<Commitment<E1>>,

  prev_IC: Option<E1::Scalar>,
  comm_omega_prev: Option<Commitment<E1>>,
  r_i: Option<E1::Scalar>,
  r_i_next: E1::Scalar,
}

impl<E1> AugmentedCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  /// Create a new augmented circuit inputs
  pub fn new(
    pp_digest: E1::Base,
    i: E1::Scalar,
    z0: Vec<E1::Scalar>,
    zi: Option<Vec<E1::Scalar>>,
    data_p: Option<FoldingData<E1>>,
    data_c_1: Option<FoldingData<Dual<E1>>>,
    data_c_2: Option<FoldingData<Dual<E1>>>,
    E_new: Option<Commitment<E1>>,
    W_new: Option<Commitment<E1>>,
    prev_IC: Option<E1::Scalar>,
    comm_omega_prev: Option<Commitment<E1>>,
    r_i: Option<E1::Scalar>,
    r_i_next: E1::Scalar,
  ) -> Self {
    Self {
      pp_digest,
      i,
      z0,
      zi,
      data_p,
      data_c_1,
      data_c_2,
      E_new,
      W_new,
      prev_IC,
      comm_omega_prev,
      r_i,
      r_i_next,
    }
  }
}

/// Augmented circuit
pub struct AugmentedCircuit<'a, E1, SC>
where
  E1: CurveCycleEquipped,
  SC: StepCircuit<E1::Scalar>,
{
  params: &'a AugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E1>>,
  inputs: Option<AugmentedCircuitInputs<E1>>,
  step_circuit: &'a SC,
}

impl<'a, E1, SC> AugmentedCircuit<'a, E1, SC>
where
  E1: CurveCycleEquipped,
  SC: StepCircuit<E1::Scalar>,
{
  /// Create a new augmented circuit
  pub const fn new(
    params: &'a AugmentedCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E1>>,
    inputs: Option<AugmentedCircuitInputs<E1>>,
    step_circuit: &'a SC,
  ) -> Self {
    Self {
      params,
      ro_consts,
      inputs,
      step_circuit,
    }
  }

  fn alloc_witness<CS: ConstraintSystem<<E1 as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<E1::Scalar>,                     // pp_digest
      AllocatedNum<E1::Scalar>,                     // i
      Vec<AllocatedNum<E1::Scalar>>,                // z0
      Vec<AllocatedNum<E1::Scalar>>,                // zi
      emulated::AllocatedFoldingData<Dual<E1>>,     //data_p
      AllocatedCycleFoldData<Dual<E1>>,             // data_c_1
      AllocatedCycleFoldData<Dual<E1>>,             // data_c_2
      AllocatedEmulPoint<<Dual<E1> as Engine>::GE>, // E_new
      AllocatedEmulPoint<<Dual<E1> as Engine>::GE>, // W_new
      AllocatedNum<E1::Scalar>,                     // prev_IC
      AllocatedEmulPoint<<Dual<E1> as Engine>::GE>, // comm_omega_prev
      AllocatedNum<E1::Scalar>,                     // r_i
      AllocatedNum<E1::Scalar>,                     // r_i_next
    ),
    SynthesisError,
  > {
    let pp_digest = alloc_scalar_as_base::<Dual<E1>, _>(
      cs.namespace(|| "params"),
      self.inputs.as_ref().map(|inputs| inputs.pp_digest),
    )?;

    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    let z_0 = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("z0_{i}")), || {
          Ok(self.inputs.get()?.z0[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E1::Scalar>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero_vec = vec![E1::Scalar::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero_vec)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E1::Scalar>>, _>>()?;

    let data_p = emulated::AllocatedFoldingData::alloc(
      cs.namespace(|| "data_p"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.data_p.as_ref()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let data_c_1 = AllocatedCycleFoldData::alloc(
      cs.namespace(|| "data_c_1"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.data_c_1.as_ref()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let data_c_2 = AllocatedCycleFoldData::alloc(
      cs.namespace(|| "data_c_2"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.data_c_2.as_ref()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let E_new = AllocatedEmulPoint::alloc(
      cs.namespace(|| "E_new"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.E_new.as_ref())
        .map(|E_new| E_new.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let W_new = AllocatedEmulPoint::alloc(
      cs.namespace(|| "W_new"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.W_new.as_ref())
        .map(|W_new| W_new.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let prev_IC = AllocatedNum::alloc(cs.namespace(|| "prev_IC"), || {
      Ok(
        *self
          .inputs
          .get()?
          .prev_IC
          .as_ref()
          .unwrap_or(&E1::Scalar::ZERO),
      )
    })?;

    let comm_omega_prev = AllocatedEmulPoint::alloc(
      cs.namespace(|| "comm_omega_prev"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.comm_omega_prev.as_ref())
        .map(|comm_omega_prev| comm_omega_prev.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let r_i = AllocatedNum::alloc(cs.namespace(|| "r_i"), || {
      Ok(*self.inputs.get()?.r_i.as_ref().unwrap_or(&E1::Scalar::ZERO))
    })?;
    let r_i_next = AllocatedNum::alloc(cs.namespace(|| "r_i_next"), || {
      Ok(self.inputs.get()?.r_i_next)
    })?;

    Ok((
      pp_digest,
      i,
      z_0,
      z_i,
      data_p,
      data_c_1,
      data_c_2,
      E_new,
      W_new,
      prev_IC,
      comm_omega_prev,
      r_i,
      r_i_next,
    ))
  }

  /// Synthesize the base case of the augmented circuit
  pub fn synthesize_base_case<CS: ConstraintSystem<<E1 as Engine>::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<
    (
      AllocatedRelaxedR1CSInstance<Dual<E1>, NIO_CYCLE_FOLD>,
      emulated::AllocatedEmulRelaxedR1CSInstance<Dual<E1>>,
    ),
    SynthesisError,
  > {
    let U_c_default = AllocatedRelaxedR1CSInstance::default(
      cs.namespace(|| "Allocate U_c_default"),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let U_p_default = emulated::AllocatedEmulRelaxedR1CSInstance::default(
      cs.namespace(|| "Allocated U_p_default"),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // In the first folding step return the default relaxed instances for both the CycleFold and
    // primary running accumulators
    Ok((U_c_default, U_p_default))
  }

  /// Synthesize the non-base case of the augmented circuit
  pub fn synthesize_non_base_case<CS: ConstraintSystem<E1::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E1::Scalar>,
    i: &AllocatedNum<E1::Scalar>,
    z_0: &[AllocatedNum<E1::Scalar>],
    z_i: &[AllocatedNum<E1::Scalar>],
    data_p: &emulated::AllocatedFoldingData<Dual<E1>>,
    data_c_1: &AllocatedCycleFoldData<Dual<E1>>,
    data_c_2: &AllocatedCycleFoldData<Dual<E1>>,
    E_new: AllocatedEmulPoint<<Dual<E1> as Engine>::GE>,
    W_new: AllocatedEmulPoint<<Dual<E1> as Engine>::GE>,
    arity: usize,
    prev_IC: &AllocatedNum<E1::Scalar>,
    r_i: &AllocatedNum<E1::Scalar>,
  ) -> Result<
    (
      AllocatedRelaxedR1CSInstance<Dual<E1>, NIO_CYCLE_FOLD>,
      emulated::AllocatedEmulRelaxedR1CSInstance<Dual<E1>>,
      AllocatedBit,
    ),
    SynthesisError,
  > {
    /*
     * check that u_i.x = hash(vk, i, z0, zi, Ui)
     */

    // Check u_i.x0
    let mut ro_p = <Dual<E1> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      4 + 2 * arity + 2 * NUM_FE_IN_EMULATED_POINT + 3,
    );
    ro_p.absorb(pp_digest);
    ro_p.absorb(i);
    for e in z_0 {
      ro_p.absorb(e)
    }
    for e in z_i {
      ro_p.absorb(e)
    }
    data_p
      .U
      .absorb_in_ro(cs.namespace(|| "absorb U_p"), &mut ro_p)?;
    ro_p.absorb(prev_IC);
    ro_p.absorb(r_i);
    let hash_bits_p = ro_p.squeeze(cs.namespace(|| "primary hash bits"), NUM_HASH_BITS)?;
    let hash_p = le_bits_to_num(cs.namespace(|| "primary hash"), &hash_bits_p)?;
    let check_primary = alloc_num_equals(
      cs.namespace(|| "u.X[0] = H(params, i, z0, zi, U_p)"),
      &data_p.u_x0,
      &hash_p,
    )?;

    // Check u_i.x1
    let mut ro_c = <Dual<E1> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      1 + 1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS, // r_i + pp + i + W + E + u + X
    );
    ro_c.absorb(pp_digest);
    ro_c.absorb(i);
    data_c_1
      .U
      .absorb_in_ro(cs.namespace(|| "absorb U_c"), &mut ro_c)?;
    ro_c.absorb(r_i);
    let hash_c_bits = ro_c.squeeze(cs.namespace(|| "cyclefold hash bits"), NUM_HASH_BITS)?;
    let hash_c = le_bits_to_num(cs.namespace(|| "cyclefold hash"), &hash_c_bits)?;
    let check_cyclefold = alloc_num_equals(
      cs.namespace(|| "u.X[1] = H(params, U_c)"),
      &data_p.u_x1,
      &hash_c,
    )?;

    // Check for u_i.x0 && u_i.x1
    let check_io = AllocatedBit::and(
      cs.namespace(|| "both IOs match"),
      &check_primary,
      &check_cyclefold,
    )?;

    /*
     * Nova-CycleFold verifier circuit
     */

    // Primary NIFS.V
    let U_p = data_p.U.fold_with_r1cs(
      cs.namespace(|| "fold u_p into U_p"),
      pp_digest,
      W_new,
      E_new,
      &data_p.u_W,
      &data_p.u_x0,
      &data_p.u_x1,
      &data_p.T,
      self.ro_consts.clone(),
    )?;

    // Run NIVC.V on U_c, u_c_1, T_c_1
    let U_int = data_c_1.apply_fold(
      cs.namespace(|| "fold u_c_1 into U_c"),
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    /*
     *Check the intermediate-calculated running instance matches the non-deterministic advice provided to the prover
     */
    // Calculate h_int = H(pp, U_c_int)
    let mut ro_c_int = <Dual<E1> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS, // pp + W + E + u + X
    );
    ro_c_int.absorb(pp_digest);
    U_int.absorb_in_ro(cs.namespace(|| "absorb U_c_int"), &mut ro_c_int)?;
    let h_c_int_bits =
      ro_c_int.squeeze(cs.namespace(|| "intermediate hash bits"), NUM_HASH_BITS)?;
    let h_c_int = le_bits_to_num(cs.namespace(|| "intermediate hash"), &h_c_int_bits)?;

    // Calculate h_1 = H(pp, U_c_1)
    let mut ro_c_1 = <Dual<E1> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS, // pp + W + E + u + X
    );
    ro_c_1.absorb(pp_digest);
    data_c_2
      .U
      .absorb_in_ro(cs.namespace(|| "absorb U_c_1"), &mut ro_c_1)?;
    let h_c_1_bits = ro_c_1.squeeze(cs.namespace(|| "cyclefold_1 hash bits"), NUM_HASH_BITS)?;
    let h_c_1 = le_bits_to_num(cs.namespace(|| "cyclefold_1 hash"), &h_c_1_bits)?;
    let check_cyclefold_int = alloc_num_equals(cs.namespace(|| "h_int = h_c_1"), &h_c_int, &h_c_1)?;

    // CycleFold NIFS.V
    let U_c = data_c_2.apply_fold(
      cs.namespace(|| "fold u_c_2 into U_c_1"),
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    let checks_pass = AllocatedBit::and(
      cs.namespace(|| "all checks passed"),
      &check_io,
      &check_cyclefold_int,
    )?;

    Ok((U_c, U_p, checks_pass))
  }

  /// Synthesize the augmented circuit
  pub fn synthesize<CS: ConstraintSystem<E1::Scalar>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E1::Scalar>>, SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate the witness
    let (
      pp_digest,
      i,
      z_0,
      z_i,
      data_p,
      data_c_1,
      data_c_2,
      E_new,
      W_new,
      prev_IC,
      comm_omega_prev,
      r_i,
      r_i_next,
    ) = self.alloc_witness(cs.namespace(|| "alloc_witness"), arity)?;

    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "is base case"), &i, &zero)?;

    //  If i=0:
    //
    //  Set Ui+1 ← (u⊥,...,u⊥)
    let (Unew_c_base, Unew_p_base) = self.synthesize_base_case(cs.namespace(|| "base case"))?;

    // Otherwise:
    //
    // compute Ui+1 ← NIFS.V(vk, U, u, T ),
    let (Unew_c_non_base, Unew_p_non_base, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      &pp_digest,
      &i,
      &z_0,
      &z_i,
      &data_p,
      &data_c_1,
      &data_c_2,
      E_new,
      W_new,
      arity,
      &prev_IC,
      &r_i,
    )?;

    // Check that u references U in the output of the prior iteration
    let should_be_false = AllocatedBit::nor(
      cs.namespace(|| "check_non_base_pass nor base_case"),
      &check_non_base_pass,
      &is_base_case,
    )?;
    cs.enforce(
      || "check_non_base_pass nor base_case = false",
      |lc| lc + should_be_false.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    // select the new running primary instance
    let Unew_p = Unew_p_base.conditionally_select(
      cs.namespace(|| "compute Unew_p"),
      &Unew_p_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // select the new running CycleFold instance
    let Unew_c = Unew_c_base.conditionally_select(
      cs.namespace(|| "compute Unew_c"),
      &Unew_c_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + E1::Scalar::ONE)
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
    );

    // Compute z_{i+1}
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute the next output zi+1 ← Fj(zi,ωi).
    let z_next = self
      .step_circuit
      .synthesize(&mut cs.namespace(|| "F"), &z_input)?;

    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }

    // compute incremental commitment
    let IC_i_base_case =
      alloc_scalar_as_base::<Dual<E1>, _>(cs.namespace(|| "IC_0"), Some(E1::Base::ZERO))?;

    let IC_i_non_base_case = {
      let mut ro = <Dual<E1> as Engine>::ROCircuit::new(
        self.ro_consts.clone(),
        1 + NUM_FE_IN_EMULATED_POINT, // IC_prev + comm_omega_prev
      );

      ro.absorb(&prev_IC);
      comm_omega_prev.absorb_in_ro(cs.namespace(|| "absorb comm_omega_prev"), &mut ro)?;

      let hash_IC_bits = ro.squeeze(cs.namespace(|| "hash_IC_bits"), NUM_HASH_BITS)?;
      le_bits_to_num(cs.namespace(|| "hash_IC"), &hash_IC_bits)?
    };

    let IC_i = conditionally_select(
      cs.namespace(|| "select IC"),
      &IC_i_base_case,
      &IC_i_non_base_case,
      &Boolean::from(is_base_case),
    )?;

    // Calculate the first component of the public IO as the hash of the calculated primary running
    // instance
    let mut ro_p = <Dual<E1> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      4 + 2 * arity + (2 * NUM_FE_IN_EMULATED_POINT + 3), // pp + IC + i + z_0 + z_next + (U_p)
    );
    ro_p.absorb(&pp_digest);
    ro_p.absorb(&i_new);
    for e in &z_0 {
      ro_p.absorb(e);
    }
    for e in &z_next {
      ro_p.absorb(e);
    }
    Unew_p.absorb_in_ro(cs.namespace(|| "absorb Unew_p"), &mut ro_p)?;
    ro_p.absorb(&IC_i);
    ro_p.absorb(&r_i_next);

    let hash_p_bits = ro_p.squeeze(cs.namespace(|| "hash_p_bits"), NUM_HASH_BITS)?;
    let hash_p = le_bits_to_num(cs.namespace(|| "hash_p"), &hash_p_bits)?;

    // Calculate the second component of the public IO as the hash of the calculated CycleFold running
    // instance
    let mut ro_c = <Dual<E1> as Engine>::ROCircuit::new(
      self.ro_consts,
      1 + 1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS, // r_i + pp + i + W + E + u + X
    );
    ro_c.absorb(&pp_digest);
    ro_c.absorb(&i_new);
    Unew_c.absorb_in_ro(cs.namespace(|| "absorb Unew_c"), &mut ro_c)?;
    ro_c.absorb(&r_i_next);
    let hash_c_bits = ro_c.squeeze(cs.namespace(|| "hash_c_bits"), NUM_HASH_BITS)?;
    let hash_c = le_bits_to_num(cs.namespace(|| "hash_c"), &hash_c_bits)?;
    hash_p.inputize(cs.namespace(|| "u_p.x[0] = hash_p"))?;
    hash_c.inputize(cs.namespace(|| "u_p.x[1] = hash_c"))?;

    Ok(z_next)
  }
}
