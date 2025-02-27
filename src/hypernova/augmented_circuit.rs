use super::{nifs::NIFS, rs::StepCircuit};
use crate::{
  and_then_field,
  constants::{DEFAULT_ABSORBS, NIO_CYCLE_FOLD, NUM_HASH_BITS},
  frontend::{
    gadgets::Assignment, num::AllocatedNum, shape_cs::ShapeCS, AllocatedBit, Boolean,
    ConstraintSystem, SynthesisError,
  },
  gadgets::{
    alloc_num_equals, alloc_tuple, alloc_tuple_comms, alloc_zero, conditionally_select,
    conditionally_select_vec,
    emulated::AllocatedEmulPoint,
    hypernova::{
      alloc_sized_vec, increment, AllocatedLR1CSInstance, AllocatedNIFS, AllocatedSplitR1CSInstance,
    },
    le_bits_to_num, AllocatedRelaxedR1CSInstance,
  },
  map_field,
  r1cs::{
    split::{LR1CSInstance, SplitR1CSInstance},
    RelaxedR1CSInstance,
  },
  spartan::math::Math,
  traits::{
    commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROCircuitTrait,
    ROConstantsCircuit,
  },
  AugmentedCircuitParams, Commitment,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct AugmentedCircuit<'a, E, SC>
where
  SC: StepCircuit<E::Scalar>,
  E: CurveCycleEquipped,
{
  step_circuit: &'a SC,
  params: &'a AugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E>>,
  inputs: Option<AugmentedCircuitInputs<E>>,
  num_rounds: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AugmentedCircuitInputs<E>
where
  E: CurveCycleEquipped,
{
  pp_digest: E::Scalar,
  i: E::Scalar,
  z_0: Vec<E::Scalar>,
  z_i: Option<Vec<E::Scalar>>,
  nifs: Option<NIFS<E>>,
  U: Option<LR1CSInstance<E>>,
  u: Option<SplitR1CSInstance<E>>,
  W_new: Option<Commitment<E>>,
  U_cyclefold: Option<RelaxedR1CSInstance<Dual<E>>>,
  pre_committed: Option<(Commitment<E>, Commitment<E>)>,
  prev_IC: Option<(E::Scalar, E::Scalar)>,
}

impl<E> AugmentedCircuitInputs<E>
where
  E: CurveCycleEquipped,
{
  pub fn new(
    pp_digest: E::Scalar,
    i: E::Scalar,
    z_0: Vec<E::Scalar>,
    z_i: Option<Vec<E::Scalar>>,
    nifs: Option<NIFS<E>>,
    U: Option<LR1CSInstance<E>>,
    u: Option<SplitR1CSInstance<E>>,
    W_new: Option<Commitment<E>>,
    U_cyclefold: Option<RelaxedR1CSInstance<Dual<E>>>,
    pre_committed: Option<(Commitment<E>, Commitment<E>)>,
    prev_IC: Option<(E::Scalar, E::Scalar)>,
  ) -> Self {
    Self {
      pp_digest,
      i,
      z_0,
      z_i,
      nifs,
      U,
      u,
      W_new,
      U_cyclefold,
      pre_committed,
      prev_IC,
    }
  }
}

impl<'a, E, SC> AugmentedCircuit<'a, E, SC>
where
  E: CurveCycleEquipped,
  SC: StepCircuit<E::Scalar>,
{
  pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    // Allocate the witness
    let arity = self.step_circuit.arity();
    let (pp_digest, i, z_0, z_i, nifs, U, u, W_new, U_cyclefold, pre_committed, prev_IC) =
      self.alloc_witness(cs.namespace(|| "alloc_witness"), arity)?;

    // --- Base case: i = 0 ---
    //
    // 1. Check if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "is base case"), &i, &zero)?;
    // 2. Get the default running instance.
    let (U_default, U_cyclefold_default) =
      self.synthesize_base_case(cs.namespace(|| "base case"))?;

    // --- Non-base case: i > 0 ---
    //
    // Compute Hash check and U <- NIFS.V
    let (U_non_base_case, U_cyclefold_non_base_case, check_non_base_pass) = self
      .synthesize_non_base_case(
        cs.namespace(|| "non base case"),
        &pp_digest,
        &i,
        &z_0,
        &z_i,
        &self.ro_consts,
        &nifs,
        &U,
        &u,
        W_new,
        &U_cyclefold,
        pre_committed,
        (&prev_IC.0, &prev_IC.1),
      )?;

    // --- Check that u references U in the output of the prior iteration ---
    //
    // Hash check: u.X[0] = H(pp, i, z_0, z_i, U) && u.X[1] = H(pp, i, U_cyclefold)
    self.enforce_hash_check(
      cs.namespace(|| "enforce_hash_check"),
      &check_non_base_pass,
      &is_base_case,
    )?;

    // --- Select the new running instances. ---
    //
    // 1. Select the new U based on whether this is the base case
    let U_new = U_default.conditionally_select(
      cs.namespace(|| "compute U_new"),
      &U_non_base_case,
      &Boolean::from(is_base_case.clone()),
    )?;
    // 2. Select the new U_cyclefold based on whether this is the base case
    let U_new_cyclefold = U_cyclefold_default.conditionally_select(
      cs.namespace(|| "compute U_new_cyclefold"),
      &U_cyclefold_non_base_case,
      &Boolean::from(is_base_case.clone()),
    )?;

    // --- Synthesize the step circuit (F) and compute the next output. ---
    //
    // 1.  Select the z input based on whether this is the base case
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case.clone()),
    )?;
    // 2. Compute the next output z_next ← F(z_input)
    let z_next = self
      .step_circuit
      .synthesize(&mut cs.namespace(|| "F"), &z_input)?;
    // 3. Check step_circuit_i (F_i) conforms to structure F
    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }
    // 4. Compute i++
    let i_new = increment(cs.namespace(|| "i++"), &i)?;

    // If i = 0 then C_i ← ⊥, else C_i ← hash(C_i−1, C_ωi−1)
    let IC = self.increment_ic(
      cs.namespace(|| "increment IC"),
      prev_IC,
      (&u.pre_committed.0, &u.pre_committed.1),
      &is_base_case,
    )?;

    // --- Output hash ---
    //
    // 1. u.X[0] = H(pp, i, z_0, z_i, U)
    let hash = self.calculate_hash(
      cs.namespace(|| "calculate_hash"),
      &pp_digest,
      &i_new,
      &z_0,
      &z_next,
      &U_new,
      (&IC.0, &IC.1),
    )?;
    hash.inputize(cs.namespace(|| "u.x[0] = hash"))?;
    // 2. Calculate the second component of the public IO as the hash of the
    //    calculated CycleFold running instance
    //
    //    u.X[1] = H(pp, i, U_cyclefold)
    let hash_cyclefold = self.calculate_hash_cyclefold(
      cs.namespace(|| "calculate_hash_cyclefold"),
      &pp_digest,
      &i_new,
      &U_new_cyclefold,
    )?;
    hash_cyclefold.inputize(cs.namespace(|| "u.x[1] = hash_cyclefold"))?;
    Ok(z_next)
  }

  pub fn synthesize_non_base_case<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    z_0: &[AllocatedNum<E::Scalar>],
    z_i: &[AllocatedNum<E::Scalar>],
    ro_consts: &ROConstantsCircuit<Dual<E>>,
    nifs: &AllocatedNIFS<E>,
    U: &AllocatedLR1CSInstance<E>,
    u: &AllocatedSplitR1CSInstance<E>,
    W_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    U_cyclefold: &AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
    pre_committed: (
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    ),
    prev_IC: (&AllocatedNum<E::Scalar>, &AllocatedNum<E::Scalar>),
  ) -> Result<
    (
      AllocatedLR1CSInstance<E>,
      AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
      AllocatedBit,
    ),
    SynthesisError,
  > {
    // Hash check: u.X[0] = H(pp, i, z_0, z_i, U)
    //             u.X[1] = H(pp, i, U_cyclefold)
    let io_check = self.io_check(
      cs.namespace(|| "io_check"),
      pp_digest,
      i,
      z_0,
      z_i,
      U,
      u,
      U_cyclefold,
      prev_IC,
    )?;

    // # NIFS.V
    //
    // Compute folded U and U_cyclefold
    let (U, U_cyclefold) = nifs.verify(
      cs.namespace(|| "NIFS.V"),
      pp_digest,
      ro_consts,
      U,
      u,
      W_new,
      pre_committed,
      U_cyclefold,
      self.num_rounds,
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    Ok((U, U_cyclefold, io_check))
  }

  pub fn synthesize_base_case<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<
    (
      AllocatedLR1CSInstance<E>,
      AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
    ),
    SynthesisError,
  > {
    let U_default = AllocatedLR1CSInstance::default(
      cs.namespace(|| "Allocated U_default"),
      self.params.limb_width,
      self.params.n_limbs,
      self.num_rounds,
    )?;
    let U_cyclefold_default = AllocatedRelaxedR1CSInstance::default(
      cs.namespace(|| "Allocate U_c_default"),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    Ok((U_default, U_cyclefold_default))
  }

  fn alloc_witness<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,                               // pp_digest
      AllocatedNum<E::Scalar>,                               // i
      Vec<AllocatedNum<E::Scalar>>,                          // z_0
      Vec<AllocatedNum<E::Scalar>>,                          // z_i
      AllocatedNIFS<E>,                                      // nifs
      AllocatedLR1CSInstance<E>,                             // U
      AllocatedSplitR1CSInstance<E>,                         // u
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,           // W_new
      AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>, // U_cyclefold
      (
        AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
        AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
      ), // pre_committed
      (AllocatedNum<E::Scalar>, AllocatedNum<E::Scalar>),    // prev_IC
    ),
    SynthesisError,
  > {
    // Allocate primitives: pp_digest, i, z_0
    let pp_digest = AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || {
      Ok(self.inputs.get()?.pp_digest)
    })?;
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;
    let z_0 = alloc_sized_vec(
      cs.namespace(|| "z_0"),
      map_field!(self.inputs, ref, z_0),
      arity,
    )?;

    // Allocate z_i. If inputs.z_i is not provided (base case) allocate default value 0
    let z_i = alloc_sized_vec(
      cs.namespace(|| "z_i"),
      and_then_field!(self.inputs, z_i),
      arity,
    )?;

    // Allocate primary folding data
    let nifs = AllocatedNIFS::alloc(
      cs.namespace(|| "nifs"),
      and_then_field!(self.inputs, nifs),
      self.num_rounds,
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let U = AllocatedLR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      and_then_field!(self.inputs, U),
      self.params.limb_width,
      self.params.n_limbs,
      self.num_rounds,
    )?;
    let u = AllocatedSplitR1CSInstance::alloc(
      cs.namespace(|| "allocate u"),
      and_then_field!(self.inputs, u),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let W_new = AllocatedEmulPoint::alloc(
      cs.namespace(|| "allocate W_new"),
      and_then_field!(self.inputs, W_new).map(|W_new| W_new.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let U_cyclefold = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U_cyclefold"),
      and_then_field!(self.inputs, U_cyclefold),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let pre_committed = alloc_tuple_comms::<_, E>(
      cs.namespace(|| "pre_committed"),
      self.inputs.as_ref().and_then(|inputs| inputs.pre_committed),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let prev_IC = alloc_tuple(
      cs.namespace(|| "prev_IC"),
      self.inputs.as_ref().and_then(|inputs| inputs.prev_IC),
    )?;
    Ok((
      pp_digest,
      i,
      z_0,
      z_i,
      nifs,
      U,
      u,
      W_new,
      U_cyclefold,
      pre_committed,
      prev_IC,
    ))
  }

  pub fn io_check<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    z_0: &[AllocatedNum<E::Scalar>],
    z_i: &[AllocatedNum<E::Scalar>],
    U: &AllocatedLR1CSInstance<E>,
    u: &AllocatedSplitR1CSInstance<E>,
    U_cyclefold: &AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
    prev_IC: (&AllocatedNum<E::Scalar>, &AllocatedNum<E::Scalar>),
  ) -> Result<AllocatedBit, SynthesisError> {
    // Hash check: u.X[0] = H(pp, i, z_0, z_i, U)
    let hash_check = self.hash_check(
      cs.namespace(|| "hash_check"),
      pp_digest,
      i,
      z_0,
      z_i,
      U,
      u,
      prev_IC,
    )?;

    // Hash check: u.X[1] = H(pp, i, U_cyclefold)
    let hash_check_cyclefold = self.hash_check_cyclefold(
      cs.namespace(|| "hash_check_cyclefold"),
      pp_digest,
      i,
      U_cyclefold,
      u,
    )?;

    // Check for u_i.x0 && u_i.x1
    let io_check = AllocatedBit::and(
      cs.namespace(|| "both IOs match"),
      &hash_check,
      &hash_check_cyclefold,
    )?;
    Ok(io_check)
  }

  pub fn hash_check<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    z_0: &[AllocatedNum<E::Scalar>],
    z_i: &[AllocatedNum<E::Scalar>],
    U: &AllocatedLR1CSInstance<E>,
    u: &AllocatedSplitR1CSInstance<E>,
    prev_IC: (&AllocatedNum<E::Scalar>, &AllocatedNum<E::Scalar>),
  ) -> Result<AllocatedBit, SynthesisError> {
    let hash = self.calculate_hash(
      cs.namespace(|| "calculate_hash"),
      pp_digest,
      i,
      z_0,
      z_i,
      U,
      prev_IC,
    )?;
    let hash_check = alloc_num_equals(
      cs.namespace(|| "u.X[0] = H(params, i, z_0, z_i, U)"),
      &u.x0,
      &hash,
    )?;
    Ok(hash_check)
  }

  pub fn hash_check_cyclefold<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    U: &AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
    u: &AllocatedSplitR1CSInstance<E>,
  ) -> Result<AllocatedBit, SynthesisError> {
    let hash = self.calculate_hash_cyclefold(cs.namespace(|| "calculate_hash"), pp_digest, i, U)?;
    let hash_check = alloc_num_equals(
      cs.namespace(|| "u.X[1] = H(params, i, U_cyclefold)"),
      &u.x1,
      &hash,
    )?;
    Ok(hash_check)
  }

  pub fn enforce_hash_check<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    check_non_base_pass: &AllocatedBit,
    is_base_case: &AllocatedBit,
  ) -> Result<(), SynthesisError> {
    let should_be_false = AllocatedBit::nor(
      cs.namespace(|| "check_non_base_pass nor base_case"),
      check_non_base_pass,
      is_base_case,
    )?;
    cs.enforce(
      || "check_non_base_pass nor base_case = false",
      |lc| lc + should_be_false.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
    Ok(())
  }

  pub fn calculate_hash<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    z_0: &[AllocatedNum<E::Scalar>],
    z_i: &[AllocatedNum<E::Scalar>],
    U: &AllocatedLR1CSInstance<E>,
    IC: (&AllocatedNum<E::Scalar>, &AllocatedNum<E::Scalar>),
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(self.ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp_digest);
    ro.absorb(i);
    for e in z_0 {
      ro.absorb(e)
    }
    for e in z_i {
      ro.absorb(e)
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;
    ro.absorb(IC.0);
    ro.absorb(IC.1);
    let hash_bits = ro.squeeze(cs.namespace(|| "primary hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "primary hash"), &hash_bits)?;
    Ok(hash)
  }

  pub fn calculate_hash_cyclefold<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    U: &AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(self.ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp_digest);
    ro.absorb(i);
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;
    let hash_bits = ro.squeeze(cs.namespace(|| "primary hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "primary hash"), &hash_bits)?;
    Ok(hash)
  }

  pub fn increment_ic<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    prev_IC: (AllocatedNum<E::Scalar>, AllocatedNum<E::Scalar>),
    comm_advice: (
      &AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
      &AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    ),
    is_base_case: &AllocatedBit,
  ) -> Result<(AllocatedNum<E::Scalar>, AllocatedNum<E::Scalar>), SynthesisError> {
    Ok((
      self.increment_ic_sole(
        cs.namespace(|| "increment IC0"),
        prev_IC.0,
        comm_advice.0,
        is_base_case,
      )?,
      self.increment_ic_sole(
        cs.namespace(|| "increment IC1"),
        prev_IC.1,
        comm_advice.1,
        is_base_case,
      )?,
    ))
  }

  pub fn increment_ic_sole<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    prev_IC: AllocatedNum<E::Scalar>,
    comm_advice: &AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    is_base_case: &AllocatedBit,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
    let IC = {
      let mut ro = <Dual<E> as Engine>::ROCircuit::new(self.ro_consts.clone(), DEFAULT_ABSORBS);
      ro.absorb(&prev_IC);
      comm_advice.absorb_in_ro(cs.namespace(|| "absorb pre_committed0"), &mut ro)?;
      let IC_bits = ro.squeeze(cs.namespace(|| "IC_bits_IS"), NUM_HASH_BITS)?;
      le_bits_to_num(cs.namespace(|| "IC"), &IC_bits)?
    };
    conditionally_select(
      cs.namespace(|| "select IC"),
      &prev_IC,
      &IC,
      &Boolean::from(is_base_case.clone()),
    )
  }

  pub const fn new(
    params: &'a AugmentedCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    inputs: Option<AugmentedCircuitInputs<E>>,
    step_circuit: &'a SC,
    num_rounds: usize,
  ) -> Self {
    Self {
      params,
      ro_consts,
      inputs,
      step_circuit,
      num_rounds,
    }
  }
}

// The direct construction from the HyperNova paper has a circular definition: the size
// of the augmented circuit is dependent on the number of rounds of the sumcheck (`s`),
// but the number of sumcheck rounds is also dependent (logarithmically) on the size of
// the augmented circuit.
//
// Luckily, since the dependency is logarithmic we should pretty easily find a fixpoint
// where this circularity stabilizes. In an ideal world, we would project the augmented
// circuit size exactly. Unfortunately this may not be possible -- for example, at time
// of writing we use poseidon as our hash function, which does not have a fixed circuit
// size of its own. However, an upper bound will be good enough, with a small chance of
// incorporating an unnecessary sumcheck round. A further tradeoff is that if we change
// the augmented circuit then function may need to be updated.
//
// For an example of how this computation should work, imagine that the number of base
// constraints (those neither in the step circuit or in sumcheck) is 20, each sumcheck
// round has 10, and the step circuit has 2. Then we will need at least
//
//     2^4 < 22 < 2^5 --> 5
//
// sumcheck rounds. So that gives us an augmented circuit size of 72. But this means we
// will need at least
//
//     2^6 < 72 < 2^7 --> 7
//
// sumcheck rounds. That gives an augmented circuit with size 92 -- which is a fixpoint
// as 7 sumcheck rounds remains sufficient.
pub(crate) fn project_aug_circuit_size<E>(
  base_cons: usize,
  cons_per_input: usize,
  cons_per_sumcheck_round: usize,
  step_circuit: &impl StepCircuit<E::Scalar>,
) -> usize
where
  E: CurveCycleEquipped,
{
  let mut cs: ShapeCS<E> = ShapeCS::new();
  let zero: AllocatedNum<E::Scalar> = alloc_zero(cs.namespace(|| "zero"));
  let z_0 = (0..step_circuit.arity())
    .map(|_| zero.clone())
    .collect_vec();
  let _ = step_circuit.synthesize(&mut cs, &z_0);
  let step_circuit_cons = cs.num_constraints();
  let mut max_cons =
    base_cons + step_circuit_cons + (step_circuit.arity()).saturating_sub(1) * cons_per_input;

  // Initialize `low` to represent the previous round count (starting at 0).
  let mut low = 0;

  // Estimate the initial number of rounds needed based on max_cons.
  let mut high = max_cons.log_2();

  // A flag to track whether the round estimation has stabilized (converged).
  let mut eq = false;

  // Iterate until the round count stabilizes.
  while !eq {
    // Increase `max_cons` to account for additional constraints required by the new rounds.
    // (high - low) represents the increase in the round count since the last iteration,
    // and `cons_per_sumcheck_round` is the extra constraints required per round.
    max_cons += (high - low) * cons_per_sumcheck_round;

    // Update `low` to the previous round count.
    low = high;

    // Recompute the round count with the updated `max_cons`.
    high = max_cons.log_2();

    // Check if the round count has stabilized.
    // If the new round count (`high`) equals the previous count (`low`),
    // no further adjustment is necessary.
    eq = low == high;
  }
  high
}

#[cfg(test)]
mod tests {
  use std::marker::PhantomData;

  use ff::PrimeField;

  use crate::{
    constants::{
      BASE_CONSTRAINTS, BN_LIMB_WIDTH, BN_N_LIMBS, EDGE_CASE_CONSTRAINTS,
      MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT, MAX_CONSTRAINTS_PER_SUMCHECK_ROUND,
    },
    frontend::{num::AllocatedNum, shape_cs::ShapeCS, ConstraintSystem, SynthesisError},
    hypernova::{augmented_circuit::project_aug_circuit_size, rs::StepCircuit},
    provider::Bn256EngineIPA,
    spartan::math::Math,
    traits::{Dual, Engine, ROConstantsCircuit},
    AugmentedCircuitParams,
  };

  /// A trivial step circuit that simply returns the input
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub struct TrivialCircuit<F> {
    _p: PhantomData<F>,
  }

  impl<F> Default for TrivialCircuit<F>
  where
    F: PrimeField,
  {
    /// Creates a new trivial test circuit with step counter type Incremental
    fn default() -> TrivialCircuit<F> {
      Self { _p: PhantomData }
    }
  }

  impl<F> StepCircuit<F> for TrivialCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      0
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      _cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      Ok(z.to_vec())
    }
  }

  /// A trivial step circuit that simply returns the input
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub struct TrivialCircuit2<F> {
    _p: PhantomData<F>,
  }

  impl<F> Default for TrivialCircuit2<F>
  where
    F: PrimeField,
  {
    /// Creates a new trivial test circuit with step counter type Incremental
    fn default() -> TrivialCircuit2<F> {
      Self { _p: PhantomData }
    }
  }

  impl<F> StepCircuit<F> for TrivialCircuit2<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      _cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      Ok(z.to_vec())
    }
  }

  #[derive(Clone, Debug, PartialEq, Eq)]
  pub struct TrivialCircuit3<F> {
    _p: PhantomData<F>,
  }

  impl<F> Default for TrivialCircuit3<F>
  where
    F: PrimeField,
  {
    /// Creates a new trivial test circuit with step counter type Incremental
    fn default() -> TrivialCircuit3<F> {
      Self { _p: PhantomData }
    }
  }

  impl<F> StepCircuit<F> for TrivialCircuit3<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      2
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      _cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      Ok(z.to_vec())
    }
  }

  use super::AugmentedCircuit;
  type E = Bn256EngineIPA;
  type F = <E as Engine>::Scalar;

  #[test]
  fn test_circuit_constants_sumcheck() {
    // Get the round constants used in the Poseidon hash function circuit.
    let ro_consts_circuit = ROConstantsCircuit::<Dual<E>>::default();
    // Use a trivial circuit because it has 0 constraints.
    let test_circuit = TrivialCircuit::default();

    // Augmented circuit parameters.
    let augmented_circuit_params = crate::AugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS);

    // Helper closure to synthesize a circuit with a given number of sumcheck rounds,
    // returning the total number of constraints.
    let synthesize_constraints = |sumcheck_rounds: usize| -> usize {
      let circuit: AugmentedCircuit<'_, E, _> = AugmentedCircuit::new(
        &augmented_circuit_params,
        ro_consts_circuit.clone(),
        None,
        &test_circuit,
        sumcheck_rounds,
      );
      let mut cs: ShapeCS<E> = ShapeCS::new();
      let _ = circuit.synthesize(&mut cs);
      cs.num_constraints()
    };

    // --- Constraint Generation ---
    // Baseline: no sumcheck rounds.
    let base_cons = synthesize_constraints(0);
    println!("Base constraints: {}", base_cons);
    assert_eq!(base_cons, BASE_CONSTRAINTS - EDGE_CASE_CONSTRAINTS);

    // 1 sumcheck round.
    let cons_1 = synthesize_constraints(1);
    println!(
      "Additional sumcheck constraints for one round: {}",
      cons_1 - base_cons
    );
    assert_eq!(
      cons_1 - base_cons - EDGE_CASE_CONSTRAINTS,
      MAX_CONSTRAINTS_PER_SUMCHECK_ROUND
    );

    // 2 sumcheck rounds.
    let cons_2 = synthesize_constraints(2);
    println!(
      "Additional sumcheck constraints for two rounds: {}",
      cons_2 - cons_1
    );
    assert_eq!(cons_2 - cons_1, MAX_CONSTRAINTS_PER_SUMCHECK_ROUND);

    // 3 sumcheck rounds.
    let cons_3 = synthesize_constraints(3);
    println!(
      "Additional sumcheck constraints for three rounds: {}",
      cons_3 - cons_2
    );
    assert_eq!(cons_3 - cons_2, cons_2 - cons_1);

    // 17 sumcheck rounds.
    const SUMCHECK_ROUNDS: usize = 17;
    let cons_rounds = synthesize_constraints(SUMCHECK_ROUNDS);

    // --- Estimate the number of rounds ---
    let estimated_rounds = project_aug_circuit_size::<E>(
      BASE_CONSTRAINTS,
      MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT,
      MAX_CONSTRAINTS_PER_SUMCHECK_ROUND,
      &test_circuit,
    );
    println!("Estimated number of rounds: {}", estimated_rounds);

    // Calculate the actual rounds by taking the next power of two of the 15-round constraint count.
    let actual_rounds = cons_rounds.next_power_of_two().log_2();
    println!("Actual number of rounds: {}", actual_rounds);
    assert_eq!(actual_rounds, estimated_rounds);
    assert_eq!(actual_rounds, SUMCHECK_ROUNDS);
  }

  #[test]
  fn test_circuit_constants_inputs() {
    // Get the round constants used in the Poseidon hash function circuit.
    let ro_consts_circuit = ROConstantsCircuit::<Dual<E>>::default();
    // Augmented circuit parameters.
    let augmented_circuit_params = AugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS);

    // Helper closure to synthesize a circuit with a given number of sumcheck rounds,
    // returning the total number of constraints.
    fn synthesize_constraints(
      circuit: &impl StepCircuit<F>,
      sumcheck_rounds: usize,
      params: &AugmentedCircuitParams,
      ro_consts: &ROConstantsCircuit<Dual<E>>,
    ) -> usize {
      let circuit_primary: AugmentedCircuit<'_, E, _> =
        AugmentedCircuit::new(params, ro_consts.clone(), None, circuit, sumcheck_rounds);
      let mut cs: ShapeCS<E> = ShapeCS::new();
      let _ = circuit_primary.synthesize(&mut cs);
      cs.num_constraints()
    }

    // --- Constraint Generation with varying inputs ---

    // Case 1: The Step Circuit with 0 inputs.
    let base_cons = synthesize_constraints(
      &TrivialCircuit::default(),
      0,
      &augmented_circuit_params,
      &ro_consts_circuit,
    );

    // Case 2: The Step Circuit with 1 input.
    let cons1 = synthesize_constraints(
      &TrivialCircuit2::default(),
      0,
      &augmented_circuit_params,
      &ro_consts_circuit,
    );
    let input1_constraints = cons1 - base_cons;
    println!("Constraints per input (1 input): {}", input1_constraints);
    assert_eq!(input1_constraints, MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT);

    // Case 3: The Step Circuit with 2 inputs.
    let cons2 = synthesize_constraints(
      &TrivialCircuit3::default(),
      0,
      &augmented_circuit_params,
      &ro_consts_circuit,
    );
    let input2_constraints = cons2 - cons1;
    println!(
      "Constraints per additional input (2 inputs): {}",
      input2_constraints
    );
    assert_eq!(input2_constraints, MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT);
  }
}
