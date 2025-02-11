use super::{nifs::NIFS, StepCircuit};
use crate::{
  and_then_field,
  frontend::{
    gadgets::Assignment, num::AllocatedNum, shape_cs::ShapeCS, Boolean, ConstraintSystem,
    SynthesisError,
  },
  gadgets::{
    alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select_vec,
    emulated::{AllocatedEmulLR1CSInstance, AllocatedEmulPoint},
    hypernova::{
      alloc_sized_vec, increment, AllocatedLR1CSInstance, AllocatedNIFS, AllocatedR1CSInstance,
    },
  },
  map_field,
  r1cs::{LR1CSInstance, R1CSInstance},
  spartan::math::Math,
  traits::{commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROConstantsCircuit},
  AugmentedCircuitParams, Commitment,
};
use ff::Field;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AugmentedCircuitInputs<E>
where
  E: CurveCycleEquipped,
{
  pp_digest: E::Base,
  i: E::Scalar,
  z_0: Vec<E::Scalar>,
  z_i: Option<Vec<E::Scalar>>,
  nifs: Option<NIFS<E>>,
  U: Option<LR1CSInstance<E>>,
  u: Option<R1CSInstance<E>>,
  W_new: Option<Commitment<E>>,
}

impl<E> AugmentedCircuitInputs<E>
where
  E: CurveCycleEquipped,
{
  pub fn new(
    pp_digest: E::Base,
    i: E::Scalar,
    z_0: Vec<E::Scalar>,
    z_i: Option<Vec<E::Scalar>>,
    nifs: Option<NIFS<E>>,
    U: Option<LR1CSInstance<E>>,
    u: Option<R1CSInstance<E>>,
    W_new: Option<Commitment<E>>,
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
    let (pp_digest, i, z_0, z_i, nifs, U, u, W_new) =
      self.alloc_witness(cs.namespace(|| "alloc_witness"), arity)?;

    // Base case: i = 0
    // ////////////////
    //
    // 1. Check if this is the base
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "is base case"), &i, &zero)?;
    // ////////////////////////////////////
    // 2. Get the default running instance.
    let U_default = self.synthesize_base_case(cs.namespace(|| "base case"))?;

    // Non-base case: i > 0
    // ////////////////////
    //
    // 1. U <- NIFS.V
    let U = self.synthesize_non_base_case(
      cs.namespace(|| "non base case"),
      &pp_digest,
      &self.ro_consts,
      &nifs,
      &U,
      &u,
      W_new,
    )?;

    // Compute i + 1
    let i_new = increment(cs.namespace(|| "i++"), &i)?;

    // Compute z_{i+1}
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Synthesize the step circuit (F) and compute the next output zi+1 ← F(zi,ωi).
    let z_next = self
      .step_circuit
      .synthesize(&mut cs.namespace(|| "F"), &z_input)?;

    // Check step_circuit_i (F_i) conforms to structure F
    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }
    Ok(z_next)
  }

  pub fn synthesize_non_base_case<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    ro_consts: &ROConstantsCircuit<Dual<E>>,
    nifs: &AllocatedNIFS<E>,
    U: &AllocatedLR1CSInstance<E>,
    u: &AllocatedR1CSInstance<E>,
    W_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  ) -> Result<AllocatedLR1CSInstance<E>, SynthesisError> {
    nifs.verify(
      cs.namespace(|| "NIFS.V"),
      pp_digest,
      ro_consts,
      U,
      u,
      W_new,
      self.num_rounds,
    )
  }

  fn alloc_witness<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,                     // pp_digest
      AllocatedNum<E::Scalar>,                     // i
      Vec<AllocatedNum<E::Scalar>>,                // z0
      Vec<AllocatedNum<E::Scalar>>,                // zi
      AllocatedNIFS<E>,                            // nifs
      AllocatedLR1CSInstance<E>,                   // U
      AllocatedR1CSInstance<E>,                    // u
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>, // W_new
    ),
    SynthesisError,
  > {
    // Allocate primitives: pp_digest, i, z_0
    let pp_digest = alloc_scalar_as_base::<Dual<E>, _>(
      cs.namespace(|| "params"),
      map_field!(self.inputs, ref, pp_digest).copied(),
    )?;
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;
    let z_0 = alloc_sized_vec(
      cs.namespace(|| "z_0"),
      map_field!(self.inputs, ref, z_0),
      arity,
    )?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
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
    )?;
    let U = AllocatedLR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      and_then_field!(self.inputs, U),
      self.params.limb_width,
      self.params.n_limbs,
      self.num_rounds,
    )?;
    let u = AllocatedR1CSInstance::alloc(
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
    Ok((pp_digest, i, z_0, z_i, nifs, U, u, W_new))
  }

  pub fn synthesize_base_case<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedEmulLR1CSInstance<E>, SynthesisError> {
    let U_default = AllocatedEmulLR1CSInstance::default(
      cs.namespace(|| "Allocated U_default"),
      self.params.limb_width,
      self.params.n_limbs,
      self.num_rounds,
    )?;
    Ok(U_default)
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
  let z0 = (0..step_circuit.arity())
    .map(|_| zero.clone())
    .collect_vec();
  let _ = step_circuit.synthesize(&mut cs, &z0);
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
      BASE_CONSTRAINTS, BN_LIMB_WIDTH, BN_N_LIMBS, MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT,
      MAX_CONSTRAINTS_PER_SUMCHECK_ROUND,
    },
    frontend::{num::AllocatedNum, shape_cs::ShapeCS, ConstraintSystem, SynthesisError},
    hypernova::{augmented_circuit::project_aug_circuit_size, StepCircuit},
    provider::Bn256EngineIPA,
    traits::{Dual, ROConstantsCircuit},
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

  use super::AugmentedCircuit;
  type E = Bn256EngineIPA;
  #[test]
  fn test_base_aug_circuit_size_bn254() {
    // Get the round constants used in the poseidon hash function and poseidon hash function circuit
    let ro_consts_circuit = ROConstantsCircuit::<Dual<E>>::default();
    let circuit = TrivialCircuit::default();
    // Get the structure for the AugmentedCircuit and corresponding commitment key
    let augmented_circuit_params = crate::AugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS);
    let num_rounds = project_aug_circuit_size::<E>(
      BASE_CONSTRAINTS,
      MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT,
      MAX_CONSTRAINTS_PER_SUMCHECK_ROUND,
      &circuit,
    );
    let circuit_primary: AugmentedCircuit<'_, E, _> = AugmentedCircuit::new(
      &augmented_circuit_params,
      ro_consts_circuit.clone(),
      None,
      &circuit,
      0,
    );
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    println!("Constraints: {}", cs.num_constraints());
    println!("Num rounds: {}", num_rounds);
  }
}
