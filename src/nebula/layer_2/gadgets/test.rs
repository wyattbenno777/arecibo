use std::marker::PhantomData;

use crate::cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance;
use crate::traits::commitment::CommitmentTrait;
use crate::{
  cyclefold::gadgets::emulated,
  errors::NovaError,
  nebula::{
    augmented_circuit::AugmentedCircuitParams,
    layer_2::nifs::NIFS,
    rs::{PublicParams, RecursiveSNARK, StepCircuit},
  },
  provider::Bn256EngineIPA,
  r1cs::RelaxedR1CSInstance,
  traits::{snark::default_ck_hint, CurveCycleEquipped, Dual, Engine, ROConstantsCircuit},
  Commitment,
};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use num_traits::Pow;

use super::PrimaryNIFSVerifierGadget;

// Proving Engine
type E = Bn256EngineIPA;
type F = <E as Engine>::Scalar;

#[test]
fn test_folding_ivc_proofs() {}

// Simulate orchestrator node
fn sim_orchestrator_node() -> Result<(), NovaError> {
  let num_nodes = 10;
  let circuit: PowCircuit<E> = PowCircuit::new();
  let pp = PublicParams::<E>::setup(&circuit, &default_ck_hint(), &default_ck_hint());
  let snarks = sim_node_nw(&pp, &circuit, num_nodes)?;
  let verifier_circuit: VerifierCircuit<E> = VerifierCircuit::new(
    pp.augmented_circuit_params,
    pp.ro_consts_circuit.clone(),
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
  );
  let on_pp: PublicParams<E> =
    PublicParams::setup(&verifier_circuit, &default_ck_hint(), &default_ck_hint());
  Ok(())
}

// generate a collection of [`RecursiveSNARK`]'s simulating a node network producing [`RecursiveSNARK`]'s
fn sim_node_nw(
  pp: &PublicParams<E>,
  step_circuit: &impl StepCircuit<F>,
  num_nodes: usize,
) -> Result<Vec<RecursiveSNARK<E>>, NovaError> {
  let mut z0 = vec![F::from(42_u64)];
  let mut snarks = Vec::with_capacity(num_nodes);
  let mut node = || {
    let mut rs = RecursiveSNARK::new(pp, step_circuit, &z0)?;
    let mut IC_i = F::zero();
    for _ in 0..3 {
      rs.prove_step(pp, step_circuit, IC_i)?;
      IC_i = rs.increment_commitment(pp, step_circuit);
    }
    z0 = rs.verify(pp, rs.num_steps(), &z0, IC_i)?;
    snarks.push(rs);
    Ok::<(), NovaError>(())
  };
  for _ in 0..num_nodes {
    node()?;
  }
  Ok(snarks)
}

#[derive(Clone)]
pub struct VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  pp_digest: Option<E::Scalar>,
  nifs: Option<NIFS<E>>,
  U1: Option<RelaxedR1CSInstance<E>>,
  U2: Option<RelaxedR1CSInstance<E>>,
  E_new: Option<Commitment<E>>,
  W_new: Option<Commitment<E>>,
  U1_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
  U2_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
  params: AugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E>>,
}

impl<E> StepCircuit<E::Scalar> for VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  fn arity(&self) -> usize {
    0
  }

  fn synthesize<CS: bellpepper_core::ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, bellpepper_core::SynthesisError> {
    let (pp_digest, U1, U2, nifs_primary, E_new, W_new) =
      self.alloc_witness(cs.namespace(|| "alloc witness"))?;
    let U = nifs_primary.verify(
      cs.namespace(|| "nifs primary verify"),
      self.ro_consts.clone(),
      &U1,
      &U2,
      &pp_digest,
      E_new,
      W_new,
    )?;
    Ok(z.to_vec())
  }

  fn non_deterministic_advice(&self) -> Vec<E::Scalar> {
    vec![]
  }
}

impl<E> VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  fn new(
    params: AugmentedCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    pp_digest: Option<E::Scalar>,
    nifs: Option<NIFS<E>>,
    U1: Option<RelaxedR1CSInstance<E>>,
    U2: Option<RelaxedR1CSInstance<E>>,
    E_new: Option<Commitment<E>>,
    W_new: Option<Commitment<E>>,
    U1_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
    U2_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
  ) -> Self {
    Self {
      pp_digest,
      nifs,
      U1,
      U2,
      E_new,
      W_new,
      U1_secondary,
      U2_secondary,
      params,
      ro_consts,
    }
  }

  fn alloc_witness<CS>(
    &self,
    mut cs: CS,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,                               // pp_digest
      AllocatedEmulRelaxedR1CSInstance<Dual<E>>,             // U1
      AllocatedEmulRelaxedR1CSInstance<Dual<E>>,             // U2
      PrimaryNIFSVerifierGadget<E>,                          // nifs_primary
      emulated::AllocatedEmulPoint<<Dual<E> as Engine>::GE>, // E_new
      emulated::AllocatedEmulPoint<<Dual<E> as Engine>::GE>, // W_new
    ),
    SynthesisError,
  >
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let pp_digest = AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || {
      Ok(self.pp_digest.unwrap_or(E::Scalar::ZERO))
    })?;
    let U1 = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      self.U1.as_ref(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let U2 = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      self.U2.as_ref(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let nifs_primary = PrimaryNIFSVerifierGadget::alloc(
      cs.namespace(|| "primary_nifs"),
      self.nifs.as_ref().map(|nifs| &nifs.nifs_primary),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let E_new = emulated::AllocatedEmulPoint::alloc(
      cs.namespace(|| "E_new"),
      self.E_new.map(|E_new| E_new.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    let W_new = emulated::AllocatedEmulPoint::alloc(
      cs.namespace(|| "W_new"),
      self.W_new.map(|W_new| W_new.to_coordinates()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    Ok((pp_digest, U1, U2, nifs_primary, E_new, W_new))
  }
}

#[derive(Clone, Default)]
pub struct PowCircuit<E>
where
  E: Engine,
{
  _engine: PhantomData<E>,
}

impl<E> PowCircuit<E>
where
  E: Engine,
{
  pub fn new() -> Self {
    Self {
      _engine: PhantomData,
    }
  }
}

impl<E> StepCircuit<E::Scalar> for PowCircuit<E>
where
  E: Engine,
{
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: bellpepper_core::ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, bellpepper_core::SynthesisError> {
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..10 {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok(vec![y])
  }

  fn non_deterministic_advice(&self) -> Vec<E::Scalar> {
    vec![]
  }
}
