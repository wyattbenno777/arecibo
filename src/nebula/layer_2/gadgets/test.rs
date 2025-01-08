use std::marker::PhantomData;

use super::{PrimaryNIFSVerifierGadget, NUM_CHALLENGE_BITS};
use crate::constants::NUM_FE_IN_EMULATED_POINT;
use crate::cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance;
use crate::gadgets::scalar_as_base;
use crate::nebula::layer_2::utils::absorb_U;
use crate::r1cs::RelaxedR1CSWitness;
use crate::traits::commitment::CommitmentTrait;
use crate::traits::ROTrait;
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
use crate::{CommitmentKey, R1CSWithArity};
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use tracing_subscriber::{fmt, layer::SubscriberExt, EnvFilter, Registry};
use tracing_texray::TeXRayLayer;

// Proving Engine
type E = Bn256EngineIPA;
type F = <E as Engine>::Scalar;

pub struct AggregationPublicParams<E>
where
  E: CurveCycleEquipped,
{
  pp: PublicParams<E>,
  circuit_shape_F: R1CSWithArity<E>,
  digest_F: E::Scalar,
  ck: CommitmentKey<E>,
}

impl<E> AggregationPublicParams<E>
where
  E: CurveCycleEquipped,
{
  #[tracing::instrument(skip_all, name = "AggregationPublicParams::setup")]
  fn setup(pp_F: PublicParams<E>) -> Self {
    let verifier_circuit: VerifierCircuit<E> = VerifierCircuit::new(
      pp_F.augmented_circuit_params,
      pp_F.ro_consts_circuit.clone(),
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
    );
    let pp: PublicParams<E> =
      PublicParams::setup(&verifier_circuit, &*default_ck_hint(), &*default_ck_hint());
    let (circuit_shape_F, ck, digest_F) = pp_F.into_shape_ck_digest();

    Self {
      pp,
      circuit_shape_F,
      digest_F,
      ck,
    }
  }

  fn circuit_shape_cyclefold(&self) -> &R1CSWithArity<Dual<E>> {
    &self.pp.circuit_shape_cyclefold
  }

  fn ck_cyclefold(&self) -> &CommitmentKey<Dual<E>> {
    &self.pp.ck_cyclefold
  }

  fn ck(&self) -> &CommitmentKey<E> {
    &self.ck
  }

  fn augmented_circuit_params(&self) -> AugmentedCircuitParams {
    self.pp.augmented_circuit_params
  }
}

pub struct AggregationRecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  r_W: RelaxedR1CSWitness<E>,
  r_U: RelaxedR1CSInstance<E>,
  r_W_cyclefold: RelaxedR1CSWitness<Dual<E>>,
  r_U_cyclefold: RelaxedR1CSInstance<Dual<E>>,
  rs: RecursiveSNARK<E>,
  IC_i: E::Scalar,
  i: usize,
  z0: Vec<E::Scalar>,
}

impl<E> AggregationRecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  #[tracing::instrument(skip_all, name = "AggregationRecursiveSNARK::new")]
  fn new(pp: &AggregationPublicParams<E>, rs_F: &RecursiveSNARK<E>) -> Result<Self, NovaError> {
    let F_shape = &pp.circuit_shape_F.r1cs_shape;
    let r_U = RelaxedR1CSInstance::default(&pp.ck, F_shape);
    let r_W = RelaxedR1CSWitness::default(F_shape);
    let r1cs_cyclefold = &pp.circuit_shape_cyclefold().r1cs_shape;
    let r_U_cyclefold = RelaxedR1CSInstance::default(pp.ck_cyclefold(), r1cs_cyclefold);
    let r_W_cyclefold = RelaxedR1CSWitness::default(r1cs_cyclefold);
    let (U2, W2, U2_secondary, W2_secondary) = rs_F.primary_secondary_U_W();
    let (nifs, (new_r_U, new_r_W), (new_r_U_cyclefold, new_r_W_cyclefold)) = NIFS::prove(
      (pp.ck(), pp.ck_cyclefold()),
      &pp.pp.ro_consts,
      &pp.digest_F,
      (
        &pp.circuit_shape_F.r1cs_shape,
        &pp.pp.circuit_shape_cyclefold.r1cs_shape,
      ),
      (&r_U, &r_W),
      (U2, W2),
      (&r_U_cyclefold, &r_W_cyclefold),
      (U2_secondary, W2_secondary),
    )?;
    let E_new = new_r_U.comm_E;
    let W_new = new_r_U.comm_W;
    let verifier_circuit: VerifierCircuit<E> = VerifierCircuit::new(
      pp.augmented_circuit_params(),
      pp.pp.ro_consts_circuit.clone(),
      Some(pp.digest_F),
      Some(nifs),
      Some(r_U.clone()),
      Some(U2.clone()),
      Some(E_new),
      Some(W_new),
      Some(r_U_cyclefold),
      Some(U2_secondary.clone()),
    );
    let z0 = {
      let mut ro = <Dual<E> as Engine>::RO::new(
        pp.pp.ro_consts.clone(),
        2 * NUM_FE_IN_EMULATED_POINT + 2, // U.comm_E + U.comm_W + U.X
      );
      absorb_U::<E>(&r_U, &mut ro);
      let hash_U = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
      vec![hash_U]
    };

    let mut IC_i = E::Scalar::ZERO;
    let mut rs = RecursiveSNARK::new(&pp.pp, &verifier_circuit, &z0)?;
    rs.prove_step(&pp.pp, &verifier_circuit, IC_i)?;
    IC_i = rs.increment_commitment(&pp.pp, &verifier_circuit);
    Ok(Self {
      r_W: new_r_W,
      r_U: new_r_U,
      r_W_cyclefold: new_r_W_cyclefold,
      r_U_cyclefold: new_r_U_cyclefold,
      rs,
      IC_i,
      i: 0,
      z0,
    })
  }

  #[tracing::instrument(skip_all, name = "AggregationRecursiveSNARK::prove_step")]
  fn prove_step(
    &mut self,
    pp: &AggregationPublicParams<E>,
    rs_F: &RecursiveSNARK<E>,
  ) -> Result<(), NovaError> {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }
    let (U2, W2, U2_secondary, W2_secondary) = rs_F.primary_secondary_U_W();
    let (nifs, (new_r_U, new_r_W), (new_r_U_cyclefold, new_r_W_cyclefold)) = NIFS::prove(
      (pp.ck(), pp.ck_cyclefold()),
      &pp.pp.ro_consts,
      &pp.digest_F,
      (
        &pp.circuit_shape_F.r1cs_shape,
        &pp.pp.circuit_shape_cyclefold.r1cs_shape,
      ),
      (&self.r_U, &self.r_W),
      (U2, W2),
      (&self.r_U_cyclefold, &self.r_W_cyclefold),
      (U2_secondary, W2_secondary),
    )?;
    let E_new = new_r_U.comm_E;
    let W_new = new_r_U.comm_W;
    let verifier_circuit: VerifierCircuit<E> = VerifierCircuit::new(
      pp.augmented_circuit_params(),
      pp.pp.ro_consts_circuit.clone(),
      Some(pp.digest_F),
      Some(nifs),
      Some(self.r_U.clone()),
      Some(U2.clone()),
      Some(E_new),
      Some(W_new),
      Some(self.r_U_cyclefold.clone()),
      Some(U2_secondary.clone()),
    );
    self.rs.prove_step(&pp.pp, &verifier_circuit, self.IC_i)?;
    self.IC_i = self.rs.increment_commitment(&pp.pp, &verifier_circuit);
    self.r_U = new_r_U;
    self.r_W = new_r_W;
    self.r_U_cyclefold = new_r_U_cyclefold;
    self.r_W_cyclefold = new_r_W_cyclefold;
    self.i += 1;
    Ok(())
  }

  #[tracing::instrument(skip_all, name = "AggregationRecursiveSNARK::verify")]
  pub fn verify(&self, pp: &AggregationPublicParams<E>) -> Result<(), NovaError> {
    self
      .rs
      .verify(&pp.pp, self.rs.num_steps(), &self.z0, self.IC_i)?;
    let (res_r_F, res_r_cyclefold) = rayon::join(
      || {
        pp.circuit_shape_F
          .r1cs_shape
          .is_sat_relaxed(&pp.ck, &self.r_U, &self.r_W)
      },
      || {
        pp.circuit_shape_cyclefold().r1cs_shape.is_sat_relaxed(
          pp.ck_cyclefold(),
          &self.r_U_cyclefold,
          &self.r_W_cyclefold,
        )
      },
    );
    res_r_F?;
    res_r_cyclefold?;
    Ok(())
  }
}

#[test]
fn test_folding_ivc_proofs() -> Result<(), NovaError> {
  tracing_init();
  tracing_texray::examine(tracing::info_span!("sim_orchestrator_node"))
    .in_scope(sim_orchestrator_node)
}

// Simulate orchestrator node
fn sim_orchestrator_node() -> Result<(), NovaError> {
  let num_nodes = 10;
  let circuit: PowCircuit<E> = PowCircuit::new();
  let node_pp = PublicParams::<E>::setup(&circuit, &default_ck_hint(), &default_ck_hint());
  let snarks = sim_node_nw(&node_pp, &circuit, num_nodes)?;
  let on_pp = AggregationPublicParams::setup(node_pp);
  let mut on_rs = AggregationRecursiveSNARK::new(&on_pp, &snarks[0])?;
  for snark in snarks.iter() {
    on_rs.prove_step(&on_pp, snark)?;
  }
  on_rs.verify(&on_pp)?;
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
    1
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

fn tracing_init() {
  // Create an EnvFilter that filters out spans below the 'info' level
  let filter = EnvFilter::new("arecibo=info");

  // Create a TeXRayLayer
  let texray_layer = TeXRayLayer::new(); // Optional: Only show spans longer than 100ms

  // Set up the global subscriber
  let subscriber = Registry::default()
    .with(filter)
    .with(fmt::layer())
    .with(texray_layer);
  tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
}
