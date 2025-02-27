use super::{
  le_bits_to_num, r1cs::AllocatedRelaxedR1CSInstanceBn, CycleFoldNIFSVerifierGadget,
  CycleFoldRelaxedNIFSVerifierGadget, NIFSVerifierGadget, PrimaryNIFSVerifierGadget,
  NUM_CHALLENGE_BITS,
};
use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_FE_IN_EMULATED_POINT},
  cyclefold::gadgets::{emulated::AllocatedEmulRelaxedR1CSInstance, AllocatedCycleFoldInstance},
  errors::NovaError,
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{emulated::AllocatedEmulPoint, scalar_as_base},
  nebula::{
    augmented_circuit::AugmentedCircuitParams,
    layer_2::{
      nifs::NIFS,
      utils::{absorb_U, absorb_U_bn, Layer2FoldingData},
    },
    rs::{PublicParams, RecursiveSNARK, StepCircuit},
  },
  provider::PallasEngine,
  r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    commitment::CommitmentTrait, snark::default_ck_hint, CurveCycleEquipped, Dual, Engine,
    ROCircuitTrait, ROConstantsCircuit, ROTrait,
  },
  CommitmentKey, R1CSWithArity,
};
use ff::Field;
use std::marker::PhantomData;

// Proving Engine
type E = PallasEngine;
type F = <E as Engine>::Scalar;

#[test]
fn test_folding_ivc_proofs() -> Result<(), NovaError> {
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
  tracing::info_span!("aggregation").in_scope(|| -> Result<(), NovaError> {
    for snark in snarks.iter() {
      on_rs.prove_step(&on_pp, snark)?;
    }
    Ok(())
  })?;
  on_rs.verify(&on_pp)?;
  Ok(())
}

// generate a collection of [`RecursiveSNARK`]'s simulating a node network producing [`RecursiveSNARK`]'s
#[tracing::instrument(skip_all, name = "sim_node_nw")]
fn sim_node_nw(
  pp: &PublicParams<E>,
  step_circuit: &impl StepCircuit<F>,
  num_nodes: usize,
) -> Result<Vec<RecursiveSNARK<E>>, NovaError> {
  // This is the initial input value for the [`PowCircuit`].
  let mut z0 = vec![F::from(42_u64)];

  // Network's output
  let mut snarks = Vec::with_capacity(num_nodes);

  // Node logic
  let mut node = || -> Result<(), NovaError> {
    let mut rs = RecursiveSNARK::new(pp, step_circuit, &z0)?;
    let mut IC_i = F::zero();
    for _ in 0..3 {
      rs.prove_step(pp, step_circuit, IC_i)?;
      IC_i = rs.increment_commitment(pp, step_circuit);
    }
    z0 = rs.verify(pp, rs.num_steps(), &z0, IC_i)?;
    snarks.push(rs);
    Ok(())
  };

  // Network logic
  for _ in 0..num_nodes {
    node()?;
  }
  Ok(snarks)
}

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
    );
    let pp: PublicParams<E> =
      PublicParams::setup(&verifier_circuit, &*default_ck_hint(), &*default_ck_hint());
    let (circuit_shape_F, ck, digest_F) = pp_F.into_shape_ck_digest();
    Self {
      pp,
      circuit_shape_F,
      digest_F,
      ck: (*ck).clone(),
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
    let folding_data = Layer2FoldingData::new(
      Some(pp.digest_F),
      Some(nifs),
      Some(r_U.clone()),
      Some(U2.clone()),
      Some(E_new),
      Some(W_new),
      Some(U2_secondary.clone()),
    );
    let verifier_circuit: VerifierCircuit<E> = VerifierCircuit::new(
      pp.augmented_circuit_params(),
      pp.pp.ro_consts_circuit.clone(),
      Some(folding_data),
      Some(r_U_cyclefold.clone()),
    );
    let z0 = {
      let mut ro = <Dual<E> as Engine>::RO::new(
        pp.pp.ro_consts.clone(),
        (2 * NUM_FE_IN_EMULATED_POINT + 3) + (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS), // (U.comm_E + U.comm_W + U.X + U.u) + U_cyclefold
      );
      absorb_U::<E>(&r_U, &mut ro);
      absorb_U_bn(&r_U_cyclefold, &mut ro);
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
    let folding_data = Layer2FoldingData::new(
      Some(pp.digest_F),
      Some(nifs),
      Some(self.r_U.clone()),
      Some(U2.clone()),
      Some(E_new),
      Some(W_new),
      Some(U2_secondary.clone()),
    );
    let verifier_circuit: VerifierCircuit<E> = VerifierCircuit::new(
      pp.augmented_circuit_params(),
      pp.pp.ro_consts_circuit.clone(),
      Some(folding_data),
      Some(self.r_U_cyclefold.clone()),
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

#[derive(Clone)]
pub struct VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  folding_data: Option<Layer2FoldingData<E>>,
  U1_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
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

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    let U1_secondary = self.alloc_cyclefold_running_instance(cs.namespace(|| "U1_secondary"))?;
    let (pp_digest, U1, U2, E_new, W_new, U2_secondary, nifs) =
      VerifierCircuit::alloc_folding_data(
        cs.namespace(|| "alloc folding data"),
        &self.params,
        &self.folding_data,
      )?;

    // i/o hash check
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      (2 * NUM_FE_IN_EMULATED_POINT + 3) + (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS), // (U.W + U.comm_E + U.X + U.u) + U_cyclefold
    );
    U1.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;
    U1_secondary.absorb_in_ro(cs.namespace(|| "absorb U1_secondary"), &mut ro)?;
    let hash_U_bits = ro.squeeze(cs.namespace(|| "hash_U bits"), NUM_CHALLENGE_BITS)?;
    let hash_U = le_bits_to_num(cs.namespace(|| "hash_U"), &hash_U_bits)?;
    let expected_hash_U = z[0].clone();
    cs.enforce(
      || "hash_U == z0",
      |lc| lc + hash_U.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + expected_hash_U.get_variable(),
    );

    // Primary NIFS.V
    let (U, U_secondary) = nifs.verify(
      cs.namespace(|| "nifs"),
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
      &U1,
      &U2,
      &U1_secondary,
      &U2_secondary,
      &pp_digest,
      E_new,
      W_new,
    )?;

    // output hash
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      2 * NUM_FE_IN_EMULATED_POINT + 3 + (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS), // (U.W + U.comm_E + U.X + U.u) + U_cyclefold
    );
    U.absorb_in_ro(cs.namespace(|| "absorb folded U"), &mut ro)?;
    U_secondary.absorb_in_ro(cs.namespace(|| "absorb folded U_secondary"), &mut ro)?;
    let hash_U_bits = ro.squeeze(cs.namespace(|| "hash_folded_U bits"), NUM_CHALLENGE_BITS)?;
    let hash_U = le_bits_to_num(cs.namespace(|| "hash_folded_U"), &hash_U_bits)?;
    Ok(vec![hash_U])
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
    folding_data: Option<Layer2FoldingData<E>>,
    U1_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
  ) -> Self {
    Self {
      params,
      ro_consts,
      folding_data,
      U1_secondary,
    }
  }

  fn alloc_cyclefold_running_instance<CS>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    AllocatedRelaxedR1CSInstanceBn::alloc(
      cs.namespace(|| "U1_secondary"),
      self.U1_secondary.as_ref(),
      self.params.limb_width,
      self.params.n_limbs,
    )
  }
  fn alloc_folding_data<CS>(
    mut cs: CS,
    params: &AugmentedCircuitParams,
    folding_data: &Option<Layer2FoldingData<E>>,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,                                 // pp_digest
      AllocatedEmulRelaxedR1CSInstance<Dual<E>>,               // U1
      AllocatedEmulRelaxedR1CSInstance<Dual<E>>,               // U2
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,             // E_new
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,             // W_new
      AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>, // U2_secondary
      NIFSVerifierGadget<E>,                                   // nifs
    ),
    SynthesisError,
  >
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    // Primary folding data
    let pp_digest = AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || {
      Ok(
        folding_data
          .as_ref()
          .and_then(|data| data.pp_digest)
          .map_or(E::Scalar::ZERO, |pp_digest| pp_digest),
      )
    })?;
    let U1 = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      folding_data.as_ref().and_then(|data| data.U1.as_ref()),
      params.limb_width,
      params.n_limbs,
    )?;
    let U2 = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      folding_data.as_ref().and_then(|data| data.U2.as_ref()),
      params.limb_width,
      params.n_limbs,
    )?;
    let nifs_primary = PrimaryNIFSVerifierGadget::alloc(
      cs.namespace(|| "primary_nifs"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_primary),
      params.limb_width,
      params.n_limbs,
    )?;
    let E_new = AllocatedEmulPoint::alloc(
      cs.namespace(|| "E_new"),
      folding_data
        .as_ref()
        .and_then(|data| data.E_new)
        .map(|E_new| E_new.to_coordinates()),
      params.limb_width,
      params.n_limbs,
    )?;
    let W_new = AllocatedEmulPoint::alloc(
      cs.namespace(|| "W_new"),
      folding_data
        .as_ref()
        .and_then(|data| data.W_new)
        .map(|W_new| W_new.to_coordinates()),
      params.limb_width,
      params.n_limbs,
    )?;

    // First CycleFold data
    let nifs_E1 = CycleFoldNIFSVerifierGadget::alloc(
      cs.namespace(|| "nifs_E1"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_E1),
    )?;
    let l_u_cyclefold_E1 = AllocatedCycleFoldInstance::alloc(
      cs.namespace(|| "l_u_cyclefold_E1"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.l_u_cyclefold_E1),
      params.limb_width,
      params.n_limbs,
    )?;

    // Second CycleFold data
    let nifs_E2 = CycleFoldNIFSVerifierGadget::alloc(
      cs.namespace(|| "nifs_E2"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_E2),
    )?;
    let l_u_cyclefold_E2 = AllocatedCycleFoldInstance::alloc(
      cs.namespace(|| "l_u_cyclefold_E2"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.l_u_cyclefold_E2),
      params.limb_width,
      params.n_limbs,
    )?;

    // Third CycleFold data
    let nifs_W = CycleFoldNIFSVerifierGadget::alloc(
      cs.namespace(|| "nifs_W"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_W),
    )?;
    let l_u_cyclefold_W = AllocatedCycleFoldInstance::alloc(
      cs.namespace(|| "l_u_cyclefold_W"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.l_u_cyclefold_W),
      params.limb_width,
      params.n_limbs,
    )?;

    // fourth CycleFold data
    let U2_secondary = AllocatedRelaxedR1CSInstanceBn::alloc(
      cs.namespace(|| "U2_secondary"),
      folding_data
        .as_ref()
        .and_then(|data| data.U2_secondary.as_ref()),
      params.limb_width,
      params.n_limbs,
    )?;
    let nifs_final_cyclefold = CycleFoldRelaxedNIFSVerifierGadget::alloc(
      cs.namespace(|| "nifs_final_cyclefold"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_final_cyclefold),
    )?;
    let nifs = NIFSVerifierGadget {
      nifs_primary,
      nifs_E1,
      nifs_E2,
      nifs_W,
      nifs_final_cyclefold,
      l_u_cyclefold_E1,
      l_u_cyclefold_E2,
      l_u_cyclefold_W,
    };
    Ok((pp_digest, U1, U2, E_new, W_new, U2_secondary, nifs))
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

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
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
