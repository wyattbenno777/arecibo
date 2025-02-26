//! Nebula API

use super::{
  ic::increment_ic,
  product_circuits::{convert_advice_separate, BatchedOpsCircuit, OpsCircuit, ScanCircuit},
};
use crate::{
  digest::{DigestComputer, SimpleDigestible},
  hypernova::{
    nebula::product_circuits::MEMORY_OPS_PER_STEP,
    pp::{
      compute_ck, AuxPublicParams, PublicParamsTrait, R1CSPublicParams, SplitPublicParams,
      SubAuxPublicParams,
    },
    rs::{IncrementalCommitment, RecursiveSNARK, StepCircuit},
  },
  traits::{snark::default_ck_hint, CurveCycleEquipped, Engine, TranscriptEngineTrait},
  NovaError,
};
use ff::Field;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

/// Public parameters for the Nebula SNARK
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NebulaPublicParams<E>
where
  E: CurveCycleEquipped,
{
  aux: AuxPublicParams<E>,
  F: R1CSPublicParams<E>,
  ops: R1CSPublicParams<E>,
  scan: R1CSPublicParams<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E> NebulaPublicParams<E>
where
  E: CurveCycleEquipped,
{
  fn F(&self) -> SplitPublicParams<'_, E> {
    (&self.aux, &self.F, self.digest())
  }

  fn ops(&self) -> SplitPublicParams<'_, E> {
    (&self.aux, &self.ops, self.digest())
  }

  fn scan(&self) -> SplitPublicParams<'_, E> {
    (&self.aux, &self.scan, self.digest())
  }

  /// Calculate the digest of the public parameters.
  pub fn digest(&self) -> E::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure in retrieving digest")
  }
}

impl<E> SimpleDigestible for NebulaPublicParams<E> where E: CurveCycleEquipped {}

/// A SNARK that proves correct execution of a vm and that the vm maintained
/// memory correctly.
pub struct NebulaSNARK<E>
where
  E: CurveCycleEquipped,
{
  F: RecursiveSNARK<E>,
  ops: RecursiveSNARK<E>,
  scan: RecursiveSNARK<E>,
}

impl<E> NebulaSNARK<E>
where
  E: CurveCycleEquipped,
  <E as Engine>::Scalar: PartialOrd,
{
  /// Fn used to obtain setup material for producing succinct arguments for
  /// WASM program executions
  pub fn setup(F: &impl StepCircuit<E::Scalar>, step_size: StepSize) -> NebulaPublicParams<E> {
    let sub_aux_params = SubAuxPublicParams::<E>::setup(&*default_ck_hint());
    let F_pp = R1CSPublicParams::<E>::setup(F, &sub_aux_params);
    let ops_pp = R1CSPublicParams::<E>::setup(
      &BatchedOpsCircuit::empty(step_size.execution),
      &sub_aux_params,
    );
    let scan_pp =
      R1CSPublicParams::<E>::setup(&ScanCircuit::empty(step_size.memory), &sub_aux_params);
    let ck = compute_ck(
      &[
        &F_pp.circuit_shape,
        &ops_pp.circuit_shape,
        &scan_pp.circuit_shape,
      ],
      &*default_ck_hint(),
    );
    let aux_pp = AuxPublicParams::setup(ck, sub_aux_params);
    NebulaPublicParams {
      aux: aux_pp,
      F: F_pp,
      ops: ops_pp,
      scan: scan_pp,
      digest: OnceCell::new(),
    }
  }

  /// Produce a SNARK that proves correct execution of a vm and that the vm maintained
  /// memory correctly.
  pub fn prove(
    pp: &NebulaPublicParams<E>,
    step_size: StepSize,
    vm_multi_sets: VMMultiSets,
    F_engine: impl RecursiveSNARKEngine<E>,
  ) -> Result<(Self, NebulaInstance<E>), NovaError> {
    let (init_memory, final_memory, read_ops, write_ops) = vm_multi_sets;

    // --- Run the F (transition) circuit ---
    //
    // We use commitment-carrying IVC to prove the repeated execution of F
    let (F_rs, F_ic, F_z_0) = RecursiveSNARKEngine::run(|| F_engine, &pp.F())?;

    // --- Get challenges gamma and alpha ---
    //
    // * Compute commitment to IS & FS -> IC_Audit
    // * hash ic_ops & ic_scan and get gamma, alpha
    let (gamma, alpha) = Self::gamma_alpha(
      &pp.scan(),
      &init_memory,
      &final_memory,
      F_ic.0,
      step_size.memory,
    )?;

    // Grand product checks for RS & WS
    let (ops_rs, ops_ic, ops_z_0) = RecursiveSNARKEngine::run(
      || OpsGrandProductEngine::new(read_ops, write_ops, gamma, alpha, step_size),
      &pp.ops(),
    )?;

    // Grand product checks for IS & FS
    let (scan_rs, scan_ic, scan_z_0) = RecursiveSNARKEngine::run(
      || ScanGrandProductEngine::new(init_memory, final_memory, gamma, alpha, step_size),
      &pp.scan(),
    )?;
    Ok((
      Self {
        F: F_rs,
        ops: ops_rs,
        scan: scan_rs,
      },
      NebulaInstance {
        F_z_0,
        F_ic,
        ops_z_0,
        ops_ic,
        scan_z_0,
        scan_ic,
      },
    ))
  }

  /// Verify the [`NebulaSNARK`]
  pub fn verify(&self, pp: &NebulaPublicParams<E>, U: &NebulaInstance<E>) -> Result<(), NovaError> {
    // verify F
    self
      .F
      .verify(&pp.F(), self.F.num_steps(), &U.F_z_0, U.F_ic)?;

    // verify F_ops
    let ops_z_i = self
      .ops
      .verify(&pp.ops(), self.ops.num_steps(), &U.ops_z_0, U.ops_ic)?;

    // verify F_scan
    let scan_z_i = self
      .scan
      .verify(&pp.scan(), self.scan.num_steps(), &U.scan_z_0, U.scan_ic)?;

    // 1. check h_IS = h_RS = h_WS = h_FS = 1 // initial values are correct
    let (init_h_is, init_h_rs, init_h_ws, init_h_fs) =
      { (U.scan_z_0[2], U.ops_z_0[3], U.ops_z_0[4], U.scan_z_0[3]) };
    if init_h_is != E::Scalar::ONE
      || init_h_rs != E::Scalar::ONE
      || init_h_ws != E::Scalar::ONE
      || init_h_fs != E::Scalar::ONE
    {
      return Err(NovaError::InvalidMultisetProof);
    }

    // TODO: implement other multiset check

    // --- 4. check h_IS' · h_WS' = h_RS' · h_FS'.---
    //
    // Inputs for multiset check
    let (h_is, h_rs, h_ws, h_fs) = { (scan_z_i[2], ops_z_i[3], ops_z_i[4], scan_z_i[3]) };
    if h_is * h_ws != h_rs * h_fs {
      return Err(NovaError::InvalidMultisetProof);
    }

    Ok(())
  }

  fn gamma_alpha(
    pp: &impl PublicParamsTrait<E>,
    init_memory: &[(usize, u64, u64)],
    final_memory: &[(usize, u64, u64)],
    ic_F: E::Scalar,
    memory_size: usize,
  ) -> Result<(E::Scalar, E::Scalar), NovaError> {
    let mut ic_scan = IncrementalCommitment::<E>::default();
    for (init_memory_chunk, final_memory_chunk) in init_memory
      .chunks(memory_size)
      .zip_eq(final_memory.chunks(memory_size))
    {
      ic_scan = increment_ic::<E>(
        pp.ck(),
        pp.ro_consts(),
        ic_scan,
        (
          &convert_advice_separate(init_memory_chunk),
          &convert_advice_separate(final_memory_chunk),
        ),
        &pp.circuit_shape().r1cs_shape,
      );
    }
    let mut keccak = E::TE::new(b"compute MCC challenges");
    keccak.absorb(b"C_n", &ic_F);
    keccak.absorb(b"C_pprime", &ic_scan.0);
    keccak.absorb(b"C_pprime", &ic_scan.1);
    let gamma = keccak.squeeze(b"gamma")?;
    let alpha = keccak.squeeze(b"alpha")?;
    Ok((gamma, alpha))
  }
}

/// A trait that encapsulates the common steps for a recursive SNARK prover:
/// 1. Building its circuits,
/// 2. Providing its initial input, and
/// 3. Running the recursive proving loop.
pub trait RecursiveSNARKEngine<E>
where
  E: CurveCycleEquipped,
  Self: Sized,
{
  /// Type of circuit to prove
  type Circuit: StepCircuit<E::Scalar>;

  /// Build the circuits that will be used by the recursive SNARK.
  /// (A mutable reference is required if the implementation consumes internal
  /// data.)
  fn circuits(&mut self) -> Result<Vec<Self::Circuit>, NovaError>;

  /// Return the initial input vector for the recursive SNARK.
  fn z0(&self) -> Vec<E::Scalar>;

  /// Run the recursive proving loop over the built circuits.
  fn prove_recursive(
    &mut self,
    pp: &impl PublicParamsTrait<E>,
  ) -> Result<(RecursiveSNARK<E>, IncrementalCommitment<E>, Vec<E::Scalar>), NovaError> {
    let circuits = self.circuits()?;
    let z_0 = self.z0();
    let first = circuits.first().ok_or(NovaError::NoCircuit)?;
    let mut rs = RecursiveSNARK::new(pp, first, &z_0)?;
    let mut ic = IncrementalCommitment::<E>::default();
    for circuit in circuits.iter() {
      rs.prove_step(pp, circuit, ic)?;
      let (advice_0, advice_1) = circuit.advice();
      ic = increment_ic::<E>(
        pp.ck(),
        pp.ro_consts(),
        ic,
        (&advice_0, &advice_1),
        &pp.circuit_shape().r1cs_shape,
      );
    }
    Ok((rs, ic, z_0))
  }

  /// Run the engine
  fn run(
    constructor: impl FnOnce() -> Self,
    pp: &impl PublicParamsTrait<E>,
  ) -> Result<(RecursiveSNARK<E>, IncrementalCommitment<E>, Vec<E::Scalar>), NovaError> {
    let mut engine = constructor();
    engine.prove_recursive(pp)
  }
}

struct OpsGrandProductEngine<E>
where
  E: CurveCycleEquipped,
{
  RS: Vec<Vec<(usize, u64, u64)>>,
  WS: Vec<Vec<(usize, u64, u64)>>,
  gamma: E::Scalar,
  alpha: E::Scalar,
  step_size: StepSize,
}

impl<E> OpsGrandProductEngine<E>
where
  E: CurveCycleEquipped,
{
  fn new(
    RS: Vec<Vec<(usize, u64, u64)>>,
    WS: Vec<Vec<(usize, u64, u64)>>,
    gamma: E::Scalar,
    alpha: E::Scalar,
    step_size: StepSize,
  ) -> Self {
    Self {
      RS,
      WS,
      gamma,
      alpha,
      step_size,
    }
  }
}

impl<E> RecursiveSNARKEngine<E> for OpsGrandProductEngine<E>
where
  E: CurveCycleEquipped,
  <E as Engine>::Scalar: PartialOrd,
{
  type Circuit = BatchedOpsCircuit;

  fn circuits(&mut self) -> Result<Vec<Self::Circuit>, NovaError> {
    // Build OpsCircuit from the stored RS and WS multisets.
    let circuits = self
      .RS
      .iter()
      .zip_eq(self.WS.iter())
      .map(|(rs, ws)| OpsCircuit::new(rs.clone(), ws.clone()))
      .collect_vec();
    Ok(
      circuits
        .chunks(self.step_size.execution)
        .map(|chunk| BatchedOpsCircuit::new(chunk.to_vec()))
        .collect::<Vec<_>>(),
    )
  }

  fn z0(&self) -> Vec<E::Scalar> {
    // The ops RS initial input is [gamma, alpha, ts=0, h_RS=1, h_WS=1, size]
    vec![
      self.gamma,
      self.alpha,
      E::Scalar::ZERO,
      E::Scalar::ONE,
      E::Scalar::ONE,
      E::Scalar::from((MEMORY_OPS_PER_STEP / 2) as u64),
    ]
  }
}

struct ScanGrandProductEngine<E>
where
  E: CurveCycleEquipped,
{
  IS: Vec<(usize, u64, u64)>,
  FS: Vec<(usize, u64, u64)>,
  gamma: E::Scalar,
  alpha: E::Scalar,
  step_size: StepSize,
}

impl<E> ScanGrandProductEngine<E>
where
  E: CurveCycleEquipped,
{
  fn new(
    IS: Vec<(usize, u64, u64)>,
    FS: Vec<(usize, u64, u64)>,
    gamma: E::Scalar,
    alpha: E::Scalar,
    step_size: StepSize,
  ) -> Self {
    Self {
      IS,
      FS,
      gamma,
      alpha,
      step_size,
    }
  }
}

impl<E> RecursiveSNARKEngine<E> for ScanGrandProductEngine<E>
where
  E: CurveCycleEquipped,
{
  type Circuit = ScanCircuit;

  fn circuits(&mut self) -> Result<Vec<Self::Circuit>, NovaError> {
    // Build ScanCircuit from the stored IS and FS multisets.
    let circuits = self
      .IS
      .chunks(self.step_size.memory)
      .zip_eq(self.FS.chunks(self.step_size.memory))
      .map(|(is_chunk, fs_chunk)| ScanCircuit::new(is_chunk.to_vec(), fs_chunk.to_vec()))
      .collect();
    Ok(circuits)
  }

  fn z0(&self) -> Vec<E::Scalar> {
    // scan_z0 = [gamma, alpha, h_IS=1, h_FS=1, size]
    vec![
      self.gamma,
      self.alpha,
      E::Scalar::ONE,
      E::Scalar::ONE,
      E::Scalar::from(self.step_size.memory as u64),
    ]
  }
}

/// Public i/o for WASM execution proving
#[derive(Clone, Debug)]
pub struct NebulaInstance<E>
where
  E: CurveCycleEquipped,
{
  F_z_0: Vec<E::Scalar>,
  F_ic: IncrementalCommitment<E>,
  ops_z_0: Vec<E::Scalar>,
  ops_ic: IncrementalCommitment<E>,
  scan_z_0: Vec<E::Scalar>,
  scan_ic: IncrementalCommitment<E>,
}

// IS, FS, RS, WS
type VMMultiSets = (
  Vec<(usize, u64, u64)>,
  Vec<(usize, u64, u64)>,
  Vec<Vec<(usize, u64, u64)>>,
  Vec<Vec<(usize, u64, u64)>>,
);

/// Step size of used for zkVM execution
#[derive(Clone, Debug, Copy)]
pub struct StepSize {
  /// How many opcodes to execute per recursive step
  pub execution: usize,
  /// How many memory addresses to audit per recursive step
  pub memory: usize,
}

impl StepSize {
  /// Create a new instance of [`StepSize`]
  ///
  /// Sets both execution and memory step size to `step_size`
  pub fn new(step_size: usize) -> Self {
    Self {
      execution: step_size,
      memory: step_size,
    }
  }

  /// Set the memory step size
  ///
  /// Returns a modified instance of [`StepSize`]
  pub fn set_memory_step_size(mut self, memory: usize) -> Self {
    self.memory = memory;
    self
  }
}
