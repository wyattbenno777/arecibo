//! This module defines the Nebula API. A CC-IVC scheme that proves the correct execution of a program
//! and that the program maintained memory correctly.

use super::{
  ic::increment_ic,
  product_circuits::{convert_advice_separate, BatchedOpsCircuit, OpsCircuit, ScanCircuit},
};
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  digest::{DigestComputer, SimpleDigestible},
  hypernova::{
    compression::{CompressedSNARK, ProverKey, VerifierKey},
    pp::{AuxPublicParams, PublicParamsTrait, R1CSPublicParams, SplitPublicParams},
    rs::{IncrementalCommitment, RecursiveSNARK, StepCircuit},
  },
  traits::{
    snark::{default_ck_hint, LinearizedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    CurveCycleEquipped, Dual, Engine, ROConstantsCircuit, TranscriptEngineTrait,
  },
  AugmentedCircuitParams, NovaError,
};
use ff::Field;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

/// Public parameters for the Nebula SNARK
///
/// /// The constant `M` is the number of memory operations per step in the vm.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NebulaPublicParams<E, S1, S2, const M: usize>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  aux: AuxPublicParams<E>,
  F: R1CSPublicParams<E>,
  ops: R1CSPublicParams<E>,
  scan: R1CSPublicParams<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
  #[serde(skip, default = "OnceCell::new")]
  pk_and_vk: OnceCell<(NebulaProverKey<E, S1, S2>, NebulaVerifierKey<E, S1, S2>)>,
}

impl<E, S1, S2, const M: usize> NebulaPublicParams<E, S1, S2, M>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
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

  /// provides a reference to a ProverKey suitable for producing a CompressedProof
  pub fn pk(&self) -> &NebulaProverKey<E, S1, S2> {
    let (pk, _vk) = self
      .pk_and_vk
      .get_or_init(|| NebulaCompressedSNARK::<E, S1, S2>::setup(self).unwrap());
    pk
  }

  /// provides a reference to a VerifierKey suitable for verifying a CompressedProof
  pub fn vk(&self) -> &NebulaVerifierKey<E, S1, S2> {
    let (_pk, vk) = self
      .pk_and_vk
      .get_or_init(|| NebulaCompressedSNARK::<E, S1, S2>::setup(self).unwrap());
    vk
  }
}

impl<E, S1, S2, const M: usize> SimpleDigestible for NebulaPublicParams<E, S1, S2, M>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
}

/// Nebula Prover key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NebulaProverKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  F: ProverKey<E, S1, S2>,
  ops: ProverKey<E, S1, S2>,
  scan: ProverKey<E, S1, S2>,
}

/// Nebula Verifier key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NebulaVerifierKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  F: VerifierKey<E, S1, S2>,
  ops: VerifierKey<E, S1, S2>,
  scan: VerifierKey<E, S1, S2>,
}

/// Apply Spartan to prove knowledge of a valid Nebula IVC proof
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NebulaCompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  F: CompressedSNARK<E, S1, S2>,
  ops: CompressedSNARK<E, S1, S2>,
  scan: CompressedSNARK<E, S1, S2>,
}

impl<E, S1, S2> NebulaCompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// Setup the Prover and Verifier keys for the Nebula SNARK
  pub fn setup<const M: usize>(
    pp: &NebulaPublicParams<E, S1, S2, M>,
  ) -> Result<(NebulaProverKey<E, S1, S2>, NebulaVerifierKey<E, S1, S2>), NovaError> {
    let (F_pk, F_vk) = CompressedSNARK::<E, S1, S2>::setup(&pp.F())?;
    let (ops_pk, ops_vk) = CompressedSNARK::<E, S1, S2>::setup(&pp.ops())?;
    let (scan_pk, scan_vk) = CompressedSNARK::<E, S1, S2>::setup(&pp.scan())?;
    Ok((
      NebulaProverKey {
        F: F_pk,
        ops: ops_pk,
        scan: scan_pk,
      },
      NebulaVerifierKey {
        F: F_vk,
        ops: ops_vk,
        scan: scan_vk,
      },
    ))
  }

  /// produce a compressed proof for the Nebula SNARK
  pub fn prove<const M: usize>(
    pp: &NebulaPublicParams<E, S1, S2, M>,
    pk: &NebulaProverKey<E, S1, S2>,
    rs: &NebulaRecursiveSNARK<E>,
  ) -> Result<NebulaCompressedSNARK<E, S1, S2>, NovaError> {
    let F = CompressedSNARK::prove(&pp.F(), &pk.F, &rs.F)?;
    let ops = CompressedSNARK::prove(&pp.ops(), &pk.ops, &rs.ops)?;
    let scan = CompressedSNARK::prove(&pp.scan(), &pk.scan, &rs.scan)?;
    Ok(NebulaCompressedSNARK { F, ops, scan })
  }

  /// verify a compressed proof for the Nebula SNARK
  pub fn verify(
    &self,
    vk: &NebulaVerifierKey<E, S1, S2>,
    U: &NebulaInstance<E>,
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>), NovaError> {
    self.F.verify(&vk.F, &U.F_z_0, U.F_num_steps)?;
    let ops_z_i = self.ops.verify(&vk.ops, &U.ops_z_0, U.ops_num_steps)?;
    let scan_z_i = self.scan.verify(&vk.scan, &U.scan_z_0, U.scan_num_steps)?;
    Ok((ops_z_i, scan_z_i))
  }
}

/// A SNARK that proves correct execution of a vm and that the vm maintained
/// memory correctly.
///
/// The constant `M` is the number of memory operations per step in the vm.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum NebulaSNARK<E, S1, S2, const M: usize>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// RecursiveSNARK for vm execution
  Recursive(Box<NebulaRecursiveSNARK<E>>),
  /// CompressedSNARK for vm execution
  Compressed(Box<NebulaCompressedSNARK<E, S1, S2>>),
}

/// A SNARK that proves correct execution of a vm and that the vm maintained
/// memory correctly.
///
/// The constant `M` is the number of memory operations per step in the vm.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NebulaRecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  F: RecursiveSNARK<E>,
  ops: RecursiveSNARK<E>,
  scan: RecursiveSNARK<E>,
}

impl<E, S1, S2, const M: usize> NebulaSNARK<E, S1, S2, M>
where
  E: CurveCycleEquipped,
  <E as Engine>::Scalar: PartialOrd,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// Fn used to obtain setup material for producing succinct arguments for
  /// WASM program executions
  pub fn setup(
    F: &impl StepCircuit<E::Scalar>,
    step_size: StepSize,
  ) -> NebulaPublicParams<E, S1, S2, M> {
    let ro_consts_circuit = ROConstantsCircuit::<Dual<E>>::default();
    let augmented_circuit_params = AugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS);
    let F_pp = R1CSPublicParams::<E>::setup(F, &ro_consts_circuit, &augmented_circuit_params);
    let ops_pp = R1CSPublicParams::<E>::setup(
      &BatchedOpsCircuit::empty::<M>(step_size.execution),
      &ro_consts_circuit,
      &augmented_circuit_params,
    );
    let scan_pp = R1CSPublicParams::<E>::setup(
      &ScanCircuit::empty(step_size.memory),
      &ro_consts_circuit,
      &augmented_circuit_params,
    );
    let aux_pp = AuxPublicParams::setup(
      &[
        &F_pp.circuit_shape,
        &ops_pp.circuit_shape,
        &scan_pp.circuit_shape,
      ],
      ro_consts_circuit,
      augmented_circuit_params,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    NebulaPublicParams {
      aux: aux_pp,
      F: F_pp,
      ops: ops_pp,
      scan: scan_pp,
      digest: OnceCell::new(),
      pk_and_vk: OnceCell::new(),
    }
  }

  /// Produce a SNARK that proves correct execution of a vm and that the vm maintained
  /// memory correctly.
  pub fn prove(
    pp: &NebulaPublicParams<E, S1, S2, M>,
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
      || OpsGrandProductEngine::<E, M>::new(read_ops, write_ops, gamma, alpha, step_size),
      &pp.ops(),
    )?;

    // Grand product checks for IS & FS
    let (scan_rs, scan_ic, scan_z_0) = RecursiveSNARKEngine::run(
      || ScanGrandProductEngine::new(init_memory, final_memory, gamma, alpha, step_size),
      &pp.scan(),
    )?;
    let U = NebulaInstance {
      F_z_0,
      F_ic,
      F_num_steps: F_rs.num_steps(),
      ops_z_0,
      ops_ic,
      ops_num_steps: ops_rs.num_steps(),
      scan_z_0,
      scan_ic,
      scan_num_steps: scan_rs.num_steps(),
    };
    Ok((
      Self::Recursive(Box::new(NebulaRecursiveSNARK {
        F: F_rs,
        ops: ops_rs,
        scan: scan_rs,
      })),
      U,
    ))
  }

  /// Apply Spartan on top of the Nebula IVC proofs
  pub fn compress(&self, pp: &NebulaPublicParams<E, S1, S2, M>) -> Result<Self, NovaError> {
    match self {
      Self::Recursive(rs) => Ok(Self::Compressed(Box::new(NebulaCompressedSNARK::prove(
        pp,
        pp.pk(),
        rs.as_ref(),
      )?))),
      Self::Compressed(..) => Err(NovaError::NotRecursive),
    }
  }

  /// Verify the [`NebulaSNARK`]
  pub fn verify(
    &self,
    pp: &NebulaPublicParams<E, S1, S2, M>,
    U: &NebulaInstance<E>,
  ) -> Result<(), NovaError> {
    let (ops_z_i, scan_z_i) = match self {
      Self::Recursive(rs) => {
        // verify F
        rs.F.verify(&pp.F(), rs.F.num_steps(), &U.F_z_0, U.F_ic)?;

        // verify F_ops
        let ops_z_i = rs
          .ops
          .verify(&pp.ops(), rs.ops.num_steps(), &U.ops_z_0, U.ops_ic)?;

        // verify F_scan
        let scan_z_i = rs
          .scan
          .verify(&pp.scan(), rs.scan.num_steps(), &U.scan_z_0, U.scan_ic)?;

        (ops_z_i, scan_z_i)
      }
      Self::Compressed(spartan) => spartan.verify(pp.vk(), U)?,
    };
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

    // --- 2. check Cn′ = Cn  ---
    //
    // commitments carried in both Πops and ΠF are the same
    if U.F_ic != U.ops_ic {
      return Err(NovaError::InvalidMultisetProof);
    }

    // --- 3. check γ and γ are derived by hashing C and C′′. ---
    //
    // Get alpha and gamma
    let mut keccak = E::TE::new(b"compute MCC challenges");
    keccak.absorb(b"ic_ops", &U.F_ic.0);
    keccak.absorb(b"ic_is", &U.scan_ic.0);
    keccak.absorb(b"ic_fs", &U.scan_ic.1);
    let gamma = keccak.squeeze(b"gamma")?;
    let alpha = keccak.squeeze(b"alpha")?;

    if U.ops_z_0[0] != gamma || U.ops_z_0[1] != alpha {
      return Err(NovaError::InvalidMultisetProof);
    }

    // --- 4. check h_IS' · h_WS' = h_RS' · h_FS'.---
    //
    // Inputs for multiset check
    let (h_is, h_rs, h_ws, h_fs) = { (scan_z_i[2], ops_z_i[3], ops_z_i[4], scan_z_i[3]) };
    if h_is * h_ws != h_rs * h_fs {
      return Err(NovaError::InvalidMultisetProof);
    }

    Ok(())
  }

  // Get MCC challenges for grand products
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
      );
    }
    let mut keccak = E::TE::new(b"compute MCC challenges");
    keccak.absorb(b"ic_ops", &ic_F);
    keccak.absorb(b"ic_is", &ic_scan.0);
    keccak.absorb(b"ic_fs", &ic_scan.1);
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
      ic = increment_ic::<E>(pp.ck(), pp.ro_consts(), ic, (&advice_0, &advice_1));
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

struct OpsGrandProductEngine<E, const M: usize>
where
  E: CurveCycleEquipped,
{
  RS: Vec<Vec<(usize, u64, u64)>>,
  WS: Vec<Vec<(usize, u64, u64)>>,
  gamma: E::Scalar,
  alpha: E::Scalar,
  step_size: StepSize,
}

impl<E, const M: usize> OpsGrandProductEngine<E, M>
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

impl<E, const M: usize> RecursiveSNARKEngine<E> for OpsGrandProductEngine<E, M>
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
    // The ops RS initial input is [gamma, alpha, ts=0, h_RS=1, h_WS=1, ms_size]
    vec![
      self.gamma,
      self.alpha,
      E::Scalar::ZERO,
      E::Scalar::ONE,
      E::Scalar::ONE,
      E::Scalar::from(M as u64),
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
    // scan_z0 = [gamma, alpha, h_IS=1, h_FS=1, ms_size]
    vec![
      self.gamma,
      self.alpha,
      E::Scalar::ONE,
      E::Scalar::ONE,
      E::Scalar::from(self.step_size.memory as u64),
    ]
  }
}

/// Public i/o for a Nebula zkVM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NebulaInstance<E>
where
  E: CurveCycleEquipped,
{
  F_z_0: Vec<E::Scalar>,
  F_ic: IncrementalCommitment<E>,
  F_num_steps: usize,
  ops_z_0: Vec<E::Scalar>,
  ops_ic: IncrementalCommitment<E>,
  ops_num_steps: usize,
  scan_z_0: Vec<E::Scalar>,
  scan_ic: IncrementalCommitment<E>,
  scan_num_steps: usize,
}

// IS, FS, RS, WS
type VMMultiSets = (
  Vec<(usize, u64, u64)>,
  Vec<(usize, u64, u64)>,
  Vec<Vec<(usize, u64, u64)>>,
  Vec<Vec<(usize, u64, u64)>>,
);

/// Step size of used for zkVM execution
#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
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
