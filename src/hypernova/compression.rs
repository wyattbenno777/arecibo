//! This module provides the components needed to compress the HyperNova IVC proofs with Spartan.

use super::{pp::PublicParamsTrait, rs::RecursiveSNARK};
use crate::{
  hypernova::nifs::PartialNIFS,
  r1cs::{
    split::{LR1CSInstance, SplitR1CSInstance, SplitR1CSWitness},
    RelaxedR1CSInstance, RelaxedR1CSWitness,
  },
  traits::{
    commitment::CommitmentEngineTrait,
    snark::{LinearizedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    CurveCycleEquipped, Dual, Engine, ROConstants,
  },
  DerandKey, NovaError,
};
use serde::{Deserialize, Serialize};

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  pk: S1::ProverKey,
  pk_cyclefold: S2::ProverKey,
}

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  vk: S1::VerifierKey,
  vk_cyclefold: S2::VerifierKey,
  dk: DerandKey<E>,
  dk_cyclefold: DerandKey<Dual<E>>,
  num_rounds: usize,
  ro_consts: ROConstants<Dual<E>>,
  pp_digest: E::Scalar,
}

/// A SNARK that proves the knowledge of a valid HyperNova [`RecursiveSNARK`]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  snark: S1,
  snark_cyclefold: S2,
  nifs: PartialNIFS<E>,
  data: CompressedSNARKData<E>,
}

impl<E, S1, S2> CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// Produce the [`ProverKey`] and [`VerifierKey`] for the [`CompressedSNARK`]
  pub fn setup(
    pp: &impl PublicParamsTrait<E>,
  ) -> Result<(ProverKey<E, S1, S2>, VerifierKey<E, S1, S2>), NovaError> {
    let (pk, vk) = S1::setup(pp.ck().clone(), &pp.circuit_shape().r1cs_shape)?;
    let (pk_cyclefold, vk_cyclefold) = S2::setup(
      pp.ck_cyclefold().clone(),
      &pp.circuit_shape_cyclefold().r1cs_shape,
    )?;
    let prover_key = ProverKey { pk, pk_cyclefold };
    let verifier_key = VerifierKey {
      vk,
      vk_cyclefold,
      dk: E::CE::derand_key(pp.ck()),
      dk_cyclefold: <Dual<E> as Engine>::CE::derand_key(pp.ck_cyclefold()),
      num_rounds: pp.num_rounds(),
      ro_consts: pp.ro_consts().clone(),
      pp_digest: pp.digest(),
    };
    Ok((prover_key, verifier_key))
  }

  /// Produce a [`CompressedSNARK`]
  pub fn prove(
    pp: &impl PublicParamsTrait<E>,
    pk: &ProverKey<E, S1, S2>,
    rs: &RecursiveSNARK<E>,
  ) -> Result<Self, NovaError> {
    // Fold r_U and l_u & derand the commitments to the witness
    let (nifs, (U, W), (U_cyclefold, W_cyclefold), data) = Self::nifs_derand(pp, rs)?;

    // Apply Spartan on primary and secondary curve for instance witness pairs:
    // (new_U, new_W) & (r_U_cyclefold, r_W_cyclefold)
    let (snark, snark_cyclefold) = rayon::join(
      || S1::prove(pp.ck(), &pk.pk, &pp.circuit_shape().r1cs_shape, &U, &W),
      || {
        S2::prove(
          pp.ck_cyclefold(),
          &pk.pk_cyclefold,
          &pp.circuit_shape_cyclefold().r1cs_shape,
          &U_cyclefold,
          &W_cyclefold,
        )
      },
    );
    Ok(Self {
      snark: snark?,
      snark_cyclefold: snark_cyclefold?,
      nifs,
      data,
    })
  }

  /// Verify a [`CompressedSNARK`] with the provided [`VerifierKey`]
  pub fn verify(
    &self,
    vk: &VerifierKey<E, S1, S2>,
    z_0: &[E::Scalar],
    num_steps: usize,
  ) -> Result<Vec<E::Scalar>, NovaError> {
    // --- Basic checks ---
    //
    // 1. check if the (relaxed) R1CS instances have two public outputs
    if self.data.r_U.X.len() != 2 || self.data.l_u.aux.X.len() != 2 {
      return Err(NovaError::ProofVerifyError);
    }
    // 2. Hash check:
    //    check if the output hashes in R1CS instances point to the right running instances
    RecursiveSNARK::hash_check(
      &vk.ro_consts,
      vk.pp_digest,
      num_steps,
      z_0,
      &self.data.z_i,
      &self.data.r_U,
      self.data.prev_ic,
      &self.data.l_u,
      &self.data.r_U_cyclefold,
    )?;

    // Verify NIFS and Spartan proofs
    let (U, U_cyclefold) = self.verify_nifs_derand(vk)?;
    let (res, res_cyclefold) = rayon::join(
      || self.snark.verify(&vk.vk, &U),
      || self.snark_cyclefold.verify(&vk.vk_cyclefold, &U_cyclefold),
    );
    res?;
    res_cyclefold?;
    Ok(self.data.z_i.clone())
  }
}

impl<E, S1, S2> CompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: LinearizedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  fn nifs_derand(
    pp: &impl PublicParamsTrait<E>,
    rs: &RecursiveSNARK<E>,
  ) -> Result<
    (
      PartialNIFS<E>,
      (LR1CSInstance<E>, SplitR1CSWitness<E>),
      (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>),
      CompressedSNARKData<E>,
    ),
    NovaError,
  > {
    // Fold r_U and l_u
    let (nifs, (U, W), _) = PartialNIFS::prove(
      &pp.circuit_shape().r1cs_shape,
      pp.ro_consts(),
      &pp.digest(),
      (&rs.r_U, &rs.r_W),
      (&rs.l_u, &rs.l_w),
    )?;

    // --- Derand the commitments to the witness ---
    // (U, W)
    let (derand_W, wit_blind) = W.derandomize();
    let derand_U = U.derandomize(&E::CE::derand_key(pp.ck()), &wit_blind);
    // (U_cyclefold, W_cyclefold)
    let (derand_W_cyclefold, wit_blind_cyclefold, err_blind_cyclefold) =
      rs.r_W_cyclefold.derandomize();
    let derand_U_cyclefold = rs.r_U_cyclefold.derandomize(
      &<Dual<E> as Engine>::CE::derand_key(pp.ck_cyclefold()),
      &wit_blind_cyclefold,
      &err_blind_cyclefold,
    );
    Ok((
      nifs,
      (derand_U, derand_W),
      (derand_U_cyclefold, derand_W_cyclefold),
      CompressedSNARKData {
        r_U: rs.r_U.clone(),
        l_u: rs.l_u.clone(),
        z_i: rs.z_i.clone(),
        prev_ic: rs.prev_ic,
        wit_blind,
        r_U_cyclefold: rs.r_U_cyclefold.clone(),
        wit_blind_cyclefold,
        err_blind_cyclefold,
      },
    ))
  }

  fn verify_nifs_derand(
    &self,
    vk: &VerifierKey<E, S1, S2>,
  ) -> Result<(LR1CSInstance<E>, RelaxedR1CSInstance<Dual<E>>), NovaError> {
    let U = self.nifs.verify(
      vk.num_rounds,
      &vk.ro_consts,
      &vk.pp_digest,
      &self.data.r_U,
      &self.data.l_u,
    )?;

    // --- Derand the commitments to the witness ---
    // U
    let derand_U = U.derandomize(&vk.dk, &self.data.wit_blind);
    // U_cyclefold
    let derand_U_cyclefold = self.data.r_U_cyclefold.derandomize(
      &vk.dk_cyclefold,
      &self.data.wit_blind_cyclefold,
      &self.data.err_blind_cyclefold,
    );
    Ok((derand_U, derand_U_cyclefold))
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
struct CompressedSNARKData<E>
where
  E: CurveCycleEquipped,
{
  r_U: LR1CSInstance<E>,
  l_u: SplitR1CSInstance<E>,
  z_i: Vec<E::Scalar>,
  prev_ic: (E::Scalar, E::Scalar),
  wit_blind: E::Scalar,
  r_U_cyclefold: RelaxedR1CSInstance<Dual<E>>,
  wit_blind_cyclefold: E::Base,
  err_blind_cyclefold: E::Base,
}

#[cfg(test)]
mod tests {
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, Split, SynthesisError},
    hypernova::{
      compression::CompressedSNARK,
      nebula::ic::increment_ic,
      pp::PublicParams,
      rs::{IncrementalCommitment, RecursiveSNARK, StepCircuit},
    },
    provider::{ipa_pc, Bn256EngineIPA},
    spartan::{lin_snark::LinearizedR1CSSNARK, snark::RelaxedR1CSSNARK},
    traits::{
      snark::{default_ck_hint, LinearizedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
      CurveCycleEquipped, Dual, Engine,
    },
    NovaError,
  };
  use ff::PrimeField;

  type E1 = Bn256EngineIPA;
  type E2 = Dual<E1>;
  type Fr = <E1 as Engine>::Scalar;
  type EE1 = ipa_pc::EvaluationEngine<E1>;
  type EE2 = ipa_pc::EvaluationEngine<E2>;
  type S1 = LinearizedR1CSSNARK<E1, EE1>;
  type S2 = RelaxedR1CSSNARK<E2, EE2>;

  #[test]
  fn test_pow_rs() -> Result<(), NovaError> {
    let circuit = PowCircuit {
      advice: Some((Fr::from(2u64), Fr::from(301u64))),
    };
    test_rs_with::<E1, S1, S2>(&circuit)
  }

  fn test_rs_with<E, S1, S2>(circuit: &impl StepCircuit<E::Scalar>) -> Result<(), NovaError>
  where
    E: CurveCycleEquipped,
    S1: LinearizedR1CSSNARKTrait<E>,
    S2: RelaxedR1CSSNARKTrait<Dual<E>>,
  {
    run_circuit::<E, S1, S2>(circuit, false)
  }

  fn run_circuit<E, S1, S2>(
    c: &impl StepCircuit<E::Scalar>,
    check_proof_size: bool,
  ) -> Result<(), NovaError>
  where
    E: CurveCycleEquipped,
    S1: LinearizedR1CSSNARKTrait<E>,
    S2: RelaxedR1CSSNARKTrait<Dual<E>>,
  {
    let pp = PublicParams::<E>::setup(c, &*default_ck_hint(), &*default_ck_hint());
    let z_0 = vec![
      E::Scalar::from(2u64),
      E::Scalar::from(0u64),
      E::Scalar::from(0u64),
    ];
    let mut ic = IncrementalCommitment::<E>::default();
    let mut recursive_snark = RecursiveSNARK::new(&pp, c, &z_0)?;
    for i in 0..3 {
      recursive_snark.prove_step(&pp, c, ic)?;
      let (advice_0, advice_1) = c.advice();
      ic = increment_ic::<E>(&pp.ck, &pp.ro_consts, ic, (&advice_0, &advice_1));
      recursive_snark.verify(&pp, i + 1, &z_0, ic)?;
    }
    let (pk, vk) = CompressedSNARK::<E, S1, S2>::setup(&pp)?;
    let snark = CompressedSNARK::<E, S1, S2>::prove(&pp, &pk, &recursive_snark)?;
    snark.verify(&vk, &z_0, recursive_snark.num_steps())?;
    if check_proof_size {
      let rs_str = serde_json::to_string(&recursive_snark).unwrap();
      println!("recursive snark: {} MB", rs_str.len() / 1024 / 1024);
      let snark_str = serde_json::to_string(&snark).unwrap();
      println!("compressed snark: {} KB", snark_str.len() / 1024);
      // sanity check deserialized snark
      let snark_deserialized: CompressedSNARK<E, S1, S2> =
        serde_json::from_str(&snark_str).unwrap();
      snark_deserialized.verify(&vk, &z_0, recursive_snark.num_steps())?;
    }
    Ok(())
  }

  #[derive(Clone, Default)]
  pub struct PowCircuit<F>
  where
    F: PrimeField,
  {
    advice: Option<(F, F)>,
  }

  impl<F> StepCircuit<F> for PowCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      3
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let mut x = z[0].clone();
      let mut y = x.clone();
      for i in 0..100 {
        y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
        x = y.clone();
      }

      let mut x_pc = AllocatedNum::alloc_pre_committed(
        cs.namespace(|| "x_pc"),
        || {
          self
            .advice
            .map(|(x, _)| x)
            .ok_or(SynthesisError::AssignmentMissing)
        },
        Split::ZERO,
      )?;
      let mut y_pc = x_pc.clone();
      for i in 0..5 {
        y_pc = x_pc.square(cs.namespace(|| format!("x_sq_pc{i}")))?;
        x_pc = y_pc.clone();
      }

      let mut x_pc_1 = AllocatedNum::alloc_pre_committed(
        cs.namespace(|| "x_pc_1"),
        || {
          self
            .advice
            .map(|(_, x)| x)
            .ok_or(SynthesisError::AssignmentMissing)
        },
        Split::ONE,
      )?;
      let mut y_pc_1 = x_pc_1.clone();
      for i in 0..5 {
        y_pc_1 = x_pc_1.square(cs.namespace(|| format!("x_sq_pc_1{i}")))?;
        x_pc_1 = y_pc_1.clone();
      }

      Ok(vec![y, y_pc, y_pc_1])
    }

    fn advice(&self) -> (Vec<F>, Vec<F>) {
      let advice = self.advice.expect("Advice should manually be set");
      (vec![advice.0], vec![advice.1])
    }
  }
}
