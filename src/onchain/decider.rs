//! Implements components to enable the compression-step for IVC proofs

use super::{
  decider_circuit::DeciderCircuit,
  gadgets::{DeciderNovaGadget, KZGProof},
};
use crate::{
  errors::NovaError,
  frontend::groth16::{
    self, create_random_proof, generate_random_parameters, verify_proof, Parameters,
    Proof as Groth16Proof,
  },
  nebula::{
    nifs::NIFS,
    rs::{PublicParams, RecursiveSNARK},
  },
  onchain::eth::ToEth,
  provider::{
    hyperkzg::EvaluationEngine,
    kzg_commitment::{KZGCommitmentEngine, KZGProverKey, KZGVerifierKey},
    Bn256EngineKZG,
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{evaluation::EvaluationEngineTrait, Engine, ROConstants},
  Commitment,
};
use halo2curves::bn256::{Bn256, Fr};
use rand::RngCore;

/// A type that holds the prover key for [`Decider`]
#[derive(Clone)]
pub struct DeciderProverKey {
  pub groth16_pk: Parameters<Bn256EngineKZG>,
  pub kzg_pk: KZGProverKey<Bn256>,
}

/// A type that holds the verifier key for [`Decider`]
#[derive(Debug, Clone)]
pub struct DeciderVerifierKey {
  pub groth16_vk: groth16::VerifyingKey<Bn256EngineKZG>,
  pub pp_hash: <Bn256EngineKZG as Engine>::Scalar,
  pub kzg_vk: KZGVerifierKey<Bn256>,
}

/// A SNARK that proves the knowledge of a valid  proof
#[derive(Debug)]
pub struct Decider {
  groth16_proof: Groth16Proof<Bn256EngineKZG>,
  rho: Fr,
  kzg_challenges: (Fr, Fr),
  kzg_proofs: (KZGProof<Bn256>, KZGProof<Bn256>),
}

impl Decider {
  /// Creates prover and verifier keys for [`Decider`]
  pub fn setup<R>(
    pp: &PublicParams<Bn256EngineKZG>,
    rng: &mut R,
  ) -> Result<(DeciderProverKey, DeciderVerifierKey), NovaError>
  where
    R: RngCore,
  {
    let pp_hash = pp.digest();
    let circuit = DeciderCircuit::<Bn256EngineKZG>::default(
      &pp.circuit_shape_primary.r1cs_shape,
      &pp.circuit_shape_cyclefold.r1cs_shape,
      ROConstants::<Bn256EngineKZG>::default(),
      pp_hash,
      1, // TODO: Parameterize
      (&*pp.ck_primary, &*pp.ck_cyclefold),
    );

    let (kzg_pk, kzg_vk) = EvaluationEngine::<Bn256, Bn256EngineKZG>::setup(pp.ck_primary.clone());

    // get the Groth16 specific setup for the circuit
    let params = generate_random_parameters::<Bn256EngineKZG, _, _>(circuit, rng).unwrap();

    let pk = DeciderProverKey {
      groth16_pk: params.clone(),
      kzg_pk,
    };
    let vk = DeciderVerifierKey {
      groth16_vk: params.vk,
      pp_hash,
      kzg_vk,
    };

    Ok((pk, vk))
  }

  /// Create a new [`CompressedSNARK`]
  pub fn prove<R>(
    pp: &PublicParams<Bn256EngineKZG>,
    pk: &DeciderProverKey,
    rs: &RecursiveSNARK<Bn256EngineKZG>,
    rng: &mut R,
  ) -> Result<Self, NovaError>
  where
    R: RngCore,
  {
    let circuit = DeciderCircuit::<Bn256EngineKZG>::new(pp, rs.clone())?;
    // TODO: Do we need D::Proof?
    let rho = circuit.randomness;
    let kzg_challenges = circuit.kzg_challenges.clone();

    let kzg_proofs = (
      KZGProof::prove_with_challenge(&pk.kzg_pk, kzg_challenges.0, &circuit.W_i1.W[..])?,
      KZGProof::prove_with_challenge(&pk.kzg_pk, kzg_challenges.1, &circuit.W_i1.E[..])?,
    );
    let groth16_proof = create_random_proof(circuit, &pk.groth16_pk, rng)?;
    Ok(Self {
      groth16_proof,
      rho,
      kzg_challenges,
      kzg_proofs,
    })
  }

  /// Verify the correctness of the [`CompressedSNARK`]
  pub fn verify(
    &self,
    vk: DeciderVerifierKey,
    i: Fr,
    z_0: Vec<Fr>,
    z_i: Vec<Fr>,
    U_commitments: (Commitment<Bn256EngineKZG>, Commitment<Bn256EngineKZG>),
    u_commitments: Commitment<Bn256EngineKZG>,
    // nifs_proof: NIFS<Bn256EngineKZG>,
  ) -> Result<(), NovaError> {
    let DeciderVerifierKey {
      groth16_vk,
      pp_hash,
      kzg_vk,
    } = vk;

    let prepared_groth16_vk = groth16::prepare_verifying_key(&groth16_vk);

    //   // 6.2. Fold the commitments
    //   // TODO
    let U_final_commitments = DeciderNovaGadget::fold_group_elements_native::<Bn256EngineKZG>(
      U_commitments,
      u_commitments,
      // nifs_proof.comm_T,
      self.rho,
    );

    let public_inputs = [
      &[pp_hash],
      &[i],
      &z_0[..],
      &z_i[..],
      // &U_final_commitments.inputize_nonnative(),
      &[self.kzg_challenges.0, self.kzg_challenges.1],
      &[self.kzg_proofs.0.eval, self.kzg_proofs.1.eval],
      // &proof.cmT.inputize_nonnative(),
    ]
    .concat();

    let snark_v = verify_proof(
      &prepared_groth16_vk,
      &self.groth16_proof,
      &public_inputs[..],
    )?;

    if !snark_v {
      return Err(NovaError::ProofVerifyError);
    }
    // TODO: 7.3 Verify KZG proofs
    //   for ((cm, &c), pi) in U_final_commitments
    //   .iter()
    //   .zip(&proof.kzg_challenges)
    //   .zip(&proof.kzg_proofs)
    // {
    //   // we're at the Ethereum EVM case, so the CS1 is KZG commitments
    //   CS1::verify_with_challenge(&cs_vp, c, cm, pi)?;
    // }
    Ok(())
  }
}

/// Prepares solidity calldata for calling the NovaDecider contract
#[allow(clippy::too_many_arguments)]
pub fn prepare_calldata(
  function_signature_check: [u8; 4],
  i: Fr,
  z_0: &Vec<Fr>,
  z_i: &Vec<Fr>,
  running_instance: &RelaxedR1CSInstance<Bn256EngineKZG>,
  incoming_instance: &R1CSInstance<Bn256EngineKZG>,
  proof: &Decider,
) -> Result<Vec<u8>, NovaError> {
  Ok(
    [
      function_signature_check.to_eth(),
      i.to_eth(),   // i
      z_0.to_eth(), // z_0
      z_i.to_eth(), // z_i
      running_instance.comm_W.to_eth(),
      running_instance.comm_E.to_eth(),
      incoming_instance.comm_W.to_eth(),
      // proof.cmT.to_eth(),                 // cmT
      proof.rho.to_eth(),              // r
      proof.groth16_proof.to_eth(),    // pA, pB, pC
      proof.kzg_challenges.0.to_eth(), // challenge_W, challenge_E
      proof.kzg_challenges.1.to_eth(), // challenge_W, challenge_E
                                       // proof.kzg_proofs[0].eval.to_eth(),  // eval W
                                       // proof.kzg_proofs[1].eval.to_eth(),  // eval E
                                       // proof.kzg_proofs[0].proof.to_eth(), // W kzg_proof
                                       // proof.kzg_proofs[1].proof.to_eth(), // E kzg_proof
    ]
    .concat(),
  )
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use super::*;
  use crate::{
    constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
    cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance,
    frontend::{
      num::AllocatedNum, r1cs::NovaShape, shape_cs::ShapeCS, test_cs::TestConstraintSystem,
      Circuit, ConstraintSystem, SynthesisError,
    },
    nebula::rs::StepCircuit,
    onchain::gadgets::{EvalGadget, KZGChallengesGadget},
    provider::{Bn256EngineKZG, GrumpkinEngine},
    traits::{snark::RelaxedR1CSSNARKTrait, Dual},
  };
  use ff::Field;
  use halo2curves::bn256::{Bn256, Fr};
  use rand::thread_rng;

  type E1 = Bn256EngineKZG;
  type E2 = GrumpkinEngine;
  type EE1 = crate::provider::hyperkzg::EvaluationEngine<Bn256, E1>;
  type EE2 = crate::provider::ipa_pc::EvaluationEngine<E2>;
  type S1 = crate::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
  type S2 = crate::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK
  /// Test circuit to be folded
  #[derive(Clone, Debug)]
  pub struct TestChallengeCircuit {
    pub relaxed_instance: RelaxedR1CSInstance<Bn256EngineKZG>,
    pub challenge_w: Fr,
    pub challenge_e: Fr,
  }

  impl TestChallengeCircuit {
    fn new(
      relaxed_instance: RelaxedR1CSInstance<Bn256EngineKZG>,
      challenge_w: Fr,
      challenge_e: Fr,
    ) -> Self {
      Self {
        relaxed_instance,
        challenge_w,
        challenge_e,
      }
    }

    fn default() -> Self {
      Self {
        relaxed_instance: RelaxedR1CSInstance {
          comm_W: Commitment::<Bn256EngineKZG>::default(),
          comm_E: Commitment::<Bn256EngineKZG>::default(),
          X: vec![Fr::from(0); 2],
          u: Fr::from(0),
        },
        challenge_w: Fr::random(&mut thread_rng()),
        challenge_e: Fr::random(&mut thread_rng()),
      }
    }
  }
  impl Circuit<halo2curves::bn256::Fr> for TestChallengeCircuit {
    fn synthesize<CS: ConstraintSystem<halo2curves::bn256::Fr>>(
      self,
      cs: &mut CS,
    ) -> Result<(), SynthesisError> {
      let kzg_alloc_rw = AllocatedNum::alloc(cs.namespace(|| "get kzg_challenges rw"), || {
        Ok(self.challenge_w)
      })?;
      kzg_alloc_rw.inputize(cs.namespace(|| "kzg challenge W"))?;

      let kzg_alloc_re = AllocatedNum::alloc(cs.namespace(|| "get kzg_challenges re"), || {
        Ok(self.challenge_e)
      })?;
      kzg_alloc_re.inputize(cs.namespace(|| "kzg challenge E"))?;

      let alloc_relaxed_instance: AllocatedEmulRelaxedR1CSInstance<Dual<Bn256EngineKZG>> =
        AllocatedEmulRelaxedR1CSInstance::alloc(
          cs.namespace(|| "relaxed instance"),
          Some(&self.relaxed_instance),
          BN_LIMB_WIDTH,
          BN_N_LIMBS,
        )?;

      let (alloc_rw, alloc_re) = KZGChallengesGadget::get_challenges_gadget::<CS, Bn256EngineKZG>(
        cs,
        alloc_relaxed_instance,
      )?;

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
      Ok(())
    }
  }

  struct TrivialCircuit {
    a: Option<Fr>,
    b: Option<Fr>,
    c: Option<Fr>,
  }

  impl Circuit<Fr> for TrivialCircuit {
    fn synthesize<CS: ConstraintSystem<Fr>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
      // Allocate the variables for a, b, and c
      let a = cs.alloc(|| "a", || self.a.ok_or(SynthesisError::AssignmentMissing))?;
      let b = cs.alloc(|| "b", || self.b.ok_or(SynthesisError::AssignmentMissing))?;
      let c = cs.alloc(|| "c", || self.c.ok_or(SynthesisError::AssignmentMissing))?;

      // Enforce the constraint a * b = c
      cs.enforce(|| "a * b = c", |lc| lc + a, |lc| lc + b, |lc| lc + c);

      Ok(())
    }
  }

  #[test]
  fn test_kzg_challenges_with_groth16_proof() -> Result<(), SynthesisError> {
    let circuit = TestChallengeCircuit::default();
    let mut shape_cs = ShapeCS::new();
    let _ = circuit.synthesize(&mut shape_cs);
    let (r1cs_shape, ck) = shape_cs.r1cs_shape(&*S1::ck_floor());
    let relaxed_instance = RelaxedR1CSInstance::default(&ck, &r1cs_shape);

    let (rw_native, re_native) =
      KZGChallengesGadget::get_challenges_native(relaxed_instance.clone());

    let circuit = TestChallengeCircuit::new(relaxed_instance.clone(), rw_native, re_native);
    let mut rng = thread_rng();
    let params = generate_random_parameters::<Bn256EngineKZG, _, _>(circuit.clone(), &mut rng)?;
    let groth16_proof = create_random_proof(circuit, &params, &mut rng)?;
    let prepared_groth16_vk = groth16::prepare_verifying_key(&params.vk);
    let verified = verify_proof(
      &prepared_groth16_vk,
      &groth16_proof,
      &[rw_native, re_native],
    )?;
    if !verified {
      return Err(SynthesisError::MalformedProofs("".to_string()));
    }
    Ok(())
  }

  #[test]
  fn test_eval_proof_with_challenge() {
    let circuit = TestChallengeCircuit::default();
    let mut shape_cs = ShapeCS::new();
    let _ = circuit.synthesize(&mut shape_cs);
    let (r1cs_shape, ck) = shape_cs.r1cs_shape(&*S1::ck_floor());
    let (kzg_pk, _) = EvaluationEngine::<Bn256, Bn256EngineKZG>::setup(Arc::new(ck.clone()));
    let (relaxed_instance, relaxed_witness) =
      r1cs_shape.sample_random_instance_witness(&ck).unwrap();

    let (challenge_w, challenge_e) =
      KZGChallengesGadget::get_challenges_native(relaxed_instance.clone());

    let eval_w = EvalGadget::evaluate_native(relaxed_witness.clone().W, challenge_w);
    let eval_e = EvalGadget::evaluate_native(relaxed_witness.clone().E, challenge_e);

    let proof_w = KZGProof::prove_with_challenge(&kzg_pk, challenge_w, &relaxed_witness.W).unwrap();

    let proof_e = KZGProof::prove_with_challenge(&kzg_pk, challenge_e, &relaxed_witness.E).unwrap();

    assert_eq!(eval_w, proof_w.eval);
    assert_eq!(eval_e, proof_e.eval);
  }

  #[test]
  fn test_eval_gadget() {
    let circuit = TrivialCircuit {
      a: Some(Fr::from(2)),
      b: Some(Fr::from(3)),
      c: Some(Fr::from(6)),
    };
    let mut cs = TestConstraintSystem::new();
    circuit.synthesize(&mut cs).unwrap();
    let random_vec = vec![Fr::random(&mut thread_rng()); 10];
    let random_allocated_vec = random_vec
      .iter()
      .enumerate()
      .map(|(i, x)| {
        AllocatedNum::alloc(cs.namespace(|| format!("random_allocated_vec_{i}")), || {
          Ok(*x)
        })
        .unwrap()
      })
      .collect::<Vec<_>>();
    let point = Fr::random(&mut thread_rng());
    let alloc_point = AllocatedNum::alloc(cs.namespace(|| "alloc_point"), || Ok(point)).unwrap();
    let eval_native = EvalGadget::evaluate_native(random_vec, point);
    let eval_gadget =
      EvalGadget::evaluate_gadget::<_, Bn256EngineKZG>(&mut cs, random_allocated_vec, &alloc_point)
        .unwrap();
    assert!(cs.is_satisfied());
    assert_eq!(eval_native, eval_gadget.get_value().unwrap());
  }

  /// Test circuit to be folded
  #[derive(Clone, Copy, Debug)]
  pub struct CubicFCircuit {}

  impl CubicFCircuit {
    fn new() -> Self {
      Self {}
    }
  }
  impl StepCircuit<halo2curves::bn256::Fr> for CubicFCircuit {
    fn arity(&self) -> usize {
      1
    }
    fn synthesize<CS: ConstraintSystem<halo2curves::bn256::Fr>>(
      &self,
      cs: &mut CS,
      z_in: &[AllocatedNum<halo2curves::bn256::Fr>],
    ) -> Result<Vec<AllocatedNum<halo2curves::bn256::Fr>>, SynthesisError> {
      let five = AllocatedNum::alloc(cs.namespace(|| "five"), || {
        Ok(halo2curves::bn256::Fr::from(5u64))
      })?;
      let z_i = z_in[0].clone();
      let z_i_sq = z_i.mul(cs.namespace(|| "z_i_sq"), &z_i)?;
      let z_i_cube = z_i_sq.mul(cs.namespace(|| "z_i_cube"), &z_i)?;
      let result = z_i_cube.add(cs.namespace(|| "add z_i"), &z_i)?;
      let result = result.add(cs.namespace(|| "add five"), &five)?;

      Ok(vec![result])
    }
    fn non_deterministic_advice(&self) -> Vec<halo2curves::bn256::Fr> {
      vec![]
    }
  }
  #[test]
  fn test_decider_proof() {
    let num_steps = 5;
    let f_circuit = CubicFCircuit::new();
    let rs_pp = PublicParams::<E1>::setup(&f_circuit, &*S1::ck_floor(), &*S2::ck_floor());
    let z0 = vec![<Bn256EngineKZG as Engine>::Scalar::from(3u64)];
    let mut recursive_snark: RecursiveSNARK<Bn256EngineKZG> =
      RecursiveSNARK::<Bn256EngineKZG>::new(&rs_pp, &f_circuit, &z0).unwrap();
    let mut IC_i = <Bn256EngineKZG as Engine>::Scalar::ZERO;
    for _i in 0..num_steps {
      recursive_snark
        .prove_step(&rs_pp, &f_circuit, IC_i)
        .unwrap();

      IC_i = recursive_snark.increment_commitment(&rs_pp, &f_circuit);
    }

    let res = recursive_snark.verify(&rs_pp, num_steps, &z0, IC_i);
    println!("RecursiveSNARK::verify: {:?}", res.is_ok(),);
    res.unwrap();
    let decider_circuit =
      DeciderCircuit::<Bn256EngineKZG>::new(&rs_pp, recursive_snark.clone()).unwrap();
    let mut cs = TestConstraintSystem::new();
    let _ = decider_circuit.synthesize(&mut cs);
    assert!(cs.is_satisfied());
  }
}
