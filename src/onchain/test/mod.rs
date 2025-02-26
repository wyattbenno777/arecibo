//! Test module.

/// Full flow tests.
pub mod full_flow;
/// Circuit structs.
pub mod circuit;

/// Tests for the decider circuit.
#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use crate::{
    constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
    frontend::{
      groth16::{self, create_random_proof, generate_random_parameters, verify_proof},
      num::AllocatedNum,
      r1cs::NovaShape,
      shape_cs::ShapeCS,
      test_cs::TestConstraintSystem,
      Circuit, ConstraintSystem, SynthesisError,
    },
    gadgets::nat_to_limbs,
    nebula::rs::{PublicParams, RecursiveSNARK, StepCircuit},
    onchain::{
      decider_circuit::DeciderCircuit,
      gadgets::{EvalGadget, FoldGadget, KZGChallengesGadget, KZGProof}, test::circuit::{BigNatCircuit, CubicFCircuit, TrivialCircuit},
    },
    provider::{
      hyperkzg::EvaluationEngine,
      kzg_commitment::{KZGCommitmentEngine, UVKZGCommitment},
      Bn256EngineKZG, GrumpkinEngine,
    },
    r1cs::{commitment_key, RelaxedR1CSInstance},
    traits::{commitment::{CommitmentEngineTrait, CommitmentTrait}, evaluation::EvaluationEngineTrait, snark::{default_ck_hint, RelaxedR1CSSNARKTrait}, Engine
    },
  };
  use ff::{Field, PrimeField};
  use group::Curve;
  use halo2curves::bn256::{Bn256, Fr};
  use num_bigint::{BigInt, Sign};
  use rand::thread_rng;

use super::circuit::TestChallengeCircuit;

  type E1 = Bn256EngineKZG;
  type E2 = GrumpkinEngine;
  type EE1 = crate::provider::hyperkzg::EvaluationEngine<Bn256, E1>;
  type EE2 = crate::provider::ipa_pc::EvaluationEngine<E2>;
  type S1 = crate::spartan::snark::RelaxedR1CSSNARK<E1, EE1>;
  type S2 = crate::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;

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
  fn test_eval_gadget() {
    let circuit = TrivialCircuit{
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

  #[test]
  fn test_decider_constraints() {
    let num_steps = 5;
    let f_circuit = CubicFCircuit::new();
    let rs_pp = PublicParams::<E1>::setup(&f_circuit, &*S1::ck_floor(), &*S2::ck_floor());
    let z0 = vec![<Bn256EngineKZG as Engine>::Scalar::from(3u64)];
    let mut rs: RecursiveSNARK<Bn256EngineKZG> =
      RecursiveSNARK::<Bn256EngineKZG>::new(&rs_pp, &f_circuit, &z0).unwrap();
    let mut IC_i = <Bn256EngineKZG as Engine>::Scalar::ZERO;
    for _i in 0..num_steps {
      rs.prove_step(&rs_pp, &f_circuit, IC_i).unwrap();

      IC_i = rs.increment_commitment(&rs_pp, &f_circuit);
    }

    let res = rs.verify(&rs_pp, num_steps, &z0, IC_i);
    res.unwrap();
    let decider_circuit = DeciderCircuit::<Bn256EngineKZG>::new(&rs_pp, rs.clone()).unwrap();
    let mut cs = TestConstraintSystem::new();
    let _ = decider_circuit.synthesize(&mut cs);
    assert!(cs.is_satisfied());
  }

  #[test]
  fn test_fold_gadget() {
    let num_steps = 5;
    let f_circuit = CubicFCircuit::new();
    let rs_pp = PublicParams::<E1>::setup(&f_circuit, &*S1::ck_floor(), &*S2::ck_floor());
    let z0 = vec![<Bn256EngineKZG as Engine>::Scalar::from(3u64)];
    let mut rs: RecursiveSNARK<Bn256EngineKZG> =
      RecursiveSNARK::<Bn256EngineKZG>::new(&rs_pp, &f_circuit, &z0).unwrap();
    let mut IC_i = <Bn256EngineKZG as Engine>::Scalar::ZERO;
    for _i in 0..num_steps {
      rs.prove_step(&rs_pp, &f_circuit, IC_i).unwrap();

      IC_i = rs.increment_commitment(&rs_pp, &f_circuit);
    }

    let (kzg_pk, kzg_vk) =
      EvaluationEngine::<Bn256, Bn256EngineKZG>::setup(rs_pp.ck_primary.clone());

    let circuit = DeciderCircuit::<Bn256EngineKZG>::new(&rs_pp, rs.clone()).unwrap();
    let rho = circuit.randomness;
    let nifs_proof = circuit.nifs_proof.clone();

    let (U_i1_cmW, U_i1_cmE) = FoldGadget::fold_group_elements_native::<Bn256EngineKZG>(
      rs.r_U_primary.comm_W,
      rs.r_U_primary.comm_E,
      rs.l_u_primary.comm_W,
      nifs_proof.nifs_primary.comm_T,
      rho,
    )
    .unwrap();

    assert_eq!(circuit.U_i1.comm_W, U_i1_cmW);
    assert_eq!(circuit.U_i1.comm_E, U_i1_cmE);

    let (kzg_challenges_w, kzg_challenges_e) = circuit.kzg_challenges.clone();

    let (kzg_proof_w, kzg_proof_e) = (
      KZGProof::prove(&kzg_pk, kzg_challenges_w, &circuit.W_i1.W[..]).unwrap(),
      KZGProof::prove(&kzg_pk, kzg_challenges_e, &circuit.W_i1.E[..]).unwrap(),
    );

    let kzg_U_i1_cmW = UVKZGCommitment::<Bn256>::new(U_i1_cmW.comm.to_affine());
    let kzg_U_i1_cmE = UVKZGCommitment::<Bn256>::new(U_i1_cmE.comm.to_affine());

    // 7.3 Verify KZG proofs
    kzg_proof_w
      .verify(&kzg_vk, &kzg_U_i1_cmW, kzg_challenges_w)
      .unwrap();
    kzg_proof_e
      .verify(&kzg_vk, &kzg_U_i1_cmE, kzg_challenges_e)
      .unwrap();
  }

  #[test]
  fn test_kzg_pairing() {
    let W = vec![Fr::random(&mut thread_rng()); 10];
    let num_vars = 30;
    let S = crate::r1cs::tests::tiny_r1cs::<Bn256EngineKZG>(num_vars);
    let r = Fr::random(&mut thread_rng());
    let challenge = Fr::random(&mut thread_rng());

    // generate generators and ro constants
    let ck = commitment_key(&S, &*default_ck_hint());
    let (kzg_pk, kzg_vk) = EvaluationEngine::<Bn256, Bn256EngineKZG>::setup(Arc::new(ck.clone()));
    let comm_W =
      <KZGCommitmentEngine<Bn256> as CommitmentEngineTrait<Bn256EngineKZG>>::commit(&ck, &W, &r);
    let proof = KZGProof::prove(&kzg_pk, challenge, &W[..]).unwrap();
    proof
      .verify(
        &kzg_vk,
        &UVKZGCommitment::<Bn256>::new(comm_W.comm.to_affine()),
        challenge,
      )
      .unwrap();
  }

  #[test]
  fn test_commitment_input() -> Result<(), SynthesisError> {
    let num_steps = 5;
    let f_circuit = CubicFCircuit::new();
    let rs_pp = PublicParams::<E1>::setup(&f_circuit, &*S1::ck_floor(), &*S2::ck_floor());
    let z0 = vec![<Bn256EngineKZG as Engine>::Scalar::from(3u64)];
    let mut rs: RecursiveSNARK<Bn256EngineKZG> =
      RecursiveSNARK::<Bn256EngineKZG>::new(&rs_pp, &f_circuit, &z0).unwrap();
    let mut IC_i = <Bn256EngineKZG as Engine>::Scalar::ZERO;
    for _i in 0..num_steps {
      rs.prove_step(&rs_pp, &f_circuit, IC_i).unwrap();
      IC_i = rs.increment_commitment(&rs_pp, &f_circuit);
    }

    let w = rs.r_U_primary.comm_W;
    let circuit = BigNatCircuit { w: w.clone() };
    let mut rng = thread_rng();
    let params = generate_random_parameters::<Bn256EngineKZG, _, _>(circuit.clone(), &mut rng)?;
    let groth16_proof = create_random_proof(circuit, &params, &mut rng)?;
    let prepared_groth16_vk = groth16::prepare_verifying_key(&params.vk);

    let (w_x, w_y) = {
      let (x, y, _id) = w.to_coordinates();
      println!("x: {:?}", x);
      println!("y: {:?}", y);
      let x_bignat = BigInt::from_bytes_le(Sign::Plus, &x.to_repr());
      let x_limbs = nat_to_limbs(&x_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
      let y_bignat = BigInt::from_bytes_le(Sign::Plus, &y.to_repr());
      let y_limbs = nat_to_limbs(&y_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
      (x_limbs, y_limbs)
    };
    println!("w_x: {:?}", w_x);
    println!("w_y: {:?}", w_y);

    let public_inputs = [&w_x[..], &w_y[..]].concat();
    let verified = verify_proof(&prepared_groth16_vk, &groth16_proof, &public_inputs)?;
    if !verified {
      return Err(SynthesisError::MalformedProofs("".to_string()));
    }
    Ok(())
  }
}
