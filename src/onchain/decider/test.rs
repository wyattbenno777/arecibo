
#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use crate::{
    constants::{BN_LIMB_WIDTH, BN_N_LIMBS}, cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance, frontend::{
      groth16::{self, create_random_proof, generate_random_parameters, verify_proof}, num::AllocatedNum, r1cs::NovaShape, shape_cs::ShapeCS, test_cs::TestConstraintSystem, Circuit, ConstraintSystem, SynthesisError
    }, nebula::rs::{PublicParams, RecursiveSNARK, StepCircuit}, onchain::{decider_circuit::DeciderCircuit, gadgets::{EvalGadget, FoldGadget, KZGChallengesGadget, KZGProof}}, provider::{hyperkzg::EvaluationEngine, kzg_commitment::UVKZGCommitment, Bn256EngineKZG, GrumpkinEngine}, r1cs::RelaxedR1CSInstance, traits::{evaluation::EvaluationEngineTrait, snark::RelaxedR1CSSNARKTrait, Dual, Engine}, Commitment
  };
  use ff::Field;
  use halo2curves::bn256::{Bn256, Fr};
  use rand::thread_rng;
  use group::Curve;
  use crate::provider::non_hiding_zeromorph::{UVKZGPCS, UVKZGPoly};

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
        alloc_relaxed_instance.clone(),
      )?;


      cs.enforce(
        || "trivial check to constrain allocated variables",
        |lc| lc + alloc_relaxed_instance.x0.get_variable() + alloc_relaxed_instance.x1.get_variable() + alloc_relaxed_instance.u.get_variable(),
        |lc| lc,
        |lc| lc,
      );

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

    let proof_w = KZGProof::prove(&kzg_pk, challenge_w, &relaxed_witness.W).unwrap();
    // let poly_w = UVKZGPoly::new(relaxed_witness.W.clone());
    // let (proof_w, p_eval_w) = UVKZGPCS::open(&kzg_pk, &poly_w, &challenge_w).unwrap();

    let proof_e = KZGProof::prove(&kzg_pk, challenge_e, &relaxed_witness.E).unwrap();

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
    // let poly = UVKZGPoly::new(random_vec.clone());
    // let ev = poly.evaluate(&point);
    let eval_native = EvalGadget::evaluate_native(random_vec, point);
    let eval_gadget =
      EvalGadget::evaluate_gadget::<_, Bn256EngineKZG>(&mut cs, random_allocated_vec, &alloc_point)
        .unwrap();
    assert!(cs.is_satisfied());
    // assert_eq!(ev, eval_native);
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
  fn test_decider_constraints() {
    let num_steps = 5;
    let f_circuit = CubicFCircuit::new();
    let rs_pp = PublicParams::<E1>::setup(&f_circuit, &*S1::ck_floor(), &*S2::ck_floor());
    let z0 = vec![<Bn256EngineKZG as Engine>::Scalar::from(3u64)];
    let mut rs: RecursiveSNARK<Bn256EngineKZG> =
      RecursiveSNARK::<Bn256EngineKZG>::new(&rs_pp, &f_circuit, &z0).unwrap();
    let mut IC_i = <Bn256EngineKZG as Engine>::Scalar::ZERO;
    for _i in 0..num_steps {
      rs
        .prove_step(&rs_pp, &f_circuit, IC_i)
        .unwrap();

      IC_i = rs.increment_commitment(&rs_pp, &f_circuit);
    }

    let res = rs.verify(&rs_pp, num_steps, &z0, IC_i);
    res.unwrap();
    let decider_circuit =
      DeciderCircuit::<Bn256EngineKZG>::new(&rs_pp, rs.clone()).unwrap();
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
    rs
      .prove_step(&rs_pp, &f_circuit, IC_i)
      .unwrap();

    IC_i = rs.increment_commitment(&rs_pp, &f_circuit);
  }

  let (kzg_pk, kzg_vk) = EvaluationEngine::<Bn256, Bn256EngineKZG>::setup(rs_pp.ck_primary.clone());

  let circuit = DeciderCircuit::<Bn256EngineKZG>::new(&rs_pp, rs.clone()).unwrap();
  let rho = circuit.randomness;
  let nifs_proof = circuit.nifs_proof.clone();
  let (kzg_challenges_w, kzg_challenges_e) = circuit.kzg_challenges.clone();

  let (kzg_proof_w, kzg_proof_e) = (
    KZGProof::prove(&kzg_pk, kzg_challenges_w, &circuit.W_i1.W[..]).unwrap(),
    KZGProof::prove(&kzg_pk, kzg_challenges_e, &circuit.W_i1.E[..]).unwrap(),
  );
  let (U_cmW, U_cmE) = FoldGadget::fold_group_elements_native::<Bn256EngineKZG>(
    (rs.r_U_primary.comm_W, rs.r_U_primary.comm_E),
    rs.l_u_primary.comm_W,
    nifs_proof.nifs_primary.comm_T,
    rho,
  ).unwrap();

  let kzg_U_cmW = UVKZGCommitment::<Bn256>::new(U_cmW.comm.to_affine());
  let kzg_U_cmE = UVKZGCommitment::<Bn256>::new(U_cmE.comm.to_affine());

  // 7.3 Verify KZG proofs
  kzg_proof_w.verify(&kzg_vk, &kzg_U_cmW, kzg_challenges_w).unwrap();
  kzg_proof_e.verify(&kzg_vk, &kzg_U_cmE, kzg_challenges_e).unwrap();
}
}