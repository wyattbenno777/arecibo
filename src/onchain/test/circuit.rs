use ff::Field;
use halo2curves::bn256::Fr;
use rand::thread_rng;

use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  cyclefold::gadgets::emulated::AllocatedEmulRelaxedR1CSInstance,
  frontend::{num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError},
  nebula::rs::StepCircuit,
  onchain::gadgets::KZGChallengesGadget,
  provider::{pedersen::Commitment, Bn256EngineKZG},
  r1cs::RelaxedR1CSInstance,
  traits::{commitment::CommitmentTrait, Dual, Engine},
  gadgets::emulated::AllocatedEmulPoint,
};

/// Test circuit to be folded
#[derive(Clone, Debug)]
pub struct TestChallengeCircuit {
  /// Relaxed R1CS instance
  pub relaxed_instance: RelaxedR1CSInstance<Bn256EngineKZG>,
  /// Challenge for witness commitment
  pub challenge_w: Fr,
  /// Challenge for error commitment
  pub challenge_e: Fr,
}

impl TestChallengeCircuit {
  /// Create a new test challenge circuit
  pub fn new(
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
}

impl Default for TestChallengeCircuit {
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
      |lc| {
        lc + alloc_relaxed_instance.x0.get_variable()
          + alloc_relaxed_instance.x1.get_variable()
          + alloc_relaxed_instance.u.get_variable()
      },
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

/// Trivial circuit
#[derive(Clone, Debug)]
pub struct TrivialCircuit {
  /// Variable a
  pub a: Option<Fr>,
  /// Variable b
  pub b: Option<Fr>,
  /// Variable c
  pub c: Option<Fr>,
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

/// Test circuit to be folded
#[derive(Clone, Copy, Debug)]
pub struct CubicFCircuit {}

impl CubicFCircuit {
  /// Create a new circuit
  pub fn new() -> Self {
    Self {}
  }
}

impl Default for CubicFCircuit {
  fn default() -> Self {
    Self::new()
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

#[derive(Clone, Debug)]
/// Test circuit to be folded
pub struct BigNatCircuit {
  /// Commitment of the witness
  pub w: Commitment<Bn256EngineKZG>,
}

impl Circuit<Fr> for BigNatCircuit {
  fn synthesize<CS: ConstraintSystem<Fr>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    let alloc_w: AllocatedEmulPoint<<Dual<Bn256EngineKZG> as Engine>::GE> =
      AllocatedEmulPoint::alloc(
        cs.namespace(|| "allocate comm_W"),
        Some(self.w.to_coordinates()),
        BN_LIMB_WIDTH,
        BN_N_LIMBS,
      )?;

    let (U_i_cmW_x, U_i_cmW_y, U_i_cmW_id) = alloc_w.to_coordinates();
    println!("U_i_cmW_x: {:?}", U_i_cmW_x.value);
    println!("U_i_cmW_y: {:?}", U_i_cmW_y.value);

    for (i, limb) in U_i_cmW_x.as_limbs().iter().enumerate() {
      let tmp = limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of x to num")))?;
      println!("i: {:?}", tmp.get_value());
      tmp.inputize(cs.namespace(|| format!("convert limb {i} of x to num")))?;
    }

    for (i, limb) in U_i_cmW_y.as_limbs().iter().enumerate() {
      let tmp = limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of y to num")))?;
      println!("i: {:?}", tmp.get_value());
      tmp.inputize(cs.namespace(|| format!("convert limb {i} of y to num")))?;
    }

    cs.enforce(
      || "dummy",
      |lc| lc + U_i_cmW_id.get_variable(),
      |lc| lc,
      |lc| lc,
    );

    Ok(())
  }
}
