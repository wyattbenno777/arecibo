use self::traits::CurveCycleEquipped;

use super::*;
use crate::{
  provider::{
    non_hiding_zeromorph::ZMPCS, Bn256EngineZKPedersen, Bn256EngineKZG, Bn256EngineZM, PallasEngine,
    Secp256k1Engine,
  },
  traits::{zkevaluation::EvaluationEngineTrait, snark::default_ck_hint},
};
use ::bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use provider::zk_ipa_pc::EvaluationEngine;
use core::{fmt::Write, marker::PhantomData};
use expect_test::{expect, Expect};
use ff::PrimeField;
use halo2curves::bn256::Bn256;
use traits::{circuit::TrivialCircuit, zksnark::RelaxedR1CSSNARKTrait};
use crate::traits::zkevaluation;
use crate::provider::zk_pedersen;
use crate::traits::commitment::ZKCommitmentEngineTrait;
use crate::Engine;
use crate::provider::traits::DlogGroup;

type EE<E> = provider::zk_ipa_pc::EvaluationEngine<E>;
type S<E, EE> = spartan::zksnark::RelaxedR1CSSNARK<E, EE>;
type SPrime<E, EE> = spartan::zkppsnark::RelaxedR1CSSNARK<E, EE>;

#[derive(Clone, Debug)]
struct CubicCircuit<F: PrimeField> {
  _p: PhantomData<F>,
  counter_type: StepCounterType,
}

impl<F> Default for CubicCircuit<F>
where
  F: PrimeField,
{
  /// Creates a new trivial test circuit with step counter type Incremental
  fn default() -> CubicCircuit<F> {
    Self {
      _p: PhantomData::default(),
      counter_type: StepCounterType::Incremental,
    }
  }
}

impl<F: PrimeField> StepCircuit<F> for CubicCircuit<F> {
  fn arity(&self) -> usize {
    1
  }

  fn get_counter_type(&self) -> StepCounterType {
    self.counter_type
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = &z[0];
    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
    })?;

    cs.enforce(
      || "y = x^3 + x + 5",
      |lc| {
        lc + x_cu.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    Ok(vec![y])
  }
}

impl<F: PrimeField> CubicCircuit<F> {
  fn output(&self, z: &[F]) -> Vec<F> {
    vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
  }
}

fn test_pp_digest_with<E1, T1, T2, EE1, EE2>(circuit1: &T1, circuit2: &T2, expected: &Expect)
where
  E1: CurveCycleEquipped + Serialize + for<'de> Deserialize<'de>,
  T1: StepCircuit<E1::Scalar>,
  T2: StepCircuit<<Dual<E1> as Engine>::Scalar>,
  EE1: EvaluationEngineTrait<E1> + zkevaluation::EvaluationEngineTrait<E1>,
  EE2: EvaluationEngineTrait<Dual<E1>> + zkevaluation::EvaluationEngineTrait<Dual<E1>>,
  // this is due to the reliance on Abomonation
  <E1::Scalar as PrimeField>::Repr: Abomonation,
  <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
  <E1 as Engine>::CE: CommitmentEngineTrait<E1, Commitment = zk_pedersen::Commitment<E1>, CommitmentKey = zk_pedersen::CommitmentKey<E1>>,
  <Dual<E1> as Engine>::CE: CommitmentEngineTrait<Dual<E1>, Commitment = zk_pedersen::Commitment<Dual<E1>>, CommitmentKey = zk_pedersen::CommitmentKey<Dual<E1>>>,
  <E1 as Engine>::GE: DlogGroup<ScalarExt = <E1 as Engine>::Scalar>,
  <Dual<E1> as Engine>::GE: DlogGroup<ScalarExt = <Dual<E1> as Engine>::Scalar>,
  <E1 as Engine>::CE: ZKCommitmentEngineTrait<E1>,
  <Dual<E1> as Engine>::CE: ZKCommitmentEngineTrait<Dual<E1>>,
  <E1 as traits::CurveCycleEquipped>::Secondary: Serialize + for<'de> Deserialize<'de>,
{
  // this tests public parameters with a size specifically intended for a spark-compressed SNARK
  let ck_hint1 = &*SPrime::<E1, EE1>::ck_floor();
  let ck_hint2 = &*SPrime::<Dual<E1>, EE2>::ck_floor();
  let pp = PublicParams::<E1>::setup(circuit1, circuit2, ck_hint1, ck_hint2).unwrap();

  let digest_str = pp
    .digest()
    .to_repr()
    .as_ref()
    .iter()
    .fold(String::new(), |mut output, b| {
      let _ = write!(output, "{b:02x}");
      output
    });

  expected.assert_eq(&digest_str);
}

#[test]
fn test_pp_digest() {
//   test_pp_digest_with::<PallasEngine, _, _, EE<_>, EE<_>>(
//     &TrivialCircuit::default(),
//     &TrivialCircuit::default(),
//     &expect!["cbbc103130b77249bfb14b86f5e9800f29704ef06fd38ff0964dcc385ac62d00"],
//   );

//   test_pp_digest_with::<PallasEngine, _, _, EE<_>, EE<_>>(
//     &CubicCircuit::default(),
//     &TrivialCircuit::default(),
//     &expect!["f1f9ba473c14dbf6ddb1b3dddb1b3e97547020f24879baaf5bf6ef02d2de1001"],
//   );

  test_pp_digest_with::<Bn256EngineZKPedersen, _, _, EE<_>, EE<_>>(
    &TrivialCircuit::default(),
    &TrivialCircuit::default(),
    &expect!["a94b9c6e58773f7f1c06372a1b9561f7ec6d5b5c985c2d9965fc0be24eab4e03"],
  );

  test_pp_digest_with::<Bn256EngineZKPedersen, _, _, EE<_>, EE<_>>(
    &CubicCircuit::default(),
    &TrivialCircuit::default(),
    &expect!["0e984618bb3661bbfd9f2e5f013255dd26d15c5311225d1b1aef26640819be02"],
  );

//   test_pp_digest_with::<Secp256k1Engine, _, _, EE<_>, EE<_>>(
//     &TrivialCircuit::default(),
//     &TrivialCircuit::default(),
//     &expect!["8a5dc9867538ade2691aa6e23fed408d399cde10e4cd9c92359f7adb81113b02"],
//   );

//   test_pp_digest_with::<Secp256k1Engine, _, _, EE<_>, EE<_>>(
//     &CubicCircuit::default(),
//     &TrivialCircuit::default(),
//     &expect!["ffff2f5d3a9b5077ad861498b4b042029d8fe3a2fda36a38ad6e13ee4b41a703"],
//   );
}

fn test_ivc_trivial_with<E1>()
where
  E1: CurveCycleEquipped,
{
  let test_circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
  let test_circuit2 = TrivialCircuit::<<Dual<E1> as Engine>::Scalar>::default();

  // produce public parameters
  let pp = PublicParams::<E1>::setup(
    &test_circuit1,
    &test_circuit2,
    &*default_ck_hint(),
    &*default_ck_hint(),
  )
  .unwrap();
  let num_steps = 1;

  // produce a recursive SNARK
  let mut recursive_snark = RecursiveSNARK::new(
    &pp,
    &test_circuit1,
    &test_circuit2,
    &[<E1 as Engine>::Scalar::ZERO],
    &[<Dual<E1> as Engine>::Scalar::ZERO],
  )
  .unwrap();

  recursive_snark
    .prove_step(&pp, &test_circuit1, &test_circuit2)
    .unwrap();

  // verify the recursive SNARK
  recursive_snark
    .verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )
    .unwrap();
}

#[test]
fn test_ivc_trivial() {
//   test_ivc_trivial_with::<PallasEngine>();
  test_ivc_trivial_with::<Bn256EngineZKPedersen>();
//   test_ivc_trivial_with::<Secp256k1Engine>();
}

fn test_ivc_nontrivial_with<E1>()
where
  E1: CurveCycleEquipped,
{
  let circuit_primary = TrivialCircuit::default();
  let circuit_secondary = CubicCircuit::default();

  // produce public parameters
  let pp = PublicParams::<E1>::setup(
    &circuit_primary,
    &circuit_secondary,
    &*default_ck_hint(),
    &*default_ck_hint(),
  )
  .unwrap();

  let num_steps = 3;

  // produce a recursive SNARK
  let mut recursive_snark = RecursiveSNARK::<E1>::new(
    &pp,
    &circuit_primary,
    &circuit_secondary,
    &[<E1 as Engine>::Scalar::ONE],
    &[<Dual<E1> as Engine>::Scalar::ZERO],
  )
  .unwrap();

  for i in 0..num_steps {
    recursive_snark
      .prove_step(&pp, &circuit_primary, &circuit_secondary)
      .unwrap();

    // verify the recursive snark at each step of recursion
    recursive_snark
      .verify(
        &pp,
        i + 1,
        &[<E1 as Engine>::Scalar::ONE],
        &[<Dual<E1> as Engine>::Scalar::ZERO],
      )
      .unwrap();
  }

  // verify the recursive SNARK
  let (zn_primary, zn_secondary) = recursive_snark
    .verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )
    .unwrap();

  // sanity: check the claimed output with a direct computation of the same
  assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
  let mut zn_secondary_direct = vec![<Dual<E1> as Engine>::Scalar::ZERO];
  for _i in 0..num_steps {
    zn_secondary_direct = circuit_secondary.clone().output(&zn_secondary_direct);
  }
  assert_eq!(zn_secondary, zn_secondary_direct);
  assert_eq!(
    zn_secondary,
    vec![<Dual<E1> as Engine>::Scalar::from(2460515u64)]
  );
}

#[test]
fn test_ivc_nontrivial() {
//   test_ivc_nontrivial_with::<PallasEngine>();
  test_ivc_nontrivial_with::<Bn256EngineZKPedersen>();
//   test_ivc_nontrivial_with::<Secp256k1Engine>();
}

fn test_ivc_nontrivial_with_some_compression_with<E1, S1, S2>()
where
  E1: CurveCycleEquipped,
  // this is due to the reliance on Abomonation
  <E1::Scalar as PrimeField>::Repr: Abomonation,
  <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<Dual<E1>>,
{
  let circuit_primary = TrivialCircuit::default();
  let circuit_secondary = CubicCircuit::default();

  // produce public parameters, which we'll maybe use with a preprocessing compressed SNARK
  let pp = PublicParams::<E1>::setup(
    &circuit_primary,
    &circuit_secondary,
    &*S1::ck_floor(),
    &*S2::ck_floor(),
  )
  .unwrap();

  let num_steps = 3;

  // produce a recursive SNARK
  let mut recursive_snark = RecursiveSNARK::<E1>::new(
    &pp,
    &circuit_primary,
    &circuit_secondary,
    &[<E1 as Engine>::Scalar::ONE],
    &[<Dual<E1> as Engine>::Scalar::ZERO],
  )
  .unwrap();

  for _i in 0..num_steps {
    recursive_snark
      .prove_step(&pp, &circuit_primary, &circuit_secondary)
      .unwrap();
  }

  // verify the recursive SNARK
  let (zn_primary, zn_secondary) = recursive_snark
    .verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )
    .unwrap();

  // sanity: check the claimed output with a direct computation of the same
  assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
  let mut zn_secondary_direct = vec![<Dual<E1> as Engine>::Scalar::ZERO];
  for _i in 0..num_steps {
    zn_secondary_direct = circuit_secondary.clone().output(&zn_secondary_direct);
  }
  assert_eq!(zn_secondary, zn_secondary_direct);
  assert_eq!(
    zn_secondary,
    vec![<Dual<E1> as Engine>::Scalar::from(2460515u64)]
  );

  // // run the compressed snark
  // // produce the prover and verifier keys for compressed snark
  // let (pk, vk) = CompressedSNARK::<_, S1, S2>::setup(&pp).unwrap();

  // // produce a compressed SNARK
  // let compressed_snark = CompressedSNARK::<_, S1, S2>::prove(&pp, &pk, &recursive_snark).unwrap();

  // // verify the compressed SNARK
  // compressed_snark
  //   .verify(
  //     &vk,
  //     num_steps,
  //     &[<E1 as Engine>::Scalar::ONE],
  //     &[<Dual<E1> as Engine>::Scalar::ZERO],
  //   )
  //   .unwrap();
}

// fn test_ivc_nontrivial_with_compression_with<E1, EE1, EE2>()
// where
//   E1: CurveCycleEquipped,
//   EE1: EvaluationEngineTrait<Bn256EngineZKPedersen>,
//   EE2: EvaluationEngineTrait<Dual<Bn256EngineZKPedersen>>,
//   // this is due to the reliance on Abomonation
//   <E1::Scalar as PrimeField>::Repr: Abomonation,
//   <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
// {
//   test_ivc_nontrivial_with_some_compression_with::<E1, S<EE1>, S<EE2>>()
// }

// #[test]
// fn test_ivc_nontrivial_with_compression() {
// //   test_ivc_nontrivial_with_compression_with::<PallasEngine, EE<_>, EE<_>>();
//   test_ivc_nontrivial_with_compression_with::<Bn256EngineZKPedersen, EE<_>, EE<_>>();
// //   test_ivc_nontrivial_with_compression_with::<Secp256k1Engine, EE<_>, EE<_>>();
// //   test_ivc_nontrivial_with_compression_with::<Bn256EngineZM, ZMPCS<Bn256, _>, EE<_>>();
// //   test_ivc_nontrivial_with_compression_with::<
// //     Bn256EngineKZG,
// //     provider::hyperkzg::EvaluationEngine<Bn256, _>,
// //     EE<_>,
// //   >();
// }

// fn test_ivc_nontrivial_with_spark_compression_with<E1, EE1, EE2>()
// where
//   E1: CurveCycleEquipped,
//   EE1: EvaluationEngineTrait<E1> + traits::zkevaluation::EvaluationEngineTrait<provider::Bn256EngineZKPedersen>,
//   EE2: EvaluationEngineTrait<Dual<E1>> + traits::zkevaluation::EvaluationEngineTrait<provider::ZKGrumpkinEngine>,
//   // this is due to the reliance on Abomonation
//   <E1::Scalar as PrimeField>::Repr: Abomonation,
//   <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
// {
//   test_ivc_nontrivial_with_some_compression_with::<E1, SPrime<EE1>, SPrime<EE2>>()
// }

// #[test]
// fn test_ivc_nontrivial_with_spark_compression() {
// //   test_ivc_nontrivial_with_spark_compression_with::<PallasEngine, EE<_>, EE<_>>();
//   test_ivc_nontrivial_with_spark_compression_with::<Bn256EngineZKPedersen, EE<_>, EE<_>>();
// //   test_ivc_nontrivial_with_spark_compression_with::<Secp256k1Engine, EE<_>, EE<_>>();
// //   test_ivc_nontrivial_with_spark_compression_with::<Bn256EngineZM, ZMPCS<Bn256, _>, EE<_>>();
// //   test_ivc_nontrivial_with_spark_compression_with::<
// //     Bn256EngineKZG,
// //     provider::hyperkzg::EvaluationEngine<Bn256, _>,
// //     EE<_>,
// //   >();
// }

fn test_ivc_nondet_with_compression_with<E1, EE1, EE2>()
where
  E1: CurveCycleEquipped,
  EE1: EvaluationEngineTrait<E1>,
  EE2: EvaluationEngineTrait<Dual<E1>>,
  // this is due to the reliance on Abomonation
  <E1::Scalar as PrimeField>::Repr: Abomonation,
  <<Dual<E1> as Engine>::Scalar as PrimeField>::Repr: Abomonation,
{
  // y is a non-deterministic advice representing the fifth root of the input at a step.
  #[derive(Clone, Debug)]
  struct FifthRootCheckingCircuit<F> {
    y: F,
  }

  impl<F: PrimeField> FifthRootCheckingCircuit<F> {
    fn new(num_steps: usize) -> (Vec<F>, Vec<Self>) {
      let mut powers = Vec::new();
      let rng = &mut rand::rngs::OsRng;
      let mut seed = F::random(rng);
      for _i in 0..num_steps + 1 {
        seed *= seed.clone().square().square();

        powers.push(Self { y: seed });
      }

      // reverse the powers to get roots
      let roots = powers.into_iter().rev().collect::<Vec<Self>>();
      (vec![roots[0].y], roots[1..].to_vec())
    }
  }

  impl<F> StepCircuit<F> for FifthRootCheckingCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    /// Returns the type of the counter for this circuit
    fn get_counter_type(&self) -> StepCounterType {
      StepCounterType::Incremental
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let x = &z[0];

      // we allocate a variable and set it to the provided non-deterministic advice.
      let y = AllocatedNum::alloc_infallible(cs.namespace(|| "y"), || self.y);

      // We now check if y = x^{1/5} by checking if y^5 = x
      let y_sq = y.square(cs.namespace(|| "y_sq"))?;
      let y_quad = y_sq.square(cs.namespace(|| "y_quad"))?;
      let y_pow_5 = y_quad.mul(cs.namespace(|| "y_fifth"), &y)?;

      cs.enforce(
        || "y^5 = x",
        |lc| lc + y_pow_5.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + x.get_variable(),
      );

      Ok(vec![y])
    }
  }

  let circuit_primary = FifthRootCheckingCircuit {
    y: <E1 as Engine>::Scalar::ZERO,
  };

  let circuit_secondary = TrivialCircuit::default();

  // produce public parameters
  let pp = PublicParams::<E1>::setup(
    &circuit_primary,
    &circuit_secondary,
    &*default_ck_hint(),
    &*default_ck_hint(),
  )
  .unwrap();

  let num_steps = 3;

  // produce non-deterministic advice
  let (z0_primary, roots) = FifthRootCheckingCircuit::new(num_steps);
  let z0_secondary = vec![<Dual<E1> as Engine>::Scalar::ZERO];

  // produce a recursive SNARK
  let mut recursive_snark = RecursiveSNARK::<E1>::new(
    &pp,
    &roots[0],
    &circuit_secondary,
    &z0_primary,
    &z0_secondary,
  )
  .unwrap();

  for circuit_primary in roots.iter().take(num_steps) {
    recursive_snark
      .prove_step(&pp, circuit_primary, &circuit_secondary)
      .unwrap();
  }

  // verify the recursive SNARK
  recursive_snark
    .verify(&pp, num_steps, &z0_primary, &z0_secondary)
    .unwrap();

//   // produce the prover and verifier keys for compressed snark
//   let (pk, vk) = CompressedSNARK::<_, S<E1, EE1>, S<_, EE2>>::setup(&pp).unwrap();

//   // produce a compressed SNARK
//   let compressed_snark =
//     CompressedSNARK::<_, S<E1, EE1>, S<_, EE2>>::prove(&pp, &pk, &recursive_snark).unwrap();

//   // verify the compressed SNARK
//   compressed_snark
//     .verify(&vk, num_steps, &z0_primary, &z0_secondary)
//     .unwrap();
}

#[test]
fn test_ivc_nondet_with_compression() {
//   test_ivc_nondet_with_compression_with::<PallasEngine, EE<_>, EE<_>>();
  test_ivc_nondet_with_compression_with::<Bn256EngineZKPedersen, EE<_>, EE<_>>();
//   test_ivc_nondet_with_compression_with::<Secp256k1Engine, EE<_>, EE<_>>();
//   test_ivc_nondet_with_compression_with::<Bn256EngineZM, ZMPCS<Bn256, _>, EE<_>>();
}

fn test_ivc_base_with<E1>()
where
  E1: CurveCycleEquipped,
{
  let test_circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
  let test_circuit2 = CubicCircuit::<<Dual<E1> as Engine>::Scalar>::default();

  // produce public parameters
  let pp = PublicParams::<E1>::setup(
    &test_circuit1,
    &test_circuit2,
    &*default_ck_hint(),
    &*default_ck_hint(),
  )
  .unwrap();

  let num_steps = 1;

  // produce a recursive SNARK
  let mut recursive_snark = RecursiveSNARK::<E1>::new(
    &pp,
    &test_circuit1,
    &test_circuit2,
    &[<E1 as Engine>::Scalar::ONE],
    &[<Dual<E1> as Engine>::Scalar::ZERO],
  )
  .unwrap();

  // produce a recursive SNARK
  recursive_snark
    .prove_step(&pp, &test_circuit1, &test_circuit2)
    .unwrap();

  // verify the recursive SNARK
  let (zn_primary, zn_secondary) = recursive_snark
    .verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<Dual<E1> as Engine>::Scalar::ZERO],
    )
    .unwrap();

  assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
  assert_eq!(zn_secondary, vec![<Dual<E1> as Engine>::Scalar::from(5u64)]);
}

#[test]
fn test_ivc_base() {
//   test_ivc_base_with::<PallasEngine>();
  test_ivc_base_with::<Bn256EngineZKPedersen>();
//   test_ivc_base_with::<Secp256k1Engine>();
}

fn test_setup_with<E1: CurveCycleEquipped>() {
  #[derive(Clone, Debug)]
  struct CircuitWithInputize<F: PrimeField> {
    _p: PhantomData<F>,
    counter_type: StepCounterType,
  }

  impl<F> Default for CircuitWithInputize<F>
  where
    F: PrimeField,
  {
    /// Creates a new trivial test circuit with step counter type Incremental
    fn default() -> CircuitWithInputize<F> {
      Self {
        _p: PhantomData::default(),
        counter_type: StepCounterType::Incremental,
      }
    }
  }

  impl<F: PrimeField> StepCircuit<F> for CircuitWithInputize<F> {
    fn arity(&self) -> usize {
      1
    }

    /// Returns the type of the counter for this circuit
    fn get_counter_type(&self) -> StepCounterType {
      self.counter_type
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let x = &z[0];
      // a simplified version of this test would only have one input
      // but beside the Nova Public parameter requirement for a num_io = 2, being
      // probed in this test, we *also* require num_io to be even, so
      // negative testing requires at least 4 inputs
      let y = x.square(cs.namespace(|| "x_sq"))?;
      y.inputize(cs.namespace(|| "y"))?; // inputize y
      let y2 = x.square(cs.namespace(|| "x_sq2"))?;
      y2.inputize(cs.namespace(|| "y2"))?; // inputize y2
      let y3 = x.square(cs.namespace(|| "x_sq3"))?;
      y3.inputize(cs.namespace(|| "y3"))?; // inputize y2
      let y4 = x.square(cs.namespace(|| "x_sq4"))?;
      y4.inputize(cs.namespace(|| "y4"))?; // inputize y2
      Ok(vec![y, y2, y3, y4])
    }
  }

  // produce public parameters with trivial secondary
  let circuit = CircuitWithInputize::<<E1 as Engine>::Scalar>::default();
  let pp = PublicParams::<E1>::setup(
    &circuit,
    &TrivialCircuit::default(),
    &*default_ck_hint(),
    &*default_ck_hint(),
  );
  assert!(pp.is_err());
  assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));

  // produce public parameters with the trivial primary
  let circuit = CircuitWithInputize::<<Dual<E1> as Engine>::Scalar>::default();
  let pp = PublicParams::<E1>::setup(
    &TrivialCircuit::default(),
    &circuit,
    &*default_ck_hint(),
    &*default_ck_hint(),
  );
  assert!(pp.is_err());
  assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));
}

#[test]
fn test_setup() {
  test_setup_with::<Bn256EngineZKPedersen>();
}
