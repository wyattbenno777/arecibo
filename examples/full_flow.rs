#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::upper_case_acronyms)]
///
/// This example performs the full flow:
/// - define the circuit to be folded
/// - fold the circuit with Nova+CycleFold's IVC
/// - generate a DeciderEthCircuit final proof
/// - generate the Solidity contract that verifies the proof
/// - verify the proof in the EVM
use arecibo::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  nebula::rs::{PublicParams, RecursiveSNARK, StepCircuit},
  onchain::{decider::{prepare_calldata, Decider}, utils::get_function_selector_for_nova_cyclefold_verifier, verifiers::nova::NovaCycleFoldVerifierKey},
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{snark::RelaxedR1CSSNARKTrait, Engine},
};
use ff::Field;
use halo2curves::bn256::{Bn256, Fr};

use rand::thread_rng;
use std::time::Instant;

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type EE1 = arecibo::provider::hyperkzg::EvaluationEngine<Bn256, E1>;
type EE2 = arecibo::provider::ipa_pc::EvaluationEngine<E2>;
type S1 = arecibo::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
type S2 = arecibo::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK

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

fn main() {
  let num_steps = 5;

  let f_circuit = CubicFCircuit::new();

  // produce public parameters
  let start = Instant::now();
  println!("Producing public parameters...");
  let rs_pp = PublicParams::<E1>::setup(&f_circuit, &*S1::ck_floor(), &*S2::ck_floor());
  println!("PublicParams::setup, took {:?} ", start.elapsed());

  println!(
    "Number of constraints per step (primary circuit): {}",
    rs_pp.num_constraints().0
  );
  println!(
    "Number of constraints per step (secondary circuit): {}",
    rs_pp.num_constraints().1
  );

  println!(
    "Number of variables per step (primary circuit): {}",
    rs_pp.num_variables().0
  );
  println!(
    "Number of variables per step (secondary circuit): {}",
    rs_pp.num_variables().1
  );

  // produce a recursive SNARK
  println!("Generating a RecursiveSNARK...");

  let mut IC_i = <E1 as Engine>::Scalar::ZERO;
  let z0 = vec![<E1 as Engine>::Scalar::from(3u64)];
  let mut recursive_snark: RecursiveSNARK<E1> =
    RecursiveSNARK::<E1>::new(&rs_pp, &f_circuit, &z0).unwrap();

  for i in 0..num_steps {
    let start = Instant::now();
    recursive_snark
      .prove_step(&rs_pp, &f_circuit, IC_i)
      .unwrap();

    IC_i = recursive_snark.increment_commitment(&rs_pp, &f_circuit);
    println!("RecursiveSNARK::prove {} : took {:?} ", i, start.elapsed());
  }

  // verify the recursive SNARK
  println!("Verifying a RecursiveSNARK...");
  let res = recursive_snark.verify(&rs_pp, num_steps, &z0, IC_i);
  println!("RecursiveSNARK::verify: {:?}", res.is_ok(),);
  res.unwrap();

  let mut rng = thread_rng();
  let start = Instant::now();
  
  let (pk, vk) = Decider::setup(&rs_pp, &mut rng).unwrap();
  println!("Decider::setup: took {:?}", start.elapsed());
  let start = Instant::now();
  let proof = Decider::prove(&rs_pp, &pk, &recursive_snark, &mut rng);
  match &proof {
    Ok(_) => println!("CompressedSNARK::prove: Ok, took {:?}", start.elapsed()),
    Err(e) => println!(
      "CompressedSNARK::prove: Error: {:?}, took {:?}",
      e,
      start.elapsed()
    ),
  }
  assert!(proof.is_ok());

  let proof = proof.unwrap();

  let start = Instant::now();
  let res = Decider::verify(
    &proof,
     vk, 
     Fr::from(num_steps as u64), 
     z0, 
     recursive_snark.zi.clone(), 
     (recursive_snark.r_U_primary.comm_W, recursive_snark.r_U_primary.comm_E), 
     recursive_snark.l_u_primary.comm_W);
  println!("Decider::verify: {:?}, took {:?}", res.is_ok(), start.elapsed());

  let compressed_snark = res.unwrap();

  // let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  // bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();
  // let compressed_snark_encoded = encoder.finish().unwrap();
  // println!(
  //   "CompressedSNARK::len {:?} bytes",
  //   compressed_snark_encoded.len()
  // );

  // // verify the compressed SNARK
  // println!("Verifying a CompressedSNARK...");
  // let start = Instant::now();
  // let res = compressed_snark.verify(
  //   &vk,
  //   num_steps,
  //   &[<E1 as Engine>::Scalar::ZERO],
  //   &[<E2 as Engine>::Scalar::ZERO],
  // );
  // println!(
  //   "CompressedSNARK::verify: {:?}, took {:?}",
  //   res.is_ok(),
  //   start.elapsed()
  // );
  // res.unwrap();
  println!("=========================================================");

  // Now, let's generate the Solidity code that verifies this Decider final proof
  let function_selector = get_function_selector_for_nova_cyclefold_verifier(recursive_snark.z0.len() * 2 + 1);

  let calldata: Vec<u8> = prepare_calldata(
    function_selector,
    Fr::from(recursive_snark.i as u64),
    recursive_snark.z0,
    recursive_snark.zi,
    &recursive_snark.r_U_primary,
    &recursive_snark.l_u_primary,
    &proof,
  ).unwrap();

  // prepare the setup params for the solidity verifier
  // let nova_cyclefold_vk = NovaCycleFoldVerifierKey::from((vk, 1));

  // // generate the solidity code
  // let decider_solidity_code = get_decider_template_for_cyclefold_decider(nova_cyclefold_vk);

  // // verify the proof against the solidity code in the EVM
  // let nova_cyclefold_verifier_bytecode = compile_solidity(&decider_solidity_code, "NovaDecider");
  // let mut evm = Evm::default();
  // let verifier_address = evm.create(nova_cyclefold_verifier_bytecode);
  // let (_, output) = evm.call(verifier_address, calldata.clone());
  // assert_eq!(*output.last().unwrap(), 1);

  // // save smart contract and the calldata
  // println!("storing nova-verifier.sol and the calldata into files");
  // use std::fs;
  // fs::write(
  //   "./examples/nova-verifier.sol",
  //   decider_solidity_code.clone(),
  // )?;
  // fs::write("./examples/solidity-calldata.calldata", calldata.clone())?;
  // let s = solidity_verifiers::utils::get_formatted_calldata(calldata.clone());
  // fs::write("./examples/solidity-calldata.inputs", s.join(",\n")).expect("");
}
