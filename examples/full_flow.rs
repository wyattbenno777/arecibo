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
  onchain::{decider::{prepare_calldata, Decider}, eth::evm::{compile_solidity, Evm}, utils::{get_formatted_calldata, get_function_selector_for_nova_cyclefold_verifier}, verifiers::{groth16::SolidityGroth16VerifierKey, kzg::SolidityKZGVerifierKey, nebula::{get_decider_template_for_cyclefold_decider, NovaCycleFoldVerifierKey}}},
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
  let mut rs: RecursiveSNARK<E1> =
    RecursiveSNARK::<E1>::new(&rs_pp, &f_circuit, &z0).unwrap();

  for i in 0..num_steps {
    let start = Instant::now();
    rs
      .prove_step(&rs_pp, &f_circuit, IC_i)
      .unwrap();

    IC_i = rs.increment_commitment(&rs_pp, &f_circuit);
    println!("RecursiveSNARK::prove {} : took {:?} ", i, start.elapsed());
  }

  // verify the recursive SNARK
  println!("Verifying a RecursiveSNARK...");
  let res = rs.verify(&rs_pp, num_steps, &z0, IC_i);
  println!("RecursiveSNARK::verify: {:?}", res.is_ok(),);
  res.unwrap();

  let mut rng = thread_rng();
  let start = Instant::now();
  
  let (decider_pk, decider_vk) = Decider::setup(&rs_pp, &mut rng, z0.len()).unwrap();
  println!("Decider::setup: took {:?}", start.elapsed());
  let start = Instant::now();
  let proof = Decider::prove(&rs_pp, &decider_pk, &rs, &mut rng);
  match &proof {
    Ok(_) => println!("Decider::prove: Ok, took {:?}", start.elapsed()),
    Err(e) => println!(
      "Decider::prove: Error: {:?}, took {:?}",
      e,
      start.elapsed()
    ),
  }
  assert!(proof.is_ok());

  let proof = proof.unwrap();

  let start = Instant::now();
  let res = Decider::verify(
    &proof,
     decider_vk.clone(), 
     Fr::from(num_steps as u64), 
     z0, 
     rs.zi.clone(), 
     (rs.r_U_primary.comm_W, rs.r_U_primary.comm_E), 
     rs.l_u_primary.comm_W);
  println!("Decider::verify: {:?}, took {:?}", res.is_ok(), start.elapsed());

  assert!(res.is_ok());
  // Now, let's generate the Solidity code that verifies this Decider final proof
  let function_selector = get_function_selector_for_nova_cyclefold_verifier(rs.z0.len() * 2 + 1);

  // let calldata = [252, 61, 118, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 21, 162, 192, 117, 162, 202, 4, 103, 234, 127, 226, 89, 137, 215, 128, 56, 255, 5, 28, 176, 145, 62, 23, 22, 2, 101, 137, 125, 124, 150, 0, 45, 45, 81, 131, 139, 34, 118, 40, 116, 220, 41, 207, 246, 192, 57, 139, 100, 81, 92, 54, 216, 130, 203, 193, 147, 159, 58, 75, 162, 129, 194, 248, 204, 0, 6, 51, 218, 245, 230, 104, 189, 209, 20, 12, 77, 248, 233, 57, 99, 53, 141, 102, 140, 251, 21, 133, 34, 101, 166, 121, 107, 207, 205, 166, 46, 14, 90, 98, 41, 184, 202, 139, 3, 234, 157, 182, 174, 97, 8, 201, 144, 241, 56, 90, 193, 74, 219, 160, 246, 154, 42, 130, 181, 98, 76, 111, 161, 19, 111, 43, 163, 248, 159, 126, 102, 136, 123, 56, 239, 180, 170, 132, 146, 41, 88, 227, 190, 232, 203, 182, 65, 138, 210, 235, 15, 137, 58, 121, 122, 26, 38, 253, 156, 82, 146, 37, 151, 130, 31, 33, 54, 113, 66, 85, 229, 87, 208, 28, 13, 101, 249, 19, 165, 162, 10, 81, 62, 56, 82, 116, 66, 3, 83, 226, 70, 118, 95, 188, 5, 227, 224, 39, 94, 72, 92, 25, 218, 82, 153, 220, 201, 235, 190, 215, 189, 55, 49, 2, 41, 27, 107, 116, 197, 20, 240, 24, 134, 205, 250, 115, 90, 99, 190, 183, 108, 135, 156, 19, 246, 93, 56, 105, 199, 103, 132, 12, 133, 19, 92, 178, 74, 41, 203, 57, 176, 40, 197, 9, 78, 16, 219, 242, 229, 125, 233, 181, 24, 74, 192, 204, 153, 42, 29, 36, 59, 35, 49, 131, 120, 236, 147, 9, 214, 105, 78, 102, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 59, 62, 14, 25, 113, 143, 171, 220, 229, 18, 225, 183, 85, 205, 53, 10, 120, 44, 66, 182, 8, 221, 114, 79, 89, 17, 96, 247, 253, 94, 43, 207, 127, 175, 128, 202, 193, 127, 26, 231, 108, 200, 127, 182, 42, 139, 218, 6, 154, 30, 131, 174, 56, 88, 63, 76, 115, 195, 97, 28, 236, 208, 228, 30, 190, 155, 140, 54, 21, 169, 197, 182, 113, 187, 33, 247, 32, 67, 155, 26, 170, 140, 210, 56, 30, 147, 144, 195, 147, 1, 73, 205, 221, 73, 53, 82, 44, 137, 9, 113, 95, 6, 249, 117, 117, 199, 22, 213, 71, 81, 3, 17, 140, 165, 173, 57, 19, 22, 200, 193, 97, 250, 185, 16, 219, 41, 176, 169, 78, 28, 209, 218, 211, 185, 101, 208, 89, 58, 57, 39, 246, 125, 123, 4, 40, 15, 99, 204, 151, 55, 185, 124, 254, 120, 182, 81, 71, 139, 250, 130, 242, 230, 54, 64, 15, 127, 9, 249, 104, 127, 8, 99, 92, 153, 210, 12, 107, 62, 103, 27, 221, 132, 108, 88, 128, 171, 195, 12, 184, 235, 164, 129, 43, 8, 98, 87, 197, 104, 219, 96, 105, 204, 128, 128, 248, 5, 148, 37, 198, 242, 180, 5, 16, 47, 135, 25, 248, 202, 227, 34, 204, 60, 182, 5, 68, 158, 184, 8, 162, 204, 90, 71, 122, 24, 32, 173, 58, 60, 197, 37, 65, 11, 88, 155, 126, 228, 46, 130, 210, 66, 251, 10, 52, 174, 214, 61, 67, 220, 85, 182, 220, 76, 20, 248, 25, 90, 100, 104, 211, 64, 91, 8, 217, 177, 174, 43, 55, 203, 54, 108, 52, 212, 145, 47, 54, 71, 111, 119, 119, 113, 73, 19, 10, 66, 3, 177, 39, 230, 186, 41, 177, 123, 145, 20, 105, 187, 106, 199, 20, 191, 25, 44, 81, 121, 185, 222, 178, 191, 89, 109, 201, 68, 15, 249, 68, 82, 3, 212, 59, 113, 82, 114, 198, 213, 73, 34, 158, 251, 246, 165, 232, 151, 193, 2, 155, 160, 173, 236, 42, 193, 91, 12, 89, 6, 140, 134, 155, 39, 39, 185, 184, 149, 71, 3, 62, 63, 100, 40, 253, 141, 33, 52, 4, 90, 222, 206, 165, 23, 12, 155, 137, 68, 176, 187, 48, 214, 66, 245, 33, 226, 58, 189, 108, 227, 60, 122, 47, 168, 7, 5, 38, 43, 162, 89, 239, 201, 8, 72, 16, 84, 46, 137, 137, 226, 193, 195, 3, 85, 226, 45, 97, 4, 18, 221, 151, 158, 238, 206, 191, 223, 189, 38, 61, 253, 38, 105, 105, 238, 79, 89, 27, 57, 50, 96, 168, 252, 70, 111, 87, 90, 251, 244, 187, 13, 86, 232, 201, 158, 118, 128, 163, 137, 120, 29, 103, 68, 6, 79, 128, 240, 222, 160, 171, 179, 99, 1, 206, 120, 177, 36, 99, 141, 190, 231, 81, 122, 249, 45, 59, 134, 173, 132, 200, 150, 158, 0, 37, 10, 0, 34, 197, 110, 166, 229, 156, 130, 210, 238, 188, 70, 28, 198, 127, 63, 33, 188, 211, 100, 82, 61, 70, 72, 191, 111, 43, 203, 4].to_vec();


  let calldata: Vec<u8> = prepare_calldata(
    function_selector,
    Fr::from(rs.i as u64),
    &rs.z0,
    &rs.zi,
    &rs.r_U_primary,
    &rs.l_u_primary,
    &proof,
  ).unwrap();

  println!("Calldata: {:?}", calldata);

  // prepare the setup params for the solidity verifier
  let nova_cyclefold_vk = NovaCycleFoldVerifierKey::from(
    (
      decider_vk.pp_hash,
      SolidityGroth16VerifierKey::from(decider_vk.groth16_vk),
      SolidityKZGVerifierKey::from((decider_vk.kzg_vk, Vec::new())),
      rs.z0.len(),
    )
  );

  // generate the solidity code
  let decider_solidity_code = get_decider_template_for_cyclefold_decider(nova_cyclefold_vk);

  // verify the proof against the solidity code in the EVM
  let nova_cyclefold_verifier_bytecode = compile_solidity(&decider_solidity_code, "NovaDecider");
  let mut evm = Evm::default();

  let verifier_address = evm.create(nova_cyclefold_verifier_bytecode);
  println!("verifier_address: {:?}", verifier_address);
  let (gas, output) = evm.call(verifier_address, calldata.clone());
  println!("Solidity::verify: {:?}, gas: {:?}", output, gas);
  assert_eq!(*output.last().unwrap(), 1);

  // save smart contract and the calldata
  println!("storing nova-verifier.sol and the calldata into files");
  use std::fs;
  fs::write(
    "./examples/nova-verifier.sol",
    decider_solidity_code.clone(),
  ).expect("Unable to write to file");
  fs::write("./examples/solidity-calldata.calldata", calldata.clone()).expect("");
  let s = get_formatted_calldata(calldata.clone());
  fs::write("./examples/solidity-calldata.inputs", s.join(",\n")).expect("");
}

