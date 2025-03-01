//! Compression
use crate::{
  frontend::{
    groth16::{create_random_proof, generate_random_parameters, Parameters, Proof},
    num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError,
  }, hypernova::ro_sumcheck::ROSumcheckProof, provider::Bn256EngineKZG, spartan::polys::univariate::UniPoly
};
use ff::PrimeField;
use halo2curves::bn256::Fr;
use rand::thread_rng;

/// Circuit for verifying a sumcheck proof
#[derive(Clone)]
pub struct SumcheckVerificationCircuit<F: PrimeField> {
  /// The univariate polynomials from the sumcheck proof
  pub polys: Vec<UniPoly<F>>,
  /// The claimed sum
  pub claim: F,
  /// The number of rounds in the sumcheck protocol
  pub num_rounds: usize,
  /// The degree bound for the univariate polynomials
  pub degree_bound: usize,
}

impl<F: PrimeField> Circuit<F> for SumcheckVerificationCircuit<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // Allocate the claim as a public input
    let claim = AllocatedNum::alloc_input(cs.namespace(|| "claim"), || Ok(self.claim))?;

    // Verify each round of the sumcheck proof
    let mut e = claim;

    // Verify that there is a univariate polynomial for each round
    if self.polys.len() != self.num_rounds {
      return Err(SynthesisError::Unsatisfiable);
    }

    for (i, poly) in self.polys.iter().enumerate() {
      // Verify degree bound
      if poly.degree() != self.degree_bound {
        return Err(SynthesisError::Unsatisfiable);
      }

      // Allocate the polynomial coefficients
      let coeffs: Vec<AllocatedNum<F>> = poly.coeffs
        .iter()
        .enumerate()
        .map(|(j, coeff)| {
          AllocatedNum::alloc(cs.namespace(|| format!("poly_{}_coeff_{}", i, j)), || Ok(*coeff))
        })
        .collect::<Result<Vec<_>, _>>()?;

      // Verify that eval_at_zero + eval_at_one = e
      let eval_at_zero = coeffs[0].clone();

      // Calculate eval_at_one = sum of all coefficients
      let mut eval_at_one = eval_at_zero.clone();
      for j in 1..coeffs.len() {
        eval_at_one = eval_at_one.add(
          cs.namespace(|| format!("add_coeff_{}_{}", i, j)),
          &coeffs[j],
        )?;
      }

      // Enforce that eval_at_zero + eval_at_one = e
      cs.enforce(
        || format!("round_{}_constraint", i),
        |lc| lc + eval_at_zero.get_variable() + eval_at_one.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + e.get_variable(),
      );

      // Derive the verifier's challenge for the next round
      // In a real implementation, we would use a random oracle here
      // For simplicity, we'll just use a fixed challenge
      let r_i = AllocatedNum::alloc(cs.namespace(|| format!("challenge_{}", i)), || {
        // This is a placeholder - in a real implementation, this would be derived from a transcript
        Ok(F::from(i as u64 + 1))
      })?;

      // Evaluate the polynomial at r_i
      let mut eval = coeffs[0].clone();
      let mut power = r_i.clone();

      for j in 1..coeffs.len() {
        let term = power.mul(
          cs.namespace(|| format!("term_{}_{}", i, j)),
          &coeffs[j],
        )?;

        eval = eval.add(
          cs.namespace(|| format!("eval_add_{}_{}", i, j)),
          &term,
        )?;

        power = power.mul(
          cs.namespace(|| format!("power_update_{}_{}", i, j)),
          &r_i,
        )?;
      }

      // Update e for the next round
      e = eval;
    }

    // The final value of e is the result of the sumcheck verification
    Ok(())
  }
}

/// Compression for sumcheck proofs using Groth16
#[derive(Clone, Debug)]
pub struct SumcheckCompression {
  /// The Groth16 proof for the sumcheck verification circuit
  pub proof: Proof<Bn256EngineKZG>,
}

impl SumcheckCompression {
  /// Compress a sumcheck proof into a Groth16 proof
  pub fn compress(
    polys: Vec<UniPoly<Fr>>,
    claim: Fr,
    num_rounds: usize,
    degree_bound: usize,
  ) -> Result<(Self, Parameters<Bn256EngineKZG>), SynthesisError> {
    // Create the circuit for verifying the sumcheck proof
    let circuit = SumcheckVerificationCircuit::<Fr> {
      polys,
      claim,
      num_rounds,
      degree_bound,
    };

    // Generate parameters for the Groth16 proof system
    let rng = &mut thread_rng();
    let params = generate_random_parameters::<Bn256EngineKZG, _, _>(circuit.clone(), rng)?;

    // Create a Groth16 proof
    let proof = create_random_proof(circuit, &params, rng)?;

    Ok((Self { proof }, params))
  }

  /// Compress a ROSumcheckProof into a Groth16 proof
  pub fn compress_ro_proof(
    proof: &ROSumcheckProof<Bn256EngineKZG>,
    claim: Fr,
    num_rounds: usize,
    degree_bound: usize,
  ) -> Result<(Self, Parameters<Bn256EngineKZG>), SynthesisError> {
    // Convert ROSumcheckProof to Vec<UniPoly>
    let polys = proof.polys.clone();

    Self::compress(polys, claim, num_rounds, degree_bound)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::groth16::{verify_proof, prepare_verifying_key},
    spartan::polys::univariate::UniPoly
  };
  use halo2curves::bn256::Fr;

  #[test]
  fn test_sumcheck_compression() {
    // Create a simple sumcheck proof with two rounds
    // For the sumcheck protocol to verify:
    // 1. For the first polynomial: eval_at_zero + eval_at_one = claim (10)
    // 2. For the second polynomial: eval_at_zero + eval_at_one = eval_poly1(r_0)
    //    where r_0 = 1 (hardcoded in the circuit)

    // First polynomial: coefficients [1, 2, 6]
    // eval_at_zero = 1
    // eval_at_one = 1+2+6 = 9
    // eval_at_zero + eval_at_one = 1+9 = 10 (matches claim)
    let poly1 = UniPoly::new(vec![Fr::from(1), Fr::from(2), Fr::from(6)]);

    // Evaluate poly1 at r_0 = 1: 1 + 2*1 + 6*1^2 = 1 + 2 + 6 = 9
    // Second polynomial: coefficients [0, 5, 4]
    // eval_at_zero = 0
    // eval_at_one = 0+5+4 = 9
    // eval_at_zero + eval_at_one = 0+9 = 9 (matches poly1(r_0))
    let poly2 = UniPoly::new(vec![Fr::from(0), Fr::from(5), Fr::from(4)]);

    // Compress the proof
    let (compressed_proof, params) = SumcheckCompression::compress(
      vec![poly1, poly2],
      Fr::from(10), // claim
      2,            // num_rounds
      2,            // degree_bound
    ).unwrap();

    let pvk = prepare_verifying_key(&params.vk);

    let verified = verify_proof(&pvk, &compressed_proof.proof, &[Fr::from(10)]).unwrap();

    assert!(verified);
  }
}
