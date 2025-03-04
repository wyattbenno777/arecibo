//! Compression for sumcheck proofs using Groth16
use crate::{
  frontend::{
    groth16::{create_random_proof, generate_random_parameters, Parameters, Proof},
    num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError,
  },
  spartan::{
    polys::univariate::UniPoly,
    sumcheck::SumcheckProof,
  },
  provider::Bn256EngineKZG,
};
use ff::PrimeField;
use halo2curves::bn256::Fr;
use rand::thread_rng;

/// Circuit for verifying a sumcheck proof with provided challenges
#[derive(Clone)]
pub struct SumcheckVerifierCircuit<F: PrimeField> {
  /// The univariate polynomials from the sumcheck proof
  pub polys: Vec<UniPoly<F>>,
  /// The claimed sum
  pub claim: F,
  /// The number of rounds in the sumcheck protocol
  pub num_rounds: usize,
  /// The degree bound for the univariate polynomials
  pub degree_bound: usize,
  /// The verifier's challenges
  pub r: Vec<F>,
}

impl<F: PrimeField> Circuit<F> for SumcheckVerifierCircuit<F> {
  fn synthesize<CS: ConstraintSystem<F>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // Allocate the claim as a public input
    let claim = AllocatedNum::alloc_input(cs.namespace(|| "claim"), || Ok(self.claim))?;

    // Verify that there is a univariate polynomial for each round
    if self.polys.len() != self.num_rounds {
      return Err(SynthesisError::Unsatisfiable);
    }

    // Verify that there is a challenge for each round
    if self.r.len() != self.num_rounds {
      return Err(SynthesisError::Unsatisfiable);
    }

    // Start with the initial claim
    let mut e = claim;

    for i in 0..self.num_rounds {
      // Get the polynomial for this round
      let poly = &self.polys[i];

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

      // Allocate the verifier's challenge for this round
      let r_i = AllocatedNum::alloc(cs.namespace(|| format!("challenge_{}", i)), || {
        Ok(self.r[i])
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

        if j < coeffs.len() - 1 {
          power = power.mul(
            cs.namespace(|| format!("power_update_{}_{}", i, j)),
            &r_i,
          )?;
        }
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
    r: Vec<Fr>,
  ) -> Result<(Self, Parameters<Bn256EngineKZG>), SynthesisError> {
    // Create the circuit for verifying the sumcheck proof
    let circuit = SumcheckVerifierCircuit::<Fr> {
      polys,
      claim,
      num_rounds,
      degree_bound,
      r,
    };

    // Generate parameters for the Groth16 proof system
    let rng = &mut thread_rng();
    let params = generate_random_parameters::<Bn256EngineKZG, _, _>(circuit.clone(), rng)?;

    // Create a Groth16 proof
    let proof = create_random_proof(circuit, &params, rng)?;

    Ok((Self { proof }, params))
  }

  /// Compress a SumcheckProof into a Groth16 proof
  pub fn compress_proof(
    proof: &SumcheckProof<Bn256EngineKZG>,
    claim: Fr,
    num_rounds: usize,
    degree_bound: usize,
    r: Vec<Fr>,
  ) -> Result<(Self, Parameters<Bn256EngineKZG>), SynthesisError> {
    // Convert SumcheckProof to Vec<UniPoly> using the correct claim value for each round
    let mut polys = Vec::with_capacity(num_rounds);
    let mut current_claim = claim;

    for (i, compressed_poly) in proof.compressed_polys.iter().enumerate() {
      // Decompress the polynomial using the current claim value
      let poly = compressed_poly.decompress(&current_claim);

      // Update the claim for the next round if not the last round
      if i < num_rounds - 1 {
        // Evaluate the polynomial at the challenge point to get the next claim
        current_claim = poly.evaluate(&r[i]);
      }

      // Add the polynomial to our collection (clone it to avoid ownership issues)
      polys.push(poly.clone());
    }

    Self::compress(polys, claim, num_rounds, degree_bound, r)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::groth16::{prepare_verifying_key, verify_proof}, provider::Bn256EngineKZG, spartan::{
      polys::multilinear::MultilinearPolynomial,
      sumcheck::SumcheckProof,
    }, traits::{Engine, TranscriptEngineTrait}
  };
  use halo2curves::bn256::Fr;
  use rand::{Rng, thread_rng};

  #[test]
  fn test_sumcheck_compression() {
    let ns = [2, 4, 8, 16];
    let num_tries = 10;

    for &n in &ns {
      for t in 0..num_tries {
        println!("Testing n={} ({}/{})...", n, t+1, num_tries);
        // Generate random polynomials A and B
        let mut rng = thread_rng();
        let size = 1 << n;

        let mut poly_a = MultilinearPolynomial::new(
          (0..size).map(|_| Fr::from(rng.gen_range(1..10) as u64)).collect()
        );

        let mut poly_b = MultilinearPolynomial::new(
          (0..size).map(|_| Fr::from(rng.gen_range(1..10) as u64)).collect()
        );

        // Define the combination function (a * b)
        let comb_func = |a: &Fr, b: &Fr| *a * *b;

        // Calculate the claim (sum of all a_i * b_i)
        let claim = (0..size)
          .map(|i| comb_func(&poly_a[i], &poly_b[i]))
          .fold(Fr::zero(), |acc, val| acc + val);

        let num_rounds = n;
        let degree_bound = 2; // For quadratic polynomials

        // Create a transcript for the proof
        let mut transcript = <Bn256EngineKZG as Engine>::TE::new(b"test");

        // Generate the sumcheck proof
        let (proof, r, _) = SumcheckProof::prove_quad(
          &claim,
          num_rounds,
          &mut poly_a,
          &mut poly_b,
          comb_func,
          &mut transcript,
        ).unwrap();

        // Verify the original proof
        let mut verify_transcript = <Bn256EngineKZG as Engine>::TE::new(b"test");
        assert!(proof.verify(claim, num_rounds, degree_bound, &mut verify_transcript).is_ok());

        // Compress the proof
        let (compressed_proof, params) = SumcheckCompression::compress_proof(
          &proof,
          claim,
          num_rounds,
          degree_bound,
          r,
        ).unwrap();

        // Verify the compressed proof
        let pvk = prepare_verifying_key(&params.vk);
        let verified = verify_proof(&pvk, &compressed_proof.proof, &[claim]).unwrap();

        assert!(verified, "Compressed proof verification failed for n={}", n);
      }
    }
  }
}
