//! Compression for sumcheck proofs using Groth16
use crate::{
  constants::{DEFAULT_ABSORBS, NUM_CHALLENGE_BITS},
  frontend::{
    groth16::{create_random_proof, generate_random_parameters, Parameters, Proof}, num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError
  }, gadgets::{hypernova::AllocatedUniPoly, int::enforce_equal, le_bits_to_num}, provider::Bn256EngineKZG, spartan::{
    polys::univariate::UniPoly,
    sumcheck::SumcheckProof,
  },
  traits::{
    CurveCycleEquipped, Dual, Engine, ROCircuitTrait,
    ROConstantsCircuit,
  },
};
use halo2curves::bn256::Fr;
use rand::thread_rng;

/// Circuit for verifying a sumcheck proof with provided challenges
#[derive(Clone)]
pub struct SumcheckVerifierCircuit<E: Engine + CurveCycleEquipped> {
  /// The univariate polynomials from the sumcheck proof
  pub polys: Vec<UniPoly<E::Scalar>>,
  /// The claimed sum
  pub claim: E::Scalar,
  /// The degree bound for the univariate polynomials
  pub degree_bound: usize,
}

impl<E: Engine> Circuit<E::Scalar> for SumcheckVerifierCircuit<E> where E: CurveCycleEquipped {
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // Allocate the claim as a public input
    let num_rounds = self.polys.len();
    let polys = (0..num_rounds)
      .map(|i| {
        AllocatedUniPoly::alloc(cs.namespace(|| "polys"), Some(&self.polys[i].clone()))
      }).collect::<Result<Vec<_>, _>>()?;
    // Start with the initial claim
    let mut e = AllocatedNum::alloc_input(cs.namespace(|| "claim"), || Ok(self.claim))?;
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ROConstantsCircuit::<Dual<E>>::default(), DEFAULT_ABSORBS);
    let mut rx = Vec::with_capacity(num_rounds);

    for i in 0..num_rounds {
      // Get the polynomial for this round
      let poly = &polys[i];
      let s0 = poly.eval_at_zero();
      let s1 = poly.eval_at_one(cs.namespace(|| format!("eval at one {i}")))?;
      let s0_s1 = s0.add(cs.namespace(|| format!("s0 + s1 {i}")), &s1)?;
      enforce_equal(
        cs,
        || format!("poly(0) + poly(1) = e {i}"),
        &s0_s1,
        &e,
      );
      poly.absorb_in_ro(&mut ro)?;
      let r_i_bits = ro.squeeze(cs.namespace(|| format!("r_{i} bits")), NUM_CHALLENGE_BITS)?;
      let r_i = le_bits_to_num(cs.namespace(|| format!("r_{i}")), &r_i_bits)?;
      e = poly.eval(cs.namespace(|| format!("eval_{i}")), &r_i)?;
      rx.push(r_i);
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
    degree_bound: usize,
  ) -> Result<(Self, Parameters<Bn256EngineKZG>), SynthesisError> {
    // Create the circuit for verifying the sumcheck proof
    let circuit = SumcheckVerifierCircuit::<Bn256EngineKZG> {
      polys,
      claim,
      degree_bound,
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
    degree_bound: usize,
    r: Vec<Fr>,
  ) -> Result<(Self, Parameters<Bn256EngineKZG>), SynthesisError> {
    // Convert SumcheckProof to Vec<UniPoly> using the correct claim value for each round
    let num_rounds = proof.compressed_polys.len();
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

    Self::compress(polys, claim, degree_bound)
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
  use group::{Curve, Group};
  use halo2curves::bn256::{Fr, G1};
  use rand::{Rng, thread_rng};
  use rayon::prelude::*;
  use std::time::Instant;

  #[test]
  fn test_sumcheck_compression_completeness() {
    let ns = [2, 4, 8, 16];
    let num_tries = 1;

    for &n in &ns {
      for t in 0..num_tries {
        let start = Instant::now();
        println!("Testing n={} ({}/{})...", n, t+1, num_tries);
        let size = 1 << n;

        let mut poly_a = MultilinearPolynomial::new(
          (0..size)
            .into_par_iter()
            .map(|_| {
                let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
                Fr::from(rng.gen_range(1..10) as u64)
            })
            .collect()
        );

        let mut poly_b = MultilinearPolynomial::new(
          (0..size)
            .into_par_iter()
            .map(|_| {
                let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
                Fr::from(rng.gen_range(1..10) as u64)
            })
            .collect()
        );

        // Define the combination function (a * b)
        let comb_func = |a: &Fr, b: &Fr| *a * *b;

        // Calculate the claim (sum of all a_i * b_i)
        let claim = (0..size)
          .into_par_iter()
          .map(|i| comb_func(&poly_a[i], &poly_b[i]))
          .reduce(|| Fr::zero(), |acc, val| acc + val);

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
          degree_bound,
          r,
        ).unwrap();

        // Verify the compressed proof
        let pvk = prepare_verifying_key(&params.vk);
        let verified = verify_proof(&pvk, &compressed_proof.proof, &[claim]).unwrap();

        assert!(verified, "Compressed proof verification failed for n={}", n);
        let end = Instant::now();
        println!("Time elapsed: {:?}", end.duration_since(start));
      }
    }
  }

  #[test]
  fn test_sumcheck_compression_soundness() {
    let ns = [2, 4, 8, 16];
    let num_tries = 1;

    for &n in &ns {
      for t in 0..num_tries {
        let start = Instant::now();
        println!("Testing n={} ({}/{})...", n, t+1, num_tries);
        // Generate random polynomials A and B
        let mut rng = thread_rng();
        let size = 1 << n;

        let mut poly_a = MultilinearPolynomial::new(
          (0..size)
            .into_par_iter()
            .map(|_| {
                let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
                Fr::from(rng.gen_range(1..10) as u64)
            })
            .collect()
        );

        let mut poly_b = MultilinearPolynomial::new(
          (0..size)
            .into_par_iter()
            .map(|_| {
                let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
                Fr::from(rng.gen_range(1..10) as u64)
            })
            .collect()
        );

        // Define the combination function (a * b)
        let comb_func = |a: &Fr, b: &Fr| *a * *b;

        // Calculate the claim (sum of all a_i * b_i)
        let claim = (0..size)
          .into_par_iter()
          .map(|i| comb_func(&poly_a[i], &poly_b[i]))
          .reduce(|| Fr::zero(), |acc, val| acc + val);

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
          degree_bound,
          r.clone(),
        ).unwrap();

        // Verify the compressed proof for invalid claim
        let pvk = prepare_verifying_key(&params.vk);
        let invalid_claim = claim + Fr::one();
        let invalid_claim_verified = verify_proof(&pvk, &compressed_proof.proof, &[invalid_claim]).unwrap();
        assert!(!invalid_claim_verified, "Compressed proof verification should failed, but succeeded for n={}", n);

        // Verify the compressed proof for invalid proof
        let invalid_proof = Proof {
          a: compressed_proof.proof.a,
          b: compressed_proof.proof.b,
          c: G1::random(&mut rng).to_affine(),
        };
        let invalid_proof_verified = verify_proof(&pvk, &invalid_proof, &[claim]).unwrap();
        assert!(!invalid_proof_verified, "Compressed proof verification should failed, but succeeded for n={}", n);

        // Try compressing to the invalid sumcheck proof and verify it
        let invalid_sumcheck_proof = SumcheckProof::<Bn256EngineKZG>::new(proof.compressed_polys.clone().into_par_iter().map(|poly| {
          poly.clone().decompress(&(claim + Fr::one())).compress()
        }).collect());
        let compressed_invalid_proof = SumcheckCompression::compress_proof(
          &invalid_sumcheck_proof,
          claim,
          degree_bound,
          r,
        ).unwrap();
        let invalid_compressed_proof_verified = verify_proof(&pvk, &compressed_invalid_proof.0.proof, &[claim]).unwrap();
        assert!(!invalid_compressed_proof_verified, "Compressed proof verification should failed, but succeeded for n={}", n);
        let end = Instant::now();
        println!("Time elapsed: {:?}", end.duration_since(start));
      }
    }
  }
}
