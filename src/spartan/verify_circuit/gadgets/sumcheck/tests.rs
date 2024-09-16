use crate::provider::PallasEngine;

use crate::spartan::powers;

use crate::traits::TranscriptEngineTrait;
use crate::{
  spartan::{polys::multilinear::MultilinearPolynomial, sumcheck::SumcheckProof},
  traits::Engine,
};

/// The primary curve to use for the sumcheck proof.
type E1 = PallasEngine;

/// Convert a u64 to a scalar.
fn util_u64_to_scalar<E: Engine>(x: u64) -> E::Scalar {
  E::Scalar::from(x)
}

/// Convert a vector of u64s to a vector of scalars.
fn util_vec_u64_to_scalar<E: Engine>(v: Vec<u64>) -> Vec<E::Scalar> {
  v.into_iter().map(|x| util_u64_to_scalar::<E>(x)).collect()
}

#[test]
fn test_sumcheck() -> anyhow::Result<()> {
  // Get the claim to prove
  let claim = <E1 as Engine>::Scalar::from(1000);
  let num_rounds = 3_usize;

  let Z = util_vec_u64_to_scalar::<E1>(vec![6, 6, 6, 6, 6, 11, 9, 16]);
  let mut polynomial_a = MultilinearPolynomial::new(Z);

  // zero polynomial
  let mut polynomial_b =
    MultilinearPolynomial::<<E1 as Engine>::Scalar>::new(util_vec_u64_to_scalar::<E1>(vec![
      1, 1, 1, 1, 1, 1, 1, 1,
    ]));

  let mut transcript = <E1 as Engine>::TE::new(b"test");
  let comb_func = |poly_A_comp: &<E1 as Engine>::Scalar,
                   poly_B_comp: &<E1 as Engine>::Scalar|
   -> <E1 as Engine>::Scalar { *poly_A_comp * *poly_B_comp };

  let (sc_proof, _, _) = SumcheckProof::<E1>::prove_quad(
    &claim,
    num_rounds,
    &mut polynomial_a,
    &mut polynomial_b,
    comb_func,
    &mut transcript,
  )?;

  let degree_bound = 2_usize;
  let mut verifier_transcript = <E1 as Engine>::TE::new(b"test");
  let (_, _) = sc_proof.verify(claim, num_rounds, degree_bound, &mut verifier_transcript)?;

  Ok(())
}

#[test]
fn test_sumcheck_batched() -> anyhow::Result<()> {
  // Get the claim to prove
  let claims = util_vec_u64_to_scalar::<E1>(vec![66u64, 66u64]);
  let num_rounds = [3_usize; 2];

  // Get poly_A
  let Z = util_vec_u64_to_scalar::<E1>(vec![6, 6, 6, 6, 6, 11, 9, 16]);
  let polynomial_a = MultilinearPolynomial::new(Z);
  let poly_A_vec = vec![polynomial_a.clone(), polynomial_a];

  // zero polynomial
  let polynomial_b =
    MultilinearPolynomial::<<E1 as Engine>::Scalar>::new(util_vec_u64_to_scalar::<E1>(vec![
      0, 0, 0, 0, 0, 0, 0, 0,
    ]));
  let poly_B_vec = vec![polynomial_b.clone(), polynomial_b];

  // Prepare inputs for protocol proving
  let mut transcript = <E1 as Engine>::TE::new(b"test");
  let comb_func = |poly_A_comp: &<E1 as Engine>::Scalar,
                   poly_B_comp: &<E1 as Engine>::Scalar|
   -> <E1 as Engine>::Scalar { *poly_A_comp + *poly_B_comp };

  // Sample challenge for random linear-combination of outer claims
  let num_instances = poly_A_vec.len();
  let r_coeffs = transcript.squeeze(b"r_coeffs")?;
  let coeffs = powers(&r_coeffs, num_instances);

  // Run IOP
  let (sc_proof, _r, _new_claims) = SumcheckProof::<E1>::prove_quad_batch(
    &claims,
    &num_rounds,
    poly_A_vec,
    poly_B_vec,
    &coeffs,
    comb_func,
    &mut transcript,
  )?;

  // Get values for verification
  let degree_bound = 2_usize;
  let mut verifier_transcript = <E1 as Engine>::TE::new(b"test");
  let r_coeffs = transcript.squeeze(b"r_coeffs")?;
  let coeffs = powers(&r_coeffs, num_instances);

  // verify sumcheck
  sc_proof.verify_batch(
    &claims,
    &num_rounds,
    &coeffs,
    degree_bound,
    &mut verifier_transcript,
  )?;

  Ok(())
}
