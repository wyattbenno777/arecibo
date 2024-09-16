use super::poly::AllocatedUniPoly;
use super::poseidon_transcript::PoseidonTranscriptCircuit;
use crate::gadgets::alloc_zero;

use crate::{spartan::sumcheck::SumcheckProof, traits::Engine};

use bellpepper_core::Namespace;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;

use generic_array::typenum::U24;
use itertools::Itertools as _;
use poseidon_sponge::sponge::circuit::SpongeCircuit;

#[cfg(test)]
mod tests;

/// Required data to verify sumcheck proof.
#[derive(Debug, Clone)]
pub struct SCVerifyGadgetInputs<E: Engine> {
  pub(super) claim: AllocatedNum<E::Scalar>,
  pub(super) num_rounds: AllocatedNum<E::Scalar>,
  pub(super) degree_bound: AllocatedNum<E::Scalar>,
  pub(super) sc_proof: SumcheckProof<E>,
}

/// Sumcheck verifier algorithm encoded in a constraint system (r1cs).
#[derive(Debug)]
pub struct SCVerifyGadget<E: Engine> {
  inputs: SCVerifyGadgetInputs<E>,
}

impl<E: Engine> SCVerifyGadget<E> {
  /// Instantiate a new Sumcheck verify circuit.
  pub fn new(inputs: SCVerifyGadgetInputs<E>) -> Self {
    Self { inputs }
  }

  /// Run verifier algorithm for Sumcheck in the constraint system.
  pub fn verify<'a, CS: ConstraintSystem<E::Scalar>>(
    self,
    cs: &mut Namespace<'a, E::Scalar, CS>,
    transcript: &mut PoseidonTranscriptCircuit<E>,
    sponge: &mut SpongeCircuit<'a, E::Scalar, U24, CS>,
    label: &str,
  ) -> Result<(AllocatedNum<E::Scalar>, Vec<AllocatedNum<E::Scalar>>), SynthesisError> {
    let mut e = self.inputs.claim;
    let mut r = Vec::new();

    // verify that there is a univariate polynomial for each round
    let uni_polys_len =
      AllocatedNum::alloc(cs.namespace(|| format!("{label} uni_polys.len()")), || {
        Ok(E::Scalar::from(
          self.inputs.sc_proof.compressed_polys.len() as u64
        ))
      })?;

    cs.enforce(
      || format!("{label} polys.len() == num_rounds"),
      |lc| lc + uni_polys_len.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + self.inputs.num_rounds.get_variable(),
    );

    // Rounds
    for (i, _) in self.inputs.sc_proof.compressed_polys.iter().enumerate() {
      let poly = self.inputs.sc_proof.compressed_polys[i]
        .decompress(&e.get_value().unwrap_or(E::Scalar::ZERO));
      let poly = AllocatedUniPoly::alloc(cs.namespace(|| format!("uni_poly_{i}_{label}")), &poly)?;
      // verify degree bound
      let poly_degree = poly.degree(cs.namespace(|| format!("uni_poly degree_{i}_{label}")))?;
      cs.enforce(
        || format!("i: {i} poly_degree == degree_bound {label}"),
        |lc| lc + poly_degree.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + self.inputs.degree_bound.get_variable(),
      );

      // verify evaluation at zero and one == claim
      let eval_at_zero = poly.eval_at_zero(cs.namespace(|| format!("eval at zero_{i}_{label}")))?;
      let eval_at_one = poly.eval_at_one(cs.namespace(|| format!("eval at one_{i}_{label}")))?;
      cs.enforce(
        || format!("i: {i} e == uni_poly(0) + uni_poly(1) {label}"),
        |lc| lc + eval_at_zero.get_variable() + eval_at_one.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + e.get_variable(),
      );

      // derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(format!("sumcheck r_{i}_{label}"), sponge, cs)?;

      r.push(r_i.clone());

      // evaluate the claimed degree-ell polynomial at r_i
      e = poly.evaluate(cs.namespace(|| format!("eval_{i}_{label}")), &r_i)?;
    }

    Ok((e, r))
  }
}

/// Data structure used to verify multiple sumcheck claims in a constraint system.
pub struct SCVerifyBatchedGadget<E: Engine> {
  sc_proof: SumcheckProof<E>,
}

impl<E: Engine> SCVerifyBatchedGadget<E> {
  /// Instantiate a new Sumcheck verify circuit.
  pub fn new(sc_proof: SumcheckProof<E>) -> Self {
    Self { sc_proof }
  }
}

impl<E: Engine> SCVerifyBatchedGadget<E> {
  pub(crate) fn verify_batched<'a, CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut Namespace<'a, E::Scalar, CS>,
    claims: &[AllocatedNum<E::Scalar>],
    num_rounds: &[usize],
    coeffs: &[AllocatedNum<E::Scalar>],
    degree_bound: &AllocatedNum<E::Scalar>,
    transcript: &mut PoseidonTranscriptCircuit<E>,
    sponge: &mut SpongeCircuit<'a, E::Scalar, U24, CS>,
    label: &str,
  ) -> Result<(AllocatedNum<E::Scalar>, Vec<AllocatedNum<E::Scalar>>), SynthesisError> {
    let num_instances =
      AllocatedNum::alloc(cs.namespace(|| format!("{label} num_instances")), || {
        Ok(E::Scalar::from(claims.len() as u64))
      })?;

    // assert_eq!(num_rounds.len(), num_instances);
    let num_rounds_len =
      AllocatedNum::alloc(cs.namespace(|| format!("{label} num_rounds.len()")), || {
        Ok(E::Scalar::from(num_rounds.len() as u64))
      })?;

    cs.enforce(
      || format!("{label} num_rounds.len() == num_instances"),
      |lc| lc + num_rounds_len.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + num_instances.get_variable(),
    );

    // assert_eq!(coeffs.len(), num_instances);
    let coeffs_len = AllocatedNum::alloc(cs.namespace(|| format!("{label} coeffs.len()")), || {
      Ok(E::Scalar::from(coeffs.len() as u64))
    })?;

    cs.enforce(
      || format!("{label} coeffs.len() == num_instances"),
      |lc| lc + coeffs_len.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + num_instances.get_variable(),
    );

    // n = maxᵢ{nᵢ}
    let num_rounds_max_int = num_rounds.iter().max().unwrap();
    let num_rounds_max =
      AllocatedNum::alloc(cs.namespace(|| format!("{label} num_rounds_max")), || {
        Ok(E::Scalar::from(*num_rounds_max_int as u64))
      })?;

    // Random linear combination of claims,
    // where each claim is scaled by 2^{n-nᵢ} to account for the padding.
    //
    // claim = ∑ᵢ coeffᵢ⋅2^{n-nᵢ}⋅cᵢ

    // Get scaled claims
    let claim = {
      let scaled_claims = claims
        .iter()
        .zip_eq(num_rounds.iter())
        .enumerate()
        .map(|(i, (claim, num_rounds))| {
          let scaling_factor = 1 << (num_rounds_max_int - num_rounds);
          AllocatedNum::alloc(cs.namespace(|| format!("scaled_claim_{i}_{label}")), || {
            Ok(E::Scalar::from(scaling_factor as u64) * claim.get_value().expect("claim value"))
          })
        })
        .collect::<Result<Vec<_>, _>>()?;

      let coeff_mul_scaled_claims = scaled_claims
        .iter()
        .zip_eq(coeffs.iter())
        .enumerate()
        .map(|(i, (claim, coeff))| {
          claim.mul(
            cs.namespace(|| format!("claim_mul_coeff_{i}_{label}")),
            coeff,
          )
        })
        .collect::<Result<Vec<_>, _>>()?;

      let mut claim = alloc_zero(cs.namespace(|| format!("claim_{label}")));

      for (i, claim_i) in coeff_mul_scaled_claims.iter().enumerate() {
        claim = claim
          .add(cs.namespace(|| format!("claim_add_{i}_{label}")), &claim_i)
          .expect("sum up claims");
      }

      claim
    };

    // Setup Sumcheck Verify Circuit
    let sc_verify_gadget_inputs = SCVerifyGadgetInputs {
      claim: claim,
      num_rounds: num_rounds_max,
      degree_bound: degree_bound.clone(),
      sc_proof: self.sc_proof.clone(),
    };

    // Run sumcheck verify gadget
    let sc_verify_gadget = SCVerifyGadget::new(sc_verify_gadget_inputs);
    sc_verify_gadget.verify(cs, transcript, sponge, &format!("{label} sc verify"))
  }
}
