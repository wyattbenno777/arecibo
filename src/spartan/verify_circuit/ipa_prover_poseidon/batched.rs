//! This module implements `BatchedRelaxedR1CSSNARKTrait` using Spartan that is generic over the polynomial commitment
//! and evaluation argument (i.e., a PCS) This version of Spartan does not use preprocessing so the verifier keeps the
//! entire description of R1CS matrices. This is essentially optimal for the verifier when using an IPA-based polynomial
//! commitment scheme. This batched implementation batches the outer and inner sumchecks of the Spartan SNARK.

use super::ipa_pc;
use crate::spartan::verify_circuit::gadgets::poseidon_transcript::PoseidonTranscript;
use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness, SparseMatrix},
  spartan::{
    polys::{multilinear::SparsePolynomial, power::PowPolynomial},
    snark::batch_eval_verify_poseidon,
  },
  traits::{
    snark::{BatchedRelaxedR1CSSNARKTrait, DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine,
  },
  zip_with, CommitmentKey,
};
use crate::{
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
  spartan::{
    compute_eval_table_sparse,
    math::Math,
    polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
    powers,
    snark::batch_eval_reduce_poseidon,
    sumcheck::SumcheckProof,
    PolyEvalInstance, PolyEvalWitness,
  },
};

use core::slice;
use ff::Field;
use generic_array::typenum::U24;
use itertools::Itertools;
use once_cell::sync::OnceCell;

use poseidon_sponge::sponge::vanilla::Sponge;
use poseidon_sponge::sponge::vanilla::SpongeTrait;
use poseidon_sponge::Strength;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{iter, sync::Arc};

/// A succinct proof of knowledge of a witness to a batch of relaxed R1CS instances
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchedRelaxedR1CSSNARK<E: Engine> {
  pub(crate) sc_proof_outer: SumcheckProof<E>,
  // Claims ([Azᵢ(τᵢ)], [Bzᵢ(τᵢ)], [Czᵢ(τᵢ)])
  pub(crate) claims_outer: Vec<(E::Scalar, E::Scalar, E::Scalar)>,
  // [Eᵢ(r_x)]
  pub(crate) evals_E: Vec<E::Scalar>,
  pub(crate) sc_proof_inner: SumcheckProof<E>,
  // [Wᵢ(r_y[1..])]
  pub(crate) evals_W: Vec<E::Scalar>,
  pub(crate) sc_proof_batch: SumcheckProof<E>,
  // [Wᵢ(r_z), Eᵢ(r_z)]
  pub(crate) evals_batch: Vec<E::Scalar>,
  pub(crate) eval_arg: ipa_pc::InnerProductArgument<E>,
}

/// A type that represents the prover's key
#[derive(Debug)]
pub struct ProverKey<E: Engine> {
  pk_ee: ipa_pc::ProverKey<E>,
  vk_digest: E::Scalar, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Debug, Serialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine> {
  pub(crate) vk_ee: ipa_pc::VerifierKey<E>,
  pub(crate) S: Vec<R1CSShape<E>>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine> VerifierKey<E> {
  fn new(shapes: Vec<R1CSShape<E>>, vk_ee: ipa_pc::VerifierKey<E>) -> Self {
    Self {
      vk_ee,
      S: shapes,
      digest: OnceCell::new(),
    }
  }
}

impl<E: Engine> SimpleDigestible for VerifierKey<E> {}

impl<E: Engine> DigestHelperTrait<E> for VerifierKey<E> {
  /// Returns the digest of the verifier's key.
  fn digest(&self) -> E::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<E::Scalar, _>::new(self);
        dc.digest()
      })
      .cloned()
      .expect("Failure to retrieve digest!")
  }
}

impl<E: Engine> BatchedRelaxedR1CSSNARKTrait<E> for BatchedRelaxedR1CSSNARK<E>
where
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  type ProverKey = ProverKey<E>;

  type VerifierKey = VerifierKey<E>;

  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: Vec<&R1CSShape<E>>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    let (pk_ee, vk_ee) = ipa_pc::EvaluationEngine::setup(ck);

    let S = S.iter().map(|s| s.pad()).collect();

    let vk = VerifierKey::new(S, vk_ee);

    let pk = ProverKey {
      pk_ee,
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: Vec<&R1CSShape<E>>,
    U: &[RelaxedR1CSInstance<E>],
    W: &[RelaxedR1CSWitness<E>],
  ) -> Result<Self, NovaError> {
    let num_instances = U.len();
    // Pad shapes and ensure their sizes are correct
    let S = S.iter().map(|s| s.pad()).collect::<Vec<_>>();

    // Pad (W,E) for each instance
    let W = zip_with!(iter, (W, S), |w, s| w.pad(s)).collect::<Vec<RelaxedR1CSWitness<E>>>();

    let constants = Sponge::<E::Scalar, U24>::api_constants(Strength::Standard);
    let mut transcript = PoseidonTranscript::<E>::new(&constants);

    transcript.absorb(pk.vk_digest);

    // if num_instances > 1 {
    //   let num_instances_field = E::Scalar::from(num_instances as u64);
    //   transcript.absorb(b"n", &num_instances_field);
    // }
    // transcript.absorb(b"U", &U);

    let (polys_W, polys_E): (Vec<_>, Vec<_>) = W.into_iter().map(|w| (w.W, w.E)).unzip();

    // Append public inputs to W: Z = [W, u, X]
    let polys_Z = zip_with!(iter, (polys_W, U), |w, u| [
      w.clone(),
      vec![u.u],
      u.X.clone()
    ]
    .concat())
    .collect::<Vec<Vec<_>>>();

    let (num_rounds_x, num_rounds_y): (Vec<_>, Vec<_>) = S
      .iter()
      .map(|s| (s.num_cons.log_2(), s.num_vars.log_2() + 1))
      .unzip();
    let num_rounds_x_max = *num_rounds_x.iter().max().unwrap();
    let num_rounds_y_max = *num_rounds_y.iter().max().unwrap();

    // Generate tau polynomial corresponding to eq(τ, τ², τ⁴ , …)
    // for a random challenge τ
    let tau = transcript.squeeze(String::from("tau"))?;
    let all_taus = PowPolynomial::squares(&tau, num_rounds_x_max);

    let polys_tau = num_rounds_x
      .iter()
      .map(|&num_rounds_x| PowPolynomial::evals_with_powers(&all_taus, num_rounds_x))
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();

    // Compute MLEs of Az, Bz, Cz, uCz + E
    let (polys_Az, polys_Bz, polys_Cz): (Vec<_>, Vec<_>, Vec<_>) =
      zip_with!(par_iter, (S, polys_Z), |s, poly_Z| {
        let (poly_Az, poly_Bz, poly_Cz) = s.multiply_vec(poly_Z)?;
        Ok((poly_Az, poly_Bz, poly_Cz))
      })
      .collect::<Result<Vec<_>, NovaError>>()?
      .into_iter()
      .multiunzip();

    let polys_uCz_E = zip_with!(par_iter, (U, polys_E, polys_Cz), |u, poly_E, poly_Cz| {
      zip_with!(par_iter, (poly_Cz, poly_E), |cz, e| u.u * cz + e).collect::<Vec<E::Scalar>>()
    })
    .collect::<Vec<_>>();

    let comb_func_outer =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    // Sample challenge for random linear-combination of outer claims
    let outer_r = transcript.squeeze(String::from("outer_r"))?;
    let outer_r_powers = powers(&outer_r, num_instances);

    // Verify outer sumcheck: Az * Bz - uCz_E for each instance
    let (sc_proof_outer, r_x, claims_outer) =
      SumcheckProof::prove_cubic_with_additive_term_batch_poseidon(
        &vec![E::Scalar::ZERO; num_instances],
        &num_rounds_x,
        polys_tau,
        polys_Az
          .into_iter()
          .map(MultilinearPolynomial::new)
          .collect(),
        polys_Bz
          .into_iter()
          .map(MultilinearPolynomial::new)
          .collect(),
        polys_uCz_E
          .into_iter()
          .map(MultilinearPolynomial::new)
          .collect(),
        &outer_r_powers,
        comb_func_outer,
        &mut transcript,
      )?;

    let r_x = num_rounds_x
      .iter()
      .map(|&num_rounds| r_x[(num_rounds_x_max - num_rounds)..].to_vec())
      .collect::<Vec<_>>();

    // Extract evaluations of Az, Bz from Sumcheck and Cz, E at r_x
    let (evals_Az_Bz_Cz, evals_E): (Vec<_>, Vec<_>) = zip_with!(
      par_iter,
      (claims_outer[1], claims_outer[2], polys_Cz, polys_E, r_x),
      |eval_Az, eval_Bz, poly_Cz, poly_E, r_x| {
        let (eval_Cz, eval_E) = rayon::join(
          || MultilinearPolynomial::evaluate_with(poly_Cz, r_x),
          || MultilinearPolynomial::evaluate_with(poly_E, r_x),
        );
        ((*eval_Az, *eval_Bz, eval_Cz), eval_E)
      }
    )
    .unzip();

    // evals_Az_Bz_Cz.iter().zip_eq(evals_E.iter()).for_each(
    //   |(&(eval_Az, eval_Bz, eval_Cz), &eval_E)| {
    //     transcript.absorb(
    //       b"claims_outer",
    //       &[eval_Az, eval_Bz, eval_Cz, eval_E].as_slice(),
    //     )
    //   },
    // );

    let inner_r = transcript.squeeze(String::from("in_r"))?;
    let inner_r_square = inner_r.square();
    let inner_r_cube = inner_r_square * inner_r;
    let inner_r_powers = powers(&inner_r_cube, num_instances);

    let claims_inner_joint = evals_Az_Bz_Cz
      .iter()
      .map(|(eval_Az, eval_Bz, eval_Cz)| *eval_Az + inner_r * eval_Bz + inner_r_square * eval_Cz)
      .collect::<Vec<_>>();

    let polys_ABCs = {
      let inner = |M_evals_As: Vec<E::Scalar>,
                   M_evals_Bs: Vec<E::Scalar>,
                   M_evals_Cs: Vec<E::Scalar>|
       -> Vec<E::Scalar> {
        zip_with!(
          into_par_iter,
          (M_evals_As, M_evals_Bs, M_evals_Cs),
          |eval_A, eval_B, eval_C| eval_A + inner_r * eval_B + inner_r_square * eval_C
        )
        .collect::<Vec<_>>()
      };

      zip_with!(par_iter, (S, r_x), |s, r_x| {
        let evals_rx = EqPolynomial::evals_from_points(r_x);
        let (eval_A, eval_B, eval_C) = compute_eval_table_sparse(s, &evals_rx);
        MultilinearPolynomial::new(inner(eval_A, eval_B, eval_C))
      })
      .collect::<Vec<_>>()
    };

    let polys_Z = polys_Z
      .into_iter()
      .zip_eq(num_rounds_y.iter())
      .map(|(mut z, &num_rounds_y)| {
        z.resize(1 << num_rounds_y, E::Scalar::ZERO);
        MultilinearPolynomial::new(z)
      })
      .collect::<Vec<_>>();

    let comb_func = |poly_A_comp: &E::Scalar, poly_B_comp: &E::Scalar| -> E::Scalar {
      *poly_A_comp * *poly_B_comp
    };

    let (sc_proof_inner, r_y, _claims_inner): (SumcheckProof<E>, Vec<E::Scalar>, (Vec<_>, Vec<_>)) =
      SumcheckProof::prove_quad_batch_poseidon(
        &claims_inner_joint,
        &num_rounds_y,
        polys_ABCs,
        polys_Z,
        &inner_r_powers,
        comb_func,
        &mut transcript,
      )?;

    let r_y = num_rounds_y
      .iter()
      .map(|num_rounds| {
        let (_, r_y_hi) = r_y.split_at(num_rounds_y_max - num_rounds);
        r_y_hi
      })
      .collect::<Vec<_>>();

    let evals_W = zip_with!(par_iter, (polys_W, r_y), |poly, r_y| {
      MultilinearPolynomial::evaluate_with(poly, &r_y[1..])
    })
    .collect::<Vec<_>>();

    // Create evaluation instances for W(r_y[1..]) and E(r_x)
    let (w_vec, u_vec) = {
      let mut w_vec = Vec::with_capacity(2 * num_instances);
      let mut u_vec = Vec::with_capacity(2 * num_instances);
      w_vec.extend(polys_W.into_iter().map(|poly| PolyEvalWitness { p: poly }));
      u_vec.extend(zip_with!(iter, (evals_W, U, r_y), |eval, u, r_y| {
        PolyEvalInstance {
          c: u.comm_W,
          x: r_y[1..].to_vec(),
          e: *eval,
        }
      }));

      w_vec.extend(polys_E.into_iter().map(|poly| PolyEvalWitness { p: poly }));
      u_vec.extend(zip_with!(
        (evals_E.iter(), U.iter(), r_x),
        |eval_E, u, r_x| PolyEvalInstance {
          c: u.comm_E,
          x: r_x,
          e: *eval_E,
        }
      ));
      (w_vec, u_vec)
    };

    let (batched_u, batched_w, sc_proof_batch, claims_batch_left) =
      batch_eval_reduce_poseidon(u_vec, &w_vec, &mut transcript)?;

    let eval_arg = ipa_pc::EvaluationEngine::prove(
      ck,
      &pk.pk_ee,
      &batched_u.c,
      &batched_w.p,
      &batched_u.x,
      &batched_u.e,
    )?;

    Ok(Self {
      sc_proof_outer,
      claims_outer: evals_Az_Bz_Cz,
      evals_E,
      sc_proof_inner,
      evals_W,
      sc_proof_batch,
      evals_batch: claims_batch_left,
      eval_arg,
    })
  }

  fn verify(&self, vk: &Self::VerifierKey, U: &[RelaxedR1CSInstance<E>]) -> Result<(), NovaError> {
    let _num_instances = U.len();

    let constants = Sponge::<E::Scalar, U24>::api_constants(Strength::Standard);
    let mut transcript = PoseidonTranscript::<E>::new(&constants);

    transcript.absorb(vk.digest());
    // if num_instances > 1 {
    //   let num_instances_field = E::Scalar::from(num_instances as u64);
    //   transcript.absorb(b"n", &num_instances_field);
    // }
    // transcript.absorb(b"U", &U);

    let num_instances = U.len();

    let (num_rounds_x, num_rounds_y): (Vec<_>, Vec<_>) = vk
      .S
      .iter()
      .map(|s| (s.num_cons.log_2(), s.num_vars.log_2() + 1))
      .unzip();
    let num_rounds_x_max = *num_rounds_x.iter().max().unwrap();
    let num_rounds_y_max = *num_rounds_y.iter().max().unwrap();

    // Define τ polynomials of the appropriate size for each instance
    let tau = transcript.squeeze(String::from("tau"))?;
    let all_taus = PowPolynomial::squares(&tau, num_rounds_x_max);

    let polys_tau = num_rounds_x
      .iter()
      .map(|&num_rounds_x| PowPolynomial::evals_with_powers(&all_taus, num_rounds_x))
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();

    // Sample challenge for random linear-combination of outer claims
    let outer_r = transcript.squeeze(String::from("out_r"))?;
    let outer_r_powers = powers(&outer_r, num_instances);

    let (claim_outer_final, r_x) = self.sc_proof_outer.verify_batch_poseidon(
      &vec![E::Scalar::ZERO; num_instances],
      &num_rounds_x,
      &outer_r_powers,
      3,
      &mut transcript,
    )?;

    // Since each instance has a different number of rounds, the Sumcheck
    // prover skips the first num_rounds_x_max - num_rounds_x rounds.
    // The evaluation point for each instance is therefore r_x[num_rounds_x_max - num_rounds_x..]
    let r_x = num_rounds_x
      .iter()
      .map(|num_rounds| r_x[(num_rounds_x_max - num_rounds)..].to_vec())
      .collect::<Vec<_>>();

    // Extract evaluations into a vector [(Azᵢ, Bzᵢ, Czᵢ, Eᵢ)]
    let ABCE_evals = || self.claims_outer.iter().zip_eq(self.evals_E.iter());

    // Add evaluations of Az, Bz, Cz, E to transcript
    // for ((claim_Az, claim_Bz, claim_Cz), eval_E) in ABCE_evals() {
    //   transcript.absorb(
    //     b"claims_outer",
    //     &[*claim_Az, *claim_Bz, *claim_Cz, *eval_E].as_slice(),
    //   )
    // }

    let chis_r_x = r_x
      .par_iter()
      .map(|r_x| EqPolynomial::evals_from_points(r_x))
      .collect::<Vec<_>>();

    // Evaluate τ(rₓ) for each instance
    let evals_tau = zip_with!(iter, (polys_tau, chis_r_x), |poly_tau, er_x| {
      MultilinearPolynomial::evaluate_with_chis(poly_tau.evaluations(), er_x)
    });

    // Compute expected claim for all instances ∑ᵢ rⁱ⋅τ(rₓ)⋅(Azᵢ⋅Bzᵢ − uᵢ⋅Czᵢ − Eᵢ)
    let claim_outer_final_expected = zip_with!(
      (ABCE_evals(), U.iter(), evals_tau, outer_r_powers.iter()),
      |ABCE_eval, u, eval_tau, r| {
        let ((claim_Az, claim_Bz, claim_Cz), eval_E) = ABCE_eval;
        *r * eval_tau * (*claim_Az * claim_Bz - u.u * claim_Cz - eval_E)
      }
    )
    .sum::<E::Scalar>();

    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let inner_r = transcript.squeeze(String::from("in_r"))?;
    let inner_r_square = inner_r.square();
    let inner_r_cube = inner_r_square * inner_r;
    let inner_r_powers = powers(&inner_r_cube, num_instances);

    // Compute inner claims Mzᵢ = (Azᵢ + r⋅Bzᵢ + r²⋅Czᵢ),
    // which are batched by Sumcheck into one claim:  ∑ᵢ r³ⁱ⋅Mzᵢ
    let claims_inner = self
      .claims_outer
      .iter()
      .map(|(claim_Az, claim_Bz, claim_Cz)| {
        *claim_Az + inner_r * claim_Bz + inner_r_square * claim_Cz
      })
      .collect::<Vec<_>>();

    let (claim_inner_final, r_y) = self.sc_proof_inner.verify_batch_poseidon(
      &claims_inner,
      &num_rounds_y,
      &inner_r_powers,
      2,
      &mut transcript,
    )?;
    let r_y: Vec<Vec<E::Scalar>> = num_rounds_y
      .iter()
      .map(|num_rounds| r_y[(num_rounds_y_max - num_rounds)..].to_vec())
      .collect();

    // Compute evaluations of Zᵢ = [Wᵢ, uᵢ, Xᵢ] at r_y
    // Zᵢ(r_y) = (1−r_y[0])⋅W(r_y[1..]) + r_y[0]⋅MLE([uᵢ, Xᵢ])(r_y[1..])
    let evals_Z = zip_with!(iter, (self.evals_W, U, r_y), |eval_W, U, r_y| {
      let eval_X = {
        // constant term
        let poly_X = iter::once(U.u).chain(U.X.iter().cloned()).collect();
        SparsePolynomial::new(r_y.len() - 1, poly_X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * eval_W + r_y[0] * eval_X
    })
    .collect::<Vec<_>>();

    // compute evaluations of R1CS matrices M(r_x, r_y) = eq(r_y)ᵀ⋅M⋅eq(r_x)
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          chi_r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
          M.par_iter_rows()
            .enumerate()
            .map(|(row_idx, row)| {
              M.get_row(row)
                .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
                .sum::<E::Scalar>()
            })
            .sum()
        };

      let T_x = chi_r_x;
      let T_y = EqPolynomial::evals_from_points(r_y);

      M_vec
        .par_iter()
        .map(|&M_vec| evaluate_with_table(M_vec, T_x, &T_y))
        .collect()
    };

    // Compute inner claim ∑ᵢ r³ⁱ⋅(Aᵢ(r_x, r_y) + r⋅Bᵢ(r_x, r_y) + r²⋅Cᵢ(r_x, r_y))⋅Zᵢ(r_y)
    let claim_inner_final_expected = zip_with!(
      iter,
      (vk.S, chis_r_x, r_y, evals_Z, inner_r_powers),
      |S, r_x, r_y, eval_Z, r_i| {
        let evals = multi_evaluate(&[&S.A, &S.B, &S.C], r_x, r_y);
        let eval = evals[0] + inner_r * evals[1] + inner_r_square * evals[2];
        eval * r_i * eval_Z
      }
    )
    .sum::<E::Scalar>();

    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // Create evaluation instances for W(r_y[1..]) and E(r_x)
    let u_vec = {
      let mut u_vec = Vec::with_capacity(2 * num_instances);
      u_vec.extend(zip_with!(iter, (self.evals_W, U, r_y), |eval, u, r_y| {
        PolyEvalInstance {
          c: u.comm_W,
          x: r_y[1..].to_vec(),
          e: *eval,
        }
      }));

      u_vec.extend(zip_with!(iter, (self.evals_E, U, r_x), |eval, u, r_x| {
        PolyEvalInstance {
          c: u.comm_E,
          x: r_x.to_vec(),
          e: *eval,
        }
      }));
      u_vec
    };

    let batched_u = batch_eval_verify_poseidon(
      u_vec,
      &mut transcript,
      &self.sc_proof_batch,
      &self.evals_batch,
    )?;

    // verify
    ipa_pc::EvaluationEngine::verify(
      &vk.vk_ee,
      &batched_u.c,
      &batched_u.x,
      &batched_u.e,
      &self.eval_arg,
    )?;

    Ok(())
  }
}

impl<E: Engine> BatchedRelaxedR1CSSNARK<E>
where
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  pub(crate) fn verify_execution_trace(
    &self,
    vk: &VerifierKey<E>,
    U: &[RelaxedR1CSInstance<E>],
  ) -> Result<PolyEvalInstance<E>, NovaError> {
    let _num_instances = U.len();

    let constants = Sponge::<E::Scalar, U24>::api_constants(Strength::Standard);
    let mut transcript = PoseidonTranscript::<E>::new(&constants);

    transcript.absorb(vk.digest());
    // if num_instances > 1 {
    //   let num_instances_field = E::Scalar::from(num_instances as u64);
    //   transcript.absorb(b"n", &num_instances_field);
    // }
    // transcript.absorb(b"U", &U);

    let num_instances = U.len();

    let (num_rounds_x, num_rounds_y): (Vec<_>, Vec<_>) = vk
      .S
      .iter()
      .map(|s| (s.num_cons.log_2(), s.num_vars.log_2() + 1))
      .unzip();
    let num_rounds_x_max = *num_rounds_x.iter().max().unwrap();
    let num_rounds_y_max = *num_rounds_y.iter().max().unwrap();

    // Define τ polynomials of the appropriate size for each instance
    let tau = transcript.squeeze(String::from("tau"))?;
    let all_taus = PowPolynomial::squares(&tau, num_rounds_x_max);

    let polys_tau = num_rounds_x
      .iter()
      .map(|&num_rounds_x| PowPolynomial::evals_with_powers(&all_taus, num_rounds_x))
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();

    // Sample challenge for random linear-combination of outer claims
    let outer_r = transcript.squeeze(String::from("out_r"))?;
    let outer_r_powers = powers(&outer_r, num_instances);

    let (claim_outer_final, r_x) = self.sc_proof_outer.verify_batch_poseidon(
      &vec![E::Scalar::ZERO; num_instances],
      &num_rounds_x,
      &outer_r_powers,
      3,
      &mut transcript,
    )?;

    // Since each instance has a different number of rounds, the Sumcheck
    // prover skips the first num_rounds_x_max - num_rounds_x rounds.
    // The evaluation point for each instance is therefore r_x[num_rounds_x_max - num_rounds_x..]
    let r_x = num_rounds_x
      .iter()
      .map(|num_rounds| r_x[(num_rounds_x_max - num_rounds)..].to_vec())
      .collect::<Vec<_>>();

    // Extract evaluations into a vector [(Azᵢ, Bzᵢ, Czᵢ, Eᵢ)]
    let ABCE_evals = || self.claims_outer.iter().zip_eq(self.evals_E.iter());

    // Add evaluations of Az, Bz, Cz, E to transcript
    // for ((claim_Az, claim_Bz, claim_Cz), eval_E) in ABCE_evals() {
    //   transcript.absorb(
    //     b"claims_outer",
    //     &[*claim_Az, *claim_Bz, *claim_Cz, *eval_E].as_slice(),
    //   )
    // }

    let chis_r_x = r_x
      .par_iter()
      .map(|r_x| EqPolynomial::evals_from_points(r_x))
      .collect::<Vec<_>>();

    // Evaluate τ(rₓ) for each instance
    let evals_tau = zip_with!(iter, (polys_tau, chis_r_x), |poly_tau, er_x| {
      MultilinearPolynomial::evaluate_with_chis(poly_tau.evaluations(), er_x)
    });

    // Compute expected claim for all instances ∑ᵢ rⁱ⋅τ(rₓ)⋅(Azᵢ⋅Bzᵢ − uᵢ⋅Czᵢ − Eᵢ)
    let claim_outer_final_expected = zip_with!(
      (ABCE_evals(), U.iter(), evals_tau, outer_r_powers.iter()),
      |ABCE_eval, u, eval_tau, r| {
        let ((claim_Az, claim_Bz, claim_Cz), eval_E) = ABCE_eval;
        *r * eval_tau * (*claim_Az * claim_Bz - u.u * claim_Cz - eval_E)
      }
    )
    .sum::<E::Scalar>();

    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let inner_r = transcript.squeeze(String::from("in_r"))?;
    let inner_r_square = inner_r.square();
    let inner_r_cube = inner_r_square * inner_r;
    let inner_r_powers = powers(&inner_r_cube, num_instances);

    // Compute inner claims Mzᵢ = (Azᵢ + r⋅Bzᵢ + r²⋅Czᵢ),
    // which are batched by Sumcheck into one claim:  ∑ᵢ r³ⁱ⋅Mzᵢ
    let claims_inner = self
      .claims_outer
      .iter()
      .map(|(claim_Az, claim_Bz, claim_Cz)| {
        *claim_Az + inner_r * claim_Bz + inner_r_square * claim_Cz
      })
      .collect::<Vec<_>>();

    let (claim_inner_final, r_y) = self.sc_proof_inner.verify_batch_poseidon(
      &claims_inner,
      &num_rounds_y,
      &inner_r_powers,
      2,
      &mut transcript,
    )?;
    let r_y: Vec<Vec<E::Scalar>> = num_rounds_y
      .iter()
      .map(|num_rounds| r_y[(num_rounds_y_max - num_rounds)..].to_vec())
      .collect();

    // Compute evaluations of Zᵢ = [Wᵢ, uᵢ, Xᵢ] at r_y
    // Zᵢ(r_y) = (1−r_y[0])⋅W(r_y[1..]) + r_y[0]⋅MLE([uᵢ, Xᵢ])(r_y[1..])
    let evals_Z = zip_with!(iter, (self.evals_W, U, r_y), |eval_W, U, r_y| {
      let eval_X = {
        // constant term
        let poly_X = iter::once(U.u).chain(U.X.iter().cloned()).collect();
        SparsePolynomial::new(r_y.len() - 1, poly_X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * eval_W + r_y[0] * eval_X
    })
    .collect::<Vec<_>>();

    // compute evaluations of R1CS matrices M(r_x, r_y) = eq(r_y)ᵀ⋅M⋅eq(r_x)
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          chi_r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
          M.par_iter_rows()
            .enumerate()
            .map(|(row_idx, row)| {
              M.get_row(row)
                .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
                .sum::<E::Scalar>()
            })
            .sum()
        };

      let T_x = chi_r_x;
      let T_y = EqPolynomial::evals_from_points(r_y);

      M_vec
        .par_iter()
        .map(|&M_vec| evaluate_with_table(M_vec, T_x, &T_y))
        .collect()
    };

    // Compute inner claim ∑ᵢ r³ⁱ⋅(Aᵢ(r_x, r_y) + r⋅Bᵢ(r_x, r_y) + r²⋅Cᵢ(r_x, r_y))⋅Zᵢ(r_y)
    let claim_inner_final_expected = zip_with!(
      iter,
      (vk.S, chis_r_x, r_y, evals_Z, inner_r_powers),
      |S, r_x, r_y, eval_Z, r_i| {
        let evals = multi_evaluate(&[&S.A, &S.B, &S.C], r_x, r_y);
        let eval = evals[0] + inner_r * evals[1] + inner_r_square * evals[2];
        eval * r_i * eval_Z
      }
    )
    .sum::<E::Scalar>();

    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // Create evaluation instances for W(r_y[1..]) and E(r_x)
    let u_vec = {
      let mut u_vec = Vec::with_capacity(2 * num_instances);
      u_vec.extend(zip_with!(iter, (self.evals_W, U, r_y), |eval, u, r_y| {
        PolyEvalInstance {
          c: u.comm_W,
          x: r_y[1..].to_vec(),
          e: *eval,
        }
      }));

      u_vec.extend(zip_with!(iter, (self.evals_E, U, r_x), |eval, u, r_x| {
        PolyEvalInstance {
          c: u.comm_E,
          x: r_x.to_vec(),
          e: *eval,
        }
      }));
      u_vec
    };

    let batched_u = batch_eval_verify_poseidon(
      u_vec,
      &mut transcript,
      &self.sc_proof_batch,
      &self.evals_batch,
    )?;

    Ok(batched_u)
  }
}

impl<E: Engine> RelaxedR1CSSNARKTrait<E> for BatchedRelaxedR1CSSNARK<E>
where
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  type ProverKey = ProverKey<E>;

  type VerifierKey = VerifierKey<E>;

  fn ck_floor() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
    <Self as BatchedRelaxedR1CSSNARKTrait<E>>::ck_floor()
  }

  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: &R1CSShape<E>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    <Self as BatchedRelaxedR1CSSNARKTrait<E>>::setup(ck, vec![S])
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
  ) -> Result<Self, NovaError> {
    let slice_U = slice::from_ref(U);
    let slice_W = slice::from_ref(W);
    <Self as BatchedRelaxedR1CSSNARKTrait<E>>::prove(ck, pk, vec![S], slice_U, slice_W)
  }

  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<E>) -> Result<(), NovaError> {
    let slice = slice::from_ref(U);
    <Self as BatchedRelaxedR1CSSNARKTrait<E>>::verify(self, vk, slice)
  }
}
#[cfg(test)]
mod test {
  use super::ipa_pc;
  use super::BatchedRelaxedR1CSSNARK;
  use super::VerifierKey;
  use crate::gadgets::{alloc_negate, alloc_one, alloc_zero};
  use crate::spartan::verify_circuit::gadgets::poly::alloc_powers;
  use crate::spartan::verify_circuit::gadgets::poly::AllocMultilinearPolynomial;
  use crate::spartan::verify_circuit::gadgets::poly::AllocatedEqPolynomial;
  use crate::spartan::verify_circuit::gadgets::poly::AllocatedPowPolynomial;
  use crate::spartan::verify_circuit::gadgets::poly::AllocatedSparsePolynomial;
  use crate::spartan::verify_circuit::gadgets::poseidon_transcript::PoseidonTranscript;
  use crate::spartan::verify_circuit::gadgets::poseidon_transcript::PoseidonTranscriptCircuit;
  use crate::spartan::verify_circuit::gadgets::sumcheck::SCVerifyBatchedGadget;
  use crate::traits::Engine;
  use crate::NovaError;
  use crate::RelaxedR1CSInstance;
  use crate::{
    provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
    spartan::{
      math::Math,
      polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
      powers, PolyEvalInstance,
    },
  };
  use crate::{
    r1cs::SparseMatrix,
    spartan::{
      polys::{multilinear::SparsePolynomial, power::PowPolynomial},
      snark::batch_eval_verify_poseidon,
    },
    traits::snark::DigestHelperTrait,
    zip_with, CommitmentKey,
  };
  use bellpepper_core::num::AllocatedNum;
  use bellpepper_core::ConstraintSystem;
  use bellpepper_core::SynthesisError;

  use ff::Field;
  use generic_array::typenum::U24;
  use itertools::Itertools;
  use poseidon_sponge::sponge::circuit::SpongeCircuit;
  use poseidon_sponge::sponge::vanilla::Mode::Simplex;
  use poseidon_sponge::sponge::vanilla::Sponge;

  use poseidon_sponge::sponge::vanilla::SpongeTrait;
  use poseidon_sponge::Strength;
  use rayon::prelude::*;

  use std::iter;

  impl<E: Engine> BatchedRelaxedR1CSSNARK<E>
  where
    E::GE: DlogGroup,
    CommitmentKey<E>: CommitmentKeyExtTrait<E>,
  {
    pub fn verify_cs<CS: ConstraintSystem<E::Scalar>>(
      &self,
      mut cs: CS,
      vk: &VerifierKey<E>,
      U: &[RelaxedR1CSInstance<E>],
    ) -> Result<(), NovaError> {
      let alloc_one = alloc_one(cs.namespace(|| "one"));
      let alloc_zero = alloc_zero(cs.namespace(|| "zero"));
      let _num_instances = U.len();

      let constants = Sponge::<E::Scalar, U24>::api_constants(Strength::Standard);
      let mut transcript = PoseidonTranscript::<E>::new(&constants);

      // Transcript circuit
      let mut sponge = SpongeCircuit::<E::Scalar, U24, _>::new_with_constants(&constants, Simplex);
      let mut cs = cs.namespace(|| "ns");
      let mut transcript_circuit = PoseidonTranscriptCircuit::<E>::new(&mut sponge, &mut cs);

      transcript.absorb(vk.digest());

      let alloc_vk_digest = AllocatedNum::alloc(cs.namespace(|| "vk_digest"), || Ok(vk.digest()))?;
      transcript_circuit.absorb(alloc_vk_digest, &mut sponge, &mut cs);
      // if num_instances > 1 {
      //   let num_instances_field = E::Scalar::from(num_instances as u64);
      //   transcript.absorb(b"n", &num_instances_field);
      // }
      // transcript.absorb(b"U", &U);

      let num_instances = U.len();

      let (num_rounds_x, num_rounds_y): (Vec<_>, Vec<_>) = vk
        .S
        .iter()
        .map(|s| (s.num_cons.log_2(), s.num_vars.log_2() + 1))
        .unzip();
      let num_rounds_x_max = *num_rounds_x.iter().max().unwrap();
      let num_rounds_y_max = *num_rounds_y.iter().max().unwrap();

      // Define τ polynomials of the appropriate size for each instance
      let tau = transcript.squeeze(String::from("tau"))?;
      let alloc_tau = transcript_circuit.squeeze(String::from("tau"), &mut sponge, &mut cs)?;
      assert_eq!(tau, alloc_tau.get_value().unwrap());

      let all_taus = PowPolynomial::squares(&tau, num_rounds_x_max);
      let alloc_all_taus =
        AllocatedPowPolynomial::squares(cs.namespace(|| "all_taus"), &alloc_tau, num_rounds_x_max)?;

      for (alloc_tau, tau) in alloc_all_taus.iter().zip_eq(all_taus.iter()) {
        assert_eq!(alloc_tau.get_value().unwrap(), *tau);
      }

      let polys_tau = num_rounds_x
        .iter()
        .map(|&num_rounds_x| PowPolynomial::evals_with_powers(&all_taus, num_rounds_x))
        .map(MultilinearPolynomial::new)
        .collect::<Vec<_>>();

      let mut alloc_polys_tau = Vec::new();

      for (i, num_rounds_x) in num_rounds_x.iter().enumerate() {
        let alloc_pow_poly_tau = AllocatedPowPolynomial::evals_with_powers(
          cs.namespace(|| format!("polys_tau_{}", i)),
          &alloc_all_taus,
          *num_rounds_x,
        )?;

        let alloc_poly_tau = AllocMultilinearPolynomial::new(alloc_pow_poly_tau);
        alloc_polys_tau.push(alloc_poly_tau);
      }

      for (poly_tau, alloc_poly_tau) in polys_tau.iter().zip_eq(alloc_polys_tau.iter()) {
        for (eval, alloc_eval) in poly_tau.Z.iter().zip_eq(alloc_poly_tau.Z.iter()) {
          assert_eq!(eval, &alloc_eval.get_value().unwrap());
        }
      }

      // Sample challenge for random linear-combination of outer claims
      let outer_r = transcript.squeeze(String::from("out_r"))?;
      let alloc_outer_r =
        transcript_circuit.squeeze(String::from("outer_r"), &mut sponge, &mut cs)?;
      assert_eq!(outer_r, alloc_outer_r.get_value().unwrap());

      let outer_r_powers = powers(&outer_r, num_instances);
      let alloc_outer_r_powers = alloc_powers(
        cs.namespace(|| "outer_r_powers"),
        &alloc_outer_r,
        num_instances,
      )?;
      for (r, alloc_r) in outer_r_powers.iter().zip_eq(alloc_outer_r_powers.iter()) {
        assert_eq!(r, &alloc_r.get_value().unwrap());
      }

      let (claim_outer_final, r_x) = self.sc_proof_outer.verify_batch_poseidon(
        &vec![E::Scalar::ZERO; num_instances],
        &num_rounds_x,
        &outer_r_powers,
        3,
        &mut transcript,
      )?;

      let (alloc_claim_outer_final, alloc_r_x) = {
        let alloc_sc = SCVerifyBatchedGadget::new(self.sc_proof_outer.clone());
        let alloc_degree_bound = AllocatedNum::alloc(cs.namespace(|| "degree_bound"), || {
          Ok(E::Scalar::from(3_u64))
        })?;

        let alloc_claims = vec![alloc_zero.clone(); num_instances];

        alloc_sc.verify_batched(
          &mut cs,
          &alloc_claims,
          &num_rounds_x,
          &alloc_outer_r_powers,
          &alloc_degree_bound,
          &mut transcript_circuit,
          &mut sponge,
          "outer sc",
        )?
      };

      assert_eq!(
        claim_outer_final,
        alloc_claim_outer_final.get_value().unwrap()
      );
      assert_eq!(
        r_x,
        alloc_r_x
          .iter()
          .map(|r| r.get_value().unwrap())
          .collect::<Vec<_>>()
      );

      // Since each instance has a different number of rounds, the Sumcheck
      // prover skips the first num_rounds_x_max - num_rounds_x rounds.
      // The evaluation point for each instance is therefore r_x[num_rounds_x_max - num_rounds_x..]
      let r_x = num_rounds_x
        .iter()
        .map(|num_rounds| r_x[(num_rounds_x_max - num_rounds)..].to_vec())
        .collect::<Vec<_>>();

      let alloc_r_x = num_rounds_x
        .iter()
        .map(|num_rounds| alloc_r_x[(num_rounds_x_max - num_rounds)..].to_vec())
        .collect::<Vec<_>>();

      for (r_x, alloc_r_x) in r_x.iter().zip_eq(alloc_r_x.iter()) {
        for (r, alloc_r) in r_x.iter().zip_eq(alloc_r_x.iter()) {
          assert_eq!(r, &alloc_r.get_value().unwrap());
        }
      }

      // Extract evaluations into a vector [(Azᵢ, Bzᵢ, Czᵢ, Eᵢ)]
      let ABCE_evals = || self.claims_outer.iter().zip_eq(self.evals_E.iter());

      let alloc_ABCE_evals = ABCE_evals()
        .enumerate()
        .map(|(i, ((claim_Az, claim_Bz, claim_Cz), eval_E))| {
          (
            (
              AllocatedNum::alloc(cs.namespace(|| format!("claim_Az_{i}")), || Ok(*claim_Az))
                .expect("Allocated variable into cs"),
              AllocatedNum::alloc(cs.namespace(|| format!("claim_Bz_{i}")), || Ok(*claim_Bz))
                .expect("Allocated variable into cs"),
              AllocatedNum::alloc(cs.namespace(|| format!("claim_Cz_{i}")), || Ok(*claim_Cz))
                .expect("Allocated variable into cs"),
            ),
            AllocatedNum::alloc(cs.namespace(|| format!("eval_E_{i}")), || Ok(*eval_E))
              .expect("Allocated variable into cs"),
          )
        })
        .collect::<Vec<_>>();

      // Add evaluations of Az, Bz, Cz, E to transcript
      // for ((claim_Az, claim_Bz, claim_Cz), eval_E) in ABCE_evals() {
      //   transcript.absorb(
      //     b"claims_outer",
      //     &[*claim_Az, *claim_Bz, *claim_Cz, *eval_E].as_slice(),
      //   )
      // }

      let chis_r_x = r_x
        .par_iter()
        .map(|r_x| EqPolynomial::evals_from_points(r_x))
        .collect::<Vec<_>>();

      let alloc_chis_r_x = alloc_r_x
        .iter()
        .enumerate()
        .map(|(i, r_x)| {
          AllocatedEqPolynomial::evals_from_points(cs.namespace(|| format!("chis_r_x_{i}")), r_x)
        })
        .collect::<Result<Vec<_>, _>>()?;

      for (chi_r_x, alloc_chi_r_x) in chis_r_x.iter().zip_eq(alloc_chis_r_x.iter()) {
        for (chi, alloc_chi) in chi_r_x.iter().zip_eq(alloc_chi_r_x.iter()) {
          assert_eq!(chi, &alloc_chi.get_value().unwrap());
        }
      }

      // Evaluate τ(rₓ) for each instance
      let evals_tau = zip_with!(iter, (polys_tau, chis_r_x), |poly_tau, er_x| {
        MultilinearPolynomial::evaluate_with_chis(poly_tau.evaluations(), er_x)
      });

      let mut alloc_evals_tau = Vec::new();

      for (i, (alloc_poly_tau, alloc_er_x)) in alloc_polys_tau
        .iter()
        .zip_eq(alloc_chis_r_x.iter())
        .enumerate()
      {
        let alloc_eval_tau = AllocMultilinearPolynomial::evaluate_with_chis(
          cs.namespace(|| format!("eval_tau_{}", i)),
          alloc_poly_tau.evaluations(),
          &alloc_er_x,
        )?;

        alloc_evals_tau.push(alloc_eval_tau);
      }

      // Compute expected claim for all instances ∑ᵢ rⁱ⋅τ(rₓ)⋅(Azᵢ⋅Bzᵢ − uᵢ⋅Czᵢ − Eᵢ)
      let claim_outer_final_expected = zip_with!(
        (ABCE_evals(), U.iter(), evals_tau, outer_r_powers.iter()),
        |ABCE_eval, u, eval_tau, r| {
          let ((claim_Az, claim_Bz, claim_Cz), eval_E) = ABCE_eval;
          *r * eval_tau * (*claim_Az * claim_Bz - u.u * claim_Cz - eval_E)
        }
      )
      .sum::<E::Scalar>();

      let alloc_claim_outer_final_expected = {
        let mut final_claim = alloc_zero.clone();
        for (i, (((alloc_ABCE_eval, u), alloc_eval_tau), alloc_r)) in alloc_ABCE_evals
          .iter()
          .zip_eq(U.iter())
          .zip_eq(alloc_evals_tau.iter())
          .zip_eq(alloc_outer_r_powers.iter())
          .enumerate()
        {
          let ((alloc_claim_Az, alloc_claim_Bz, alloc_claim_Cz), alloc_eval_E) = alloc_ABCE_eval;
          let alloc_U_u = AllocatedNum::alloc(cs.namespace(|| format!("U_u_{}", i)), || Ok(u.u))?;

          // claim_Az * claim_Bz
          let alloc_AzBz =
            alloc_claim_Az.mul(cs.namespace(|| format!("AzBz_{}", i)), alloc_claim_Bz)?;

          // u.u * claim_Cz
          let alloc_uCz = alloc_U_u.mul(cs.namespace(|| format!("uCz_{}", i)), alloc_claim_Cz)?;

          // - u.u * claim_Cz
          let neg_alloc_uCz =
            alloc_negate(cs.namespace(|| format!("- u.u * claim_Cz {i}")), &alloc_uCz)?;

          // claim_Az * claim_Bz - u.u * claim_Cz
          let alloc_AzBz_uCz =
            alloc_AzBz.add(cs.namespace(|| format!("AzBz_uCz_{}", i)), &neg_alloc_uCz)?;

          // - eval_E
          let neg_alloc_eval_E =
            alloc_negate(cs.namespace(|| format!("- eval_E {i}")), alloc_eval_E)?;

          // claim_Az * claim_Bz - u.u * claim_Cz - eval_E
          let alloc_AzBz_uCz_E = alloc_AzBz_uCz.add(
            cs.namespace(|| format!("AzBz_uCz_E_{}", i)),
            &neg_alloc_eval_E,
          )?;

          // r * eval_tau
          let alloc_r_eval_tau = alloc_r.mul(
            cs.namespace(|| format!("r_eval_tau_{}", i)),
            &alloc_eval_tau,
          )?;

          // r * eval_tau * (claim_Az * claim_Bz - u.u * claim_Cz - eval_E)
          let alloc_claim =
            alloc_r_eval_tau.mul(cs.namespace(|| format!("claim_{}", i)), &alloc_AzBz_uCz_E)?;

          final_claim =
            final_claim.add(cs.namespace(|| format!("final_claim_{}", i)), &alloc_claim)?;
        }

        final_claim
      };

      assert_eq!(
        claim_outer_final_expected,
        alloc_claim_outer_final_expected.get_value().unwrap()
      );

      cs.enforce(
        || "claim_outer_final_expected == alloc_claim_outer_final_expected",
        |lc| lc + alloc_claim_outer_final_expected.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + alloc_claim_outer_final.get_variable(),
      );

      if claim_outer_final != claim_outer_final_expected {
        return Err(NovaError::InvalidSumcheckProof);
      }

      let inner_r = transcript.squeeze(String::from("in_r"))?;
      let alloc_inner_r =
        transcript_circuit.squeeze(String::from("inner_r"), &mut sponge, &mut cs)?;
      assert_eq!(inner_r, alloc_inner_r.get_value().unwrap());

      let inner_r_square = inner_r.square();
      let alloc_inner_r_square = alloc_inner_r.square(cs.namespace(|| "inner_r_square"))?;

      let inner_r_cube = inner_r_square * inner_r;
      let alloc_inner_r_cube =
        alloc_inner_r_square.mul(cs.namespace(|| "inner_r_cube"), &alloc_inner_r)?;

      let inner_r_powers = powers(&inner_r_cube, num_instances);
      let alloc_inner_r_powers = alloc_powers(
        cs.namespace(|| "inner_r_powers"),
        &alloc_inner_r_cube,
        num_instances,
      )?;

      for (r, alloc_r) in inner_r_powers.iter().zip_eq(alloc_inner_r_powers.iter()) {
        assert_eq!(r, &alloc_r.get_value().unwrap());
      }

      // Compute inner claims Mzᵢ = (Azᵢ + r⋅Bzᵢ + r²⋅Czᵢ),
      // which are batched by Sumcheck into one claim:  ∑ᵢ r³ⁱ⋅Mzᵢ
      let claims_inner = self
        .claims_outer
        .iter()
        .map(|(claim_Az, claim_Bz, claim_Cz)| {
          *claim_Az + inner_r * claim_Bz + inner_r_square * claim_Cz
        })
        .collect::<Vec<_>>();

      let mut alloc_claims_inner = Vec::new();

      for (i, ((claim_Az, claim_Bz, claim_Cz), _)) in alloc_ABCE_evals.iter().enumerate() {
        // inner_r_square * claim_Cz
        let alloc_inner_r_square_Cz = alloc_inner_r_square.mul(
          cs.namespace(|| format!("inner_r_square_Cz_{}", i)),
          claim_Cz,
        )?;

        // inner_r * claim_Bz
        let alloc_inner_r_Bz =
          alloc_inner_r.mul(cs.namespace(|| format!("inner_r_Bz_{}", i)), claim_Bz)?;

        // inner_r * claim_Bz + inner_r_square * claim_Cz
        let alloc_inner_r_Bz_inner_r_square_Cz = alloc_inner_r_Bz.add(
          cs.namespace(|| format!("inner_r_Bz_inner_r_square_Cz_{}", i)),
          &alloc_inner_r_square_Cz,
        )?;

        // claim_Az + inner_r * claim_Bz + inner_r_square * claim_Cz
        let claim_inner = claim_Az.add(
          cs.namespace(|| format!("claim_inner_{}", i)),
          &alloc_inner_r_Bz_inner_r_square_Cz,
        )?;

        alloc_claims_inner.push(claim_inner);
      }

      for (claim, alloc_claim) in claims_inner.iter().zip_eq(alloc_claims_inner.iter()) {
        assert_eq!(claim, &alloc_claim.get_value().unwrap());
      }

      let (claim_inner_final, r_y) = self.sc_proof_inner.verify_batch_poseidon(
        &claims_inner,
        &num_rounds_y,
        &inner_r_powers,
        2,
        &mut transcript,
      )?;

      let (alloc_claim_inner_final, alloc_r_y) = {
        let alloc_inner_degree_bound =
          AllocatedNum::alloc(cs.namespace(|| "inner_degree_bound"), || {
            Ok(E::Scalar::from(2_u64))
          })?;
        let alloc_sc = SCVerifyBatchedGadget::new(self.sc_proof_inner.clone());

        alloc_sc.verify_batched(
          &mut cs,
          &alloc_claims_inner,
          &num_rounds_y,
          &alloc_inner_r_powers,
          &alloc_inner_degree_bound,
          &mut transcript_circuit,
          &mut sponge,
          "inner cs",
        )?
      };

      assert_eq!(
        claim_inner_final,
        alloc_claim_inner_final.get_value().unwrap()
      );

      assert_eq!(
        r_y,
        alloc_r_y
          .iter()
          .map(|r| r.get_value().unwrap())
          .collect::<Vec<_>>()
      );

      let r_y: Vec<Vec<E::Scalar>> = num_rounds_y
        .iter()
        .map(|num_rounds| r_y[(num_rounds_y_max - num_rounds)..].to_vec())
        .collect();

      let alloc_r_y: Vec<Vec<AllocatedNum<E::Scalar>>> = num_rounds_y
        .iter()
        .map(|num_rounds| alloc_r_y[(num_rounds_y_max - num_rounds)..].to_vec())
        .collect();

      for (r_y, alloc_r_y) in r_y.iter().zip_eq(alloc_r_y.iter()) {
        for (r, alloc_r) in r_y.iter().zip_eq(alloc_r_y.iter()) {
          assert_eq!(r, &alloc_r.get_value().unwrap());
        }
      }

      // Compute evaluations of Zᵢ = [Wᵢ, uᵢ, Xᵢ] at r_y
      // Zᵢ(r_y) = (1−r_y[0])⋅W(r_y[1..]) + r_y[0]⋅MLE([uᵢ, Xᵢ])(r_y[1..])
      let evals_Z = zip_with!(iter, (self.evals_W, U, r_y), |eval_W, U, r_y| {
        let eval_X = {
          // constant term
          let poly_X = iter::once(U.u).chain(U.X.iter().cloned()).collect();
          SparsePolynomial::new(r_y.len() - 1, poly_X).evaluate(&r_y[1..])
        };
        (E::Scalar::ONE - r_y[0]) * eval_W + r_y[0] * eval_X
      })
      .collect::<Vec<_>>();

      let alloc_evals_W = self
        .evals_W
        .iter()
        .enumerate()
        .map(|(i, eval)| {
          AllocatedNum::alloc(cs.namespace(|| format!("eval_W_{}", i)), || Ok(*eval))
        })
        .collect::<Result<Vec<_>, _>>()?;

      let mut alloc_evals_Z = Vec::new();

      for (i, ((alloc_eval_W, U), alloc_r_y)) in alloc_evals_W
        .iter()
        .zip_eq(U.iter())
        .zip_eq(alloc_r_y.iter())
        .enumerate()
      {
        let alloc_eval_X = {
          let alloc_poly_X = {
            let poly_X: Vec<E::Scalar> = iter::once(U.u).chain(U.X.iter().cloned()).collect();
            poly_X
              .iter()
              .enumerate()
              .map(|(j, x)| {
                AllocatedNum::alloc(cs.namespace(|| format!("X_{}_{}", i, j)), || Ok(*x))
              })
              .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?
          };

          AllocatedSparsePolynomial::new(alloc_r_y.len() - 1, alloc_poly_X)
            .evaluate(cs.namespace(|| format!("eval_X_{}", i)), &alloc_r_y[1..])?
        };

        // r_y[0] * eval_X
        let alloc_r_y0_eval_X =
          alloc_r_y[0].mul(cs.namespace(|| format!("r_y0_eval_X_{}", i)), &alloc_eval_X)?;

        //  - r_y[0]
        let neg_alloc_r_y0 = alloc_negate(cs.namespace(|| format!("- r_y0_{}", i)), &alloc_r_y[0])?;

        // 1 - r_y[0]
        let alloc_1_r_y0 =
          alloc_one.add(cs.namespace(|| format!("1 - r_y0_{}", i)), &neg_alloc_r_y0)?;

        // (E::Scalar::ONE - r_y[0]) * eval_W
        let alloc_1_r_y0_eval_W = alloc_1_r_y0.mul(
          cs.namespace(|| format!("1 - r_y0_eval_W_{}", i)),
          alloc_eval_W,
        )?;

        // (E::Scalar::ONE - r_y[0]) * eval_W + r_y[0] * eval_X
        let alloc_eval_Z =
          alloc_1_r_y0_eval_W.add(cs.namespace(|| format!("eval_Z_{}", i)), &alloc_r_y0_eval_X)?;

        alloc_evals_Z.push(alloc_eval_Z);
      }

      for (eval, alloc_eval) in evals_Z.iter().zip_eq(alloc_evals_Z.iter()) {
        assert_eq!(eval, &alloc_eval.get_value().unwrap());
      }

      // compute evaluations of R1CS matrices M(r_x, r_y) = eq(r_y)ᵀ⋅M⋅eq(r_x)
      let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                            chi_r_x: &[E::Scalar],
                            r_y: &[E::Scalar]|
       -> Vec<E::Scalar> {
        let evaluate_with_table =
          |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
            M.par_iter_rows()
              .enumerate()
              .map(|(row_idx, row)| {
                M.get_row(row)
                  .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
                  .sum::<E::Scalar>()
              })
              .sum()
          };

        let T_x = chi_r_x;

        let T_y = EqPolynomial::evals_from_points(r_y);

        M_vec
          .par_iter()
          .map(|&M_vec| evaluate_with_table(M_vec, T_x, &T_y))
          .collect()
      };

      // Compute inner claim ∑ᵢ r³ⁱ⋅(Aᵢ(r_x, r_y) + r⋅Bᵢ(r_x, r_y) + r²⋅Cᵢ(r_x, r_y))⋅Zᵢ(r_y)
      let claim_inner_final_expected = zip_with!(
        iter,
        (vk.S, chis_r_x, r_y, evals_Z, inner_r_powers),
        |S, r_x, r_y, eval_Z, r_i| {
          let evals = multi_evaluate(&[&S.A, &S.B, &S.C], r_x, r_y);
          let eval = evals[0] + inner_r * evals[1] + inner_r_square * evals[2];
          eval * r_i * eval_Z
        }
      )
      .sum::<E::Scalar>();

      let mut alloc_claim_inner_final_expected = alloc_zero.clone();

      for (i, ((((S, alloc_r_x), alloc_r_y), alloc_eval_Z), alloc_r_i)) in vk
        .S
        .iter()
        .zip_eq(alloc_chis_r_x.iter())
        .zip_eq(alloc_r_y.iter())
        .zip_eq(alloc_evals_Z.iter())
        .zip_eq(alloc_inner_r_powers.iter())
        .enumerate()
      {
        let alloc_evals = alloc_multi_evaluate::<_, E>(
          cs.namespace(|| format!("multi eval {i}")),
          &[&S.A, &S.B, &S.C],
          &alloc_r_x,
          &alloc_r_y,
        )?;

        // inner_r_square * evals[2]
        let alloc_inner_r_square_evals_2 = alloc_inner_r_square.mul(
          cs.namespace(|| format!("inner_r_square_evals_2_{}", i)),
          &alloc_evals[2],
        )?;

        // inner_r * evals[1]
        let alloc_inner_r_evals_1 = alloc_inner_r.mul(
          cs.namespace(|| format!("inner_r_evals_1_{}", i)),
          &alloc_evals[1],
        )?;

        // inner_r * evals[1] + inner_r_square * evals[2]
        let alloc_inner_r_evals_1_inner_r_square_evals_2 = alloc_inner_r_evals_1.add(
          cs.namespace(|| format!("inner_r_evals_1_inner_r_square_evals_2_{}", i)),
          &alloc_inner_r_square_evals_2,
        )?;

        // evals[0] + inner_r * evals[1] + inner_r_square * evals[2]
        let alloc_eval = alloc_evals[0].add(
          cs.namespace(|| format!("eval_{}", i)),
          &alloc_inner_r_evals_1_inner_r_square_evals_2,
        )?;

        // eval * r_i
        let alloc_eval_r_i =
          alloc_eval.mul(cs.namespace(|| format!("eval_r_i_{}", i)), alloc_r_i)?;

        // eval * r_i * eval_Z
        let alloc_claim = alloc_eval_r_i.mul(
          cs.namespace(|| format!("final inner claim_{}", i)),
          alloc_eval_Z,
        )?;

        alloc_claim_inner_final_expected = alloc_claim_inner_final_expected.add(
          cs.namespace(|| format!("claim_inner_final_expected_{}", i)),
          &alloc_claim,
        )?;
      }

      assert_eq!(
        claim_inner_final_expected,
        alloc_claim_inner_final_expected.get_value().unwrap()
      );

      cs.enforce(
        || "claim_inner_final_expected == alloc_claim_inner_final_expected",
        |lc| lc + alloc_claim_inner_final_expected.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + alloc_claim_inner_final.get_variable(),
      );

      if claim_inner_final != claim_inner_final_expected {
        return Err(NovaError::InvalidSumcheckProof);
      }

      // Create evaluation instances for W(r_y[1..]) and E(r_x)
      let u_vec = {
        let mut u_vec = Vec::with_capacity(2 * num_instances);
        u_vec.extend(zip_with!(iter, (self.evals_W, U, r_y), |eval, u, r_y| {
          PolyEvalInstance {
            c: u.comm_W,
            x: r_y[1..].to_vec(),
            e: *eval,
          }
        }));

        u_vec.extend(zip_with!(iter, (self.evals_E, U, r_x), |eval, u, r_x| {
          PolyEvalInstance {
            c: u.comm_E,
            x: r_x.to_vec(),
            e: *eval,
          }
        }));
        u_vec
      };

      let batched_u = batch_eval_verify_poseidon(
        u_vec,
        &mut transcript,
        &self.sc_proof_batch,
        &self.evals_batch,
      )?;

      // verify
      ipa_pc::EvaluationEngine::verify(
        &vk.vk_ee,
        &batched_u.c,
        &batched_u.x,
        &batched_u.e,
        &self.eval_arg,
      )?;

      Ok(())
    }
  }

  fn alloc_multi_evaluate<CS: ConstraintSystem<E::Scalar>, E: Engine>(
    mut cs: CS,
    M_vec: &[&SparseMatrix<E::Scalar>],
    chi_r_x: &[AllocatedNum<E::Scalar>],
    r_y: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    let T_x = chi_r_x;

    let T_y =
      AllocatedEqPolynomial::evals_from_points(cs.namespace(|| "eq poly eval from points"), r_y)?;

    M_vec
      .iter()
      .enumerate()
      .map(|(i, &M_vec)| {
        alloc_evaluate_with_table::<_, E>(
          cs.namespace(|| format!("evaluate with table {i}")),
          M_vec,
          T_x,
          &T_y,
        )
      })
      .collect()
  }

  fn alloc_evaluate_with_table<CS: ConstraintSystem<E::Scalar>, E: Engine>(
    mut cs: CS,
    M: &SparseMatrix<E::Scalar>,
    T_x: &[AllocatedNum<E::Scalar>],
    T_y: &[AllocatedNum<E::Scalar>],
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
    let alloc_zero = AllocatedNum::alloc(cs.namespace(|| "zero"), || Ok(E::Scalar::ZERO))?;

    let outer_vals = M
      .iter_rows()
      .enumerate()
      .map(|(row_idx, row)| {
        let inner_vals = M
          .get_row(row)
          .enumerate()
          .map(|(i, (val, col_idx))| {
            // T_x[row_idx] * T_y[*col_idx]
            let alloc_T_row_col = T_x[row_idx].mul(
              cs.namespace(|| format!("[{}][{}] * {}", row_idx, i, col_idx)),
              &T_y[*col_idx],
            )?;

            let val = &AllocatedNum::alloc(
              cs.namespace(|| format!("val_{}_{}", row_idx, col_idx)),
              || Ok(*val),
            )?;

            // T_x[row_idx] * T_y[*col_idx] * val
            alloc_T_row_col.mul(
              cs.namespace(|| format!("[{}][{}] * val", row_idx, col_idx)),
              val,
            )
          })
          .collect::<Result<Vec<_>, _>>()?;

        let mut inner_res = alloc_zero.clone();

        for (i, val) in inner_vals.iter().enumerate() {
          inner_res =
            inner_res.add(cs.namespace(|| format!("inner_res_{}_{}", row_idx, i)), val)?;
        }

        Ok(inner_res)
      })
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    let mut outer_res = alloc_zero.clone();

    for (i, val) in outer_vals.iter().enumerate() {
      outer_res = outer_res.add(cs.namespace(|| format!("outer_res_{}", i)), val)?;
    }

    Ok(outer_res)
  }
}
