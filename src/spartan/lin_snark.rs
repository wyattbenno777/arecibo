//! This module implements `LinearizedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! This version of Spartan does not use preprocessing so the verifier keeps the entire
//! description of R1CS matrices. This is essentially optimal for the verifier when using
//! an IPA-based polynomial commitment scheme.

use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{
    split::{LR1CSInstance, SplitR1CSWitness},
    R1CSShape, SparseMatrix,
  },
  spartan::{
    compute_eval_table_sparse,
    polys::{
      eq::EqPolynomial,
      multilinear::{MultilinearPolynomial, SparsePolynomial},
    },
    sumcheck::SumcheckProof,
  },
  traits::{
    evaluation::EvaluationEngineTrait,
    snark::{DigestHelperTrait, LinearizedR1CSSNARKTrait},
    Engine, TranscriptEngineTrait,
  },
  CommitmentKey,
};

use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A type that represents the prover's key
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  vk_digest: E::Scalar, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  vk_ee: EE::VerifierKey,
  S: R1CSShape<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for VerifierKey<E, EE> {}

impl<E: Engine, EE: EvaluationEngineTrait<E>> VerifierKey<E, EE> {
  fn new(shape: R1CSShape<E>, vk_ee: EE::VerifierKey) -> Self {
    Self {
      vk_ee,
      S: shape,
      digest: OnceCell::new(),
    }
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> DigestHelperTrait<E> for VerifierKey<E, EE> {
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

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LinearizedR1CSSNARK<E: Engine, EE: EvaluationEngineTrait<E>> {
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  eval_arg: EE::EvaluationArgument,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> LinearizedR1CSSNARKTrait<E>
  for LinearizedR1CSSNARK<E, EE>
{
  type ProverKey = ProverKey<E, EE>;
  type VerifierKey = VerifierKey<E, EE>;

  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: &R1CSShape<E>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    let (pk_ee, vk_ee) = EE::setup(ck);
    let S = S.pad();
    let vk: VerifierKey<E, EE> = VerifierKey::new(S, vk_ee);
    let pk = ProverKey {
      pk_ee,
      vk_digest: vk.digest(),
    };
    Ok((pk, vk))
  }

  /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
  #[tracing::instrument(skip_all, name = "SNARK::prove")]
  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &LR1CSInstance<E>,
    W: &SplitR1CSWitness<E>,
  ) -> Result<Self, NovaError> {
    // pad the R1CSShape
    let S = S.pad();
    // sanity check that R1CSShape has all required size characteristics
    assert!(S.is_regular_shape());
    let W = W.padded_W(&S); // pad the witness
    let mut transcript = E::TE::new(b"LinearizedR1CSSNARK");

    // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let mut z = [W.as_slice(), [U.u].as_slice(), U.X.as_slice()].concat();

    // inner sum-check
    let num_rounds_y = usize::try_from(S.total_num_vars().ilog2()).unwrap() + 1;
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = U.vs[0] + r * U.vs[1] + r * r * U.vs[2];

    let poly_ABC = {
      // compute the initial evaluation table for R(\tau, x)
      let evals_rx = EqPolynomial::evals_from_points(&U.rx.clone());
      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&S, &evals_rx);
      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    };
    let poly_z = {
      z.resize(S.total_num_vars() * 2, E::Scalar::ZERO);
      z
    };
    let comb_func = |poly_A_comp: &E::Scalar, poly_B_comp: &E::Scalar| -> E::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      comb_func,
      &mut transcript,
    )?;

    let eval_W = MultilinearPolynomial::evaluate_with(&W, &r_y[1..]);
    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &(U.pre_committed.0 + U.pre_committed.1 + U.comm_W),
      &W,
      &r_y[1..],
      &eval_W,
    )?;

    Ok(Self {
      sc_proof_inner,
      eval_W,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey, U: &LR1CSInstance<E>) -> Result<(), NovaError> {
    let mut transcript = E::TE::new(b"LinearizedR1CSSNARK");

    // append the digest of R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U);

    // inner sum-check
    let num_rounds_y = usize::try_from(vk.S.total_num_vars().ilog2()).unwrap() + 1;
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = U.vs[0] + r * U.vs[1] + r * r * U.vs[2];
    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)?;

    // --- verify claim_inner_final ---
    let eval_Z = {
      let eval_X = {
        // public IO is (u, X)
        let X = vec![U.u]
          .into_iter()
          .chain(U.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(usize::try_from(vk.S.total_num_vars().ilog2()).unwrap(), X)
          .evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };
    // compute evaluations of R1CS matrices
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
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
      let (T_x, T_y) = rayon::join(
        || EqPolynomial::evals_from_points(r_x),
        || EqPolynomial::evals_from_points(r_y),
      );
      (0..M_vec.len())
        .into_par_iter()
        .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
        .collect()
    };
    let evals = multi_evaluate(&[&vk.S.A, &vk.S.B, &vk.S.C], &U.rx, &r_y);
    let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // verify the evaluation argument
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &(U.comm_W + U.pre_committed.0 + U.pre_committed.1),
      &r_y[1..],
      &self.eval_W,
      &self.eval_arg,
    )?;

    Ok(())
  }
}
