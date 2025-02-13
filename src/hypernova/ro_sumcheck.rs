//! Implements sumcheck with poseidon RO as the transcript

use crate::{
  constants::{DEFAULT_ABSORBS, NUM_CHALLENGE_BITS},
  errors::NovaError,
  gadgets::scalar_as_base,
  spartan::polys::{multilinear::MultilinearPolynomial, univariate::UniPoly},
  traits::{AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROConstants, ROTrait},
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Sumcheck proof
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ROSumcheckProof<E: Engine> {
  pub(crate) polys: Vec<UniPoly<E::Scalar>>,
}

impl<E: Engine> ROSumcheckProof<E> {
  /// Verify the sumcheck proof
  pub fn verify(
    &self,
    claim: E::Scalar,
    num_rounds: usize,
    degree_bound: usize,
    ro: <Dual<E> as Engine>::RO,
    ro_consts: &ROConstants<Dual<E>>,
  ) -> Result<(E::Scalar, Vec<E::Scalar>), NovaError>
  where
    E: CurveCycleEquipped,
  {
    let mut e = claim;
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut ro = ro;
    // verify that there is a univariate polynomial for each round
    if self.polys.len() != num_rounds {
      return Err(NovaError::InvalidSumcheckProof);
    }

    for i in 0..self.polys.len() {
      let poly = &self.polys[i];
      // verify degree bound
      if poly.degree() != degree_bound {
        return Err(NovaError::InvalidSumcheckProof);
      }

      if poly.eval_at_zero() + poly.eval_at_one() != e {
        return Err(NovaError::InvalidSumcheckProof);
      }

      // append the prover's message to the transcript
      <UniPoly<E::Scalar> as AbsorbInROTrait<Dual<E>>>::absorb_in_ro(poly, &mut ro);

      //derive the verifier's challenge for the next round
      let r_i = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
      ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
      ro.absorb(r_i);

      r.push(r_i);

      // evaluate the claimed degree-ell polynomial at r_i
      e = poly.evaluate(&r_i);
    }

    Ok((e, r))
  }

  #[inline]
  fn compute_eval_points_cubic_hypernova<F>(
    poly_A: &MultilinearPolynomial<E::Scalar>,
    poly_B: &MultilinearPolynomial<E::Scalar>,
    poly_C: &MultilinearPolynomial<E::Scalar>,
    poly_D: &MultilinearPolynomial<E::Scalar>,
    poly_E: &MultilinearPolynomial<E::Scalar>,
    poly_F: &MultilinearPolynomial<E::Scalar>,
    comb_func: &F,
  ) -> (E::Scalar, E::Scalar, E::Scalar)
  where
    F: Fn(E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) -> E::Scalar + Sync,
  {
    let len = poly_A.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = comb_func(
          poly_A[i], poly_B[i], poly_C[i], poly_D[i], poly_E[i], poly_F[i],
        );

        let poly_A_right_term = poly_A[len + i] - poly_A[i];
        let poly_B_right_term = poly_B[len + i] - poly_B[i];
        let poly_C_right_term = poly_C[len + i] - poly_C[i];
        let poly_D_right_term = poly_D[len + i] - poly_D[i];
        let poly_E_right_term = poly_E[len + i] - poly_E[i];
        let poly_F_right_term = poly_F[len + i] - poly_F[i];

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A_right_term;
        let poly_B_bound_point = poly_B[len + i] + poly_B_right_term;
        let poly_C_bound_point = poly_C[len + i] + poly_C_right_term;
        let poly_D_bound_point = poly_D[len + i] + poly_D_right_term;
        let poly_E_bound_point = poly_E[len + i] + poly_E_right_term;
        let poly_F_bound_point = poly_F[len + i] + poly_F_right_term;

        let eval_point_2 = comb_func(
          poly_A_bound_point,
          poly_B_bound_point,
          poly_C_bound_point,
          poly_D_bound_point,
          poly_E_bound_point,
          poly_F_bound_point,
        );

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_A_bound_point = poly_A_bound_point + poly_A_right_term;
        let poly_B_bound_point = poly_B_bound_point + poly_B_right_term;
        let poly_C_bound_point = poly_C_bound_point + poly_C_right_term;
        let poly_D_bound_point = poly_D_bound_point + poly_D_right_term;
        let poly_E_bound_point = poly_E_bound_point + poly_E_right_term;
        let poly_F_bound_point = poly_F_bound_point + poly_F_right_term;
        let eval_point_3 = comb_func(
          poly_A_bound_point,
          poly_B_bound_point,
          poly_C_bound_point,
          poly_D_bound_point,
          poly_E_bound_point,
          poly_F_bound_point,
        );
        (eval_point_0, eval_point_2, eval_point_3)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      )
  }

  pub(crate) fn prove_cubic_hypernova<F>(
    claim: E::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    poly_C: &mut MultilinearPolynomial<E::Scalar>,
    poly_D: &mut MultilinearPolynomial<E::Scalar>,
    poly_E: &mut MultilinearPolynomial<E::Scalar>,
    poly_F: &mut MultilinearPolynomial<E::Scalar>,
    comb_func: F,
    ro: <Dual<E> as Engine>::RO,
    ro_consts: &ROConstants<Dual<E>>,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), NovaError>
  where
    F: Fn(E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) -> E::Scalar + Sync,
    E: CurveCycleEquipped,
  {
    let mut ro = ro;
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut polys = Vec::new();
    let mut claim_per_round = claim;

    for _ in 0..num_rounds {
      let poly = {
        // Make an iterator returning the contributions to the evaluations
        let (eval_point_0, eval_point_2, eval_point_3) = Self::compute_eval_points_cubic_hypernova(
          poly_A, poly_B, poly_C, poly_D, poly_E, poly_F, &comb_func,
        );

        let evals = vec![
          eval_point_0,
          claim_per_round - eval_point_0,
          eval_point_2,
          eval_point_3,
        ];
        UniPoly::from_evals(&evals)
      };

      // append the prover's message to the transcript
      <UniPoly<E::Scalar> as AbsorbInROTrait<Dual<E>>>::absorb_in_ro(&poly, &mut ro);

      //derive the verifier's challenge for the next round
      let r_i = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
      ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
      ro.absorb(r_i);
      r.push(r_i);

      // Set up next round
      claim_per_round = poly.evaluate(&r_i);
      polys.push(poly);

      // bound all tables to the verifier's challenge
      rayon::join(
        || {
          rayon::join(
            || poly_A.bind_poly_var_top(&r_i),
            || {
              rayon::join(
                || poly_B.bind_poly_var_top(&r_i),
                || {
                  rayon::join(
                    || poly_C.bind_poly_var_top(&r_i),
                    || poly_D.bind_poly_var_top(&r_i),
                  )
                },
              )
            },
          )
        },
        || {
          rayon::join(
            || poly_E.bind_poly_var_top(&r_i),
            || poly_F.bind_poly_var_top(&r_i),
          )
        },
      );
    }

    Ok((
      Self { polys },
      r,
      vec![
        poly_A[0], poly_B[0], poly_C[0], poly_D[0], poly_E[0], poly_F[0],
      ],
    ))
  }
}
