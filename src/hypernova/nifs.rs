//! This module implements the HyperNova folding scheme.
use crate::constants::NUM_CHALLENGE_BITS;
use crate::gadgets::scalar_as_base;
use crate::spartan::compute_eval_table_sparse;
use crate::spartan::math::Math;
use crate::spartan::polys::eq::EqPolynomial;
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::spartan::sumcheck::SumcheckProof;
use crate::traits::ROTrait;
use crate::NovaError;
use crate::{
  constants::DEFAULT_ABSORBS,
  cyclefold::util::absorb_primary_r1cs,
  r1cs::{LR1CSInstance, R1CSInstance, R1CSShape, R1CSWitness},
  traits::{CurveCycleEquipped, Dual, Engine, ROConstants, TranscriptEngineTrait},
};
use ff::Field;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

/// A SNARK that holds the proof of a step of an incremental computation
pub struct NIFS<E: CurveCycleEquipped> {
  pub(crate) sc: SumcheckProof<E>,
  pub(crate) sigmas: Vec<E::Scalar>,
  pub(crate) thetas: Vec<E::Scalar>,
}

impl<E> NIFS<E>
where
  E: CurveCycleEquipped,
{
  /// Prove a step of an incremental computation
  pub fn prove(
    S: R1CSShape<E>,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    (U1, W1): (&LR1CSInstance<E>, &R1CSWitness<E>),
    (U2, W2): (&R1CSInstance<E>, &R1CSWitness<E>),
  ) -> Result<(Self, (LR1CSInstance<E>, R1CSWitness<E>)), NovaError> {
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(*pp_digest);
    absorb_primary_r1cs::<E, Dual<E>>(U2, &mut ro);
    let rho = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(rho);
    let gamma = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(gamma);
    let s = S.num_cons.log_2();
    let beta = {
      ro.squeeze_vec(NUM_CHALLENGE_BITS, s)
        .iter()
        .map(|b| scalar_as_base::<Dual<E>>(*b))
        .collect::<Vec<_>>()
    };
    let z1 = [U1.X.as_slice(), [U1.u].as_slice(), W1.W.as_slice()].concat();
    let z2 = [
      U2.X.as_slice(),
      [E::Scalar::ONE].as_slice(),
      W2.W.as_slice(),
    ]
    .concat();
    let mut poly_ABC = {
      // compute the initial evaluation table for R(\tau, x)
      let evals_rx = EqPolynomial::evals_from_points(&U1.rx);
      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&S, &evals_rx);
      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      let evals_ABC = (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + gamma * evals_B[i] + gamma * gamma * evals_C[i])
        .collect::<Vec<E::Scalar>>();
      MultilinearPolynomial::new(evals_ABC)
    };
    let mut poly_z = {
      let mut padded_z1 = z1.clone();
      padded_z1.resize(S.num_vars * 2, E::Scalar::ZERO);
      MultilinearPolynomial::new(padded_z1)
    };
    let L_comb_func = |abc: E::Scalar, z: E::Scalar| -> E::Scalar { abc * z };
    let eq_beta = EqPolynomial::new(beta);
    let gamma_cubed = gamma * gamma * gamma;
    let mut poly_beta =
      MultilinearPolynomial::new(eq_beta.evals().iter().map(|b| *b * gamma_cubed).collect());
    let (mut poly_Az, mut poly_Bz, mut poly_Cz) = {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(&z2)?;
      (
        MultilinearPolynomial::new(poly_Az),
        MultilinearPolynomial::new(poly_Bz),
        MultilinearPolynomial::new(poly_Cz),
      )
    };
    let Q_comb_func =
      |a: E::Scalar, b: E::Scalar, c: E::Scalar, eq: E::Scalar| -> E::Scalar { (a * b - c) * eq };
    let comb_func =
      |L_abc: E::Scalar,
       L_z: E::Scalar,
       Q_a: E::Scalar,
       Q_b: E::Scalar,
       Q_c: E::Scalar,
       Q_eq: E::Scalar|
       -> E::Scalar { L_comb_func(L_abc, L_z) + Q_comb_func(Q_a, Q_b, Q_c, Q_eq) };
    let claim = U1.vs[0] + U1.vs[1] * gamma + U1.vs[2] * gamma * gamma;
    let (sc, rx_p, _) = SumcheckProof::<E>::prove_cubic_hypernova(
      claim,
      s,
      &mut poly_ABC,
      &mut poly_z,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_Cz,
      &mut poly_beta,
      comb_func,
      &mut E::TE::new(b"transcript"),
    )?;
    let sigmas = {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(&z1)?;
      vec![
        MultilinearPolynomial::new(poly_Az).evaluate(&rx_p),
        MultilinearPolynomial::new(poly_Bz).evaluate(&rx_p),
        MultilinearPolynomial::new(poly_Cz).evaluate(&rx_p),
      ]
    };
    let thetas = {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(&z2)?;
      vec![
        MultilinearPolynomial::new(poly_Az).evaluate(&rx_p),
        MultilinearPolynomial::new(poly_Bz).evaluate(&rx_p),
        MultilinearPolynomial::new(poly_Cz).evaluate(&rx_p),
      ]
    };
    let U = U1.fold(U2, rho, &rx_p, &sigmas, &thetas)?;
    let W = W1.fold(W2, rho)?;
    Ok((Self { sc, sigmas, thetas }, (U, W)))
  }

  /// Verify a fold
  pub fn verify(
    &self,
    s: usize,
    ro_consts: &ROConstants<Dual<E>>,
    pp_digest: &E::Scalar,
    U1: &LR1CSInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<LR1CSInstance<E>, NovaError> {
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(*pp_digest);
    absorb_primary_r1cs::<E, Dual<E>>(U2, &mut ro);
    let rho = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(rho);
    let gamma = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
    let mut ro = <Dual<E> as Engine>::RO::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(gamma);
    let beta = {
      ro.squeeze_vec(NUM_CHALLENGE_BITS, s)
        .iter()
        .map(|b| scalar_as_base::<Dual<E>>(*b))
        .collect::<Vec<_>>()
    };
    let claim = U1.vs[0] + U1.vs[1] * gamma + U1.vs[2] * gamma * gamma;
    let (new_claim, rx_p) = self
      .sc
      .verify(claim, s, 3, &mut E::TE::new(b"transcript"))?;
    let e1 = EqPolynomial::new(U1.rx.to_vec()).evaluate(&rx_p);
    let cl = (self.sigmas[0] + self.sigmas[1] * gamma + self.sigmas[2] * gamma * gamma) * e1;
    let e2 = EqPolynomial::new(beta).evaluate(&rx_p);
    let gamma_cubed = gamma * gamma * gamma;
    let cr = (self.thetas[0] * self.thetas[1] - self.thetas[2]) * e2 * gamma_cubed;
    if cl + cr != new_claim {
      return Err(NovaError::InvalidSumcheckProof);
    }
    let U = U1.fold(U2, rho, &rx_p, &self.sigmas, &self.thetas)?;
    Ok(U)
  }
}
