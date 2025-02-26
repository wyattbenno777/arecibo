//! Implements split R1CS witness and corresponding R1CS instance, according to
//! the Nebula paper. Used to help facilitate randomized circuits.

use super::{util::fold_witness, R1CSInstance, R1CSShape, R1CSWitness};
use crate::{
  cyclefold::util::absorb_primary_commitment,
  hypernova::error::HyperNovaError,
  spartan::math::Math,
  traits::{
    commitment::CommitmentEngineTrait, CurveCycleEquipped, Dual, Engine, ROTrait,
    TranscriptReprTrait,
  },
  zip_with, Commitment, CommitmentKey, DerandKey, NovaError, CE,
};
use ff::Field;
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

/// A type that holds a linearized R1CS instance.
///
/// Holds split commitments by default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LR1CSInstance<E: Engine> {
  pub(crate) comm_W: Commitment<E>,
  pub(crate) pre_committed: (Commitment<E>, Commitment<E>),
  pub(crate) X: Vec<E::Scalar>,
  /// (Random) evaluation point
  pub(crate) rx: Vec<E::Scalar>,
  /// Evaluation targets
  pub(crate) vs: Vec<E::Scalar>,
  pub(crate) u: E::Scalar,
}

impl<E> LR1CSInstance<E>
where
  E: Engine,
{
  /// Produces a default [`LR1CSInstance]` given [`R1CSShape`]
  pub fn default(S: &R1CSShape<E>) -> Self {
    let comm_W = Commitment::<E>::default();
    Self {
      comm_W,
      pre_committed: (comm_W, comm_W),
      u: E::Scalar::ZERO,
      X: vec![E::Scalar::ZERO; S.num_io],
      rx: vec![E::Scalar::ZERO; S.num_cons.next_power_of_two().log_2()],
      vs: vec![E::Scalar::ZERO; 3],
    }
  }

  /// Fold and [`LR1CSInstance`] with an [`R1CSInstance`]
  pub fn fold(
    &self,
    U2: &SplitR1CSInstance<E>,
    rho: E::Scalar,
    rx: &[E::Scalar],
    sigmas: &[E::Scalar],
    thetas: &[E::Scalar],
  ) -> Result<Self, NovaError> {
    let (X1, u1, comm_W_1) = (&self.X, self.u, &self.comm_W);
    let (X2, comm_W_2) = (&U2.aux.X, &U2.aux.comm_W);
    if self.rx.len() != rx.len() {
      return Err(HyperNovaError::InvalidEvaluationPoint.into());
    }
    if sigmas.len() != thetas.len() {
      return Err(HyperNovaError::InvalidTargets.into());
    }
    let comm_W = *comm_W_1 + *comm_W_2 * rho;
    let pre_committed = (
      self.pre_committed.0 + U2.pre_committed.0 * rho,
      self.pre_committed.1 + U2.pre_committed.1 * rho,
    );
    let u = u1 + rho;
    let X = zip_with!((X1.par_iter(), X2), |a, b| *a + rho * *b).collect::<Vec<E::Scalar>>();
    let vs: Vec<E::Scalar> = sigmas
      .iter()
      .zip_eq(thetas.iter())
      .map(|(sigma, theta)| *sigma + *theta * rho)
      .collect();
    Ok(Self {
      comm_W,
      pre_committed,
      X,
      rx: rx.to_vec(),
      vs,
      u,
    })
  }

  /// Derandomizes the `LR1CSInstance` using a `DerandKey`
  pub(crate) fn derandomize(&self, dk: &DerandKey<E>, r_W: &E::Scalar) -> Self {
    Self {
      comm_W: CE::<E>::derandomize(dk, &self.comm_W, r_W),
      X: self.X.clone(),
      u: self.u,
      pre_committed: self.pre_committed,
      rx: self.rx.clone(),
      vs: self.vs.clone(),
    }
  }

  pub(crate) fn absorb_in_ro(&self, ro: &mut <Dual<E> as Engine>::RO)
  where
    E: CurveCycleEquipped,
  {
    absorb_primary_commitment::<E, Dual<E>>(&self.comm_W, ro);
    ro.absorb(self.u);
    for x in &self.X {
      ro.absorb(*x);
    }
    for x in &self.rx {
      ro.absorb(*x);
    }
    for x in &self.vs {
      ro.absorb(*x);
    }
    absorb_primary_commitment::<E, Dual<E>>(&self.pre_committed.0, ro);
    absorb_primary_commitment::<E, Dual<E>>(&self.pre_committed.1, ro);
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for LR1CSInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_W.to_transcript_bytes(),
      self.pre_committed.0.to_transcript_bytes(),
      self.pre_committed.1.to_transcript_bytes(),
      self.u.to_transcript_bytes(),
      self.X.as_slice().to_transcript_bytes(),
      self.rx.as_slice().to_transcript_bytes(),
      self.vs.as_slice().to_transcript_bytes(),
    ]
    .concat()
  }
}

/// A split R1CS instance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SplitR1CSInstance<E>
where
  E: Engine,
{
  pub(crate) aux: R1CSInstance<E>,
  pub(crate) pre_committed: (Commitment<E>, Commitment<E>),
}

/// A split R1CS witness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SplitR1CSWitness<E>
where
  E: Engine,
{
  pub(crate) aux: R1CSWitness<E>,
  pub(crate) pre_committed: (Vec<E::Scalar>, Vec<E::Scalar>),
}

impl<E> SplitR1CSWitness<E>
where
  E: Engine,
{
  /// Create a new instance of [`SplitR1CSWitness`].
  pub fn new(aux: R1CSWitness<E>, pre_committed: (Vec<E::Scalar>, Vec<E::Scalar>)) -> Self {
    Self { aux, pre_committed }
  }

  /// Folds an incoming [`SplitR1CSWitness`] into the current one
  pub(crate) fn fold(&self, W2: &SplitR1CSWitness<E>, rho: E::Scalar) -> Result<Self, NovaError> {
    let aux = self.aux.fold(&W2.aux, rho)?;
    let pre_committed = (
      fold_witness(&self.pre_committed.0, &W2.pre_committed.0, rho)?,
      fold_witness(&self.pre_committed.1, &W2.pre_committed.1, rho)?,
    );
    Ok(Self { aux, pre_committed })
  }

  /// Get the precommitted commitments
  pub fn commit(&self, ck: &CommitmentKey<E>) -> (Commitment<E>, Commitment<E>) {
    (
      CE::<E>::commit(ck, &self.pre_committed.0, &E::Scalar::ZERO),
      CE::<E>::commit_at(
        ck,
        &self.pre_committed.1,
        &E::Scalar::ZERO,
        self.pre_committed.0.len(),
      ),
    )
  }

  /// Create a default [`SplitR1CSWitness`]
  pub fn default(S: &R1CSShape<E>) -> Self {
    let aux = R1CSWitness::default(S);
    let pre_committed = (
      vec![E::Scalar::ZERO; S.num_precommitted.0],
      vec![E::Scalar::ZERO; S.num_precommitted.1],
    );
    Self { aux, pre_committed }
  }

  /// Construct the witness vector. witness_vec = [aux.W, pre_committed.0, pre_committed.1]
  pub fn W(&self) -> Vec<E::Scalar> {
    [
      &self.pre_committed.0,
      &self.pre_committed.1,
      self.aux.W.as_slice(),
    ]
    .concat()
  }

  /// Pads the provided witness to the correct length
  pub(crate) fn padded_W(&self, S: &R1CSShape<E>) -> Vec<E::Scalar> {
    let mut W = self.W();
    W.extend(vec![E::Scalar::ZERO; S.total_num_vars() - W.len()]);
    W
  }

  /// Derandomizes the `R1CSWitness`
  pub(crate) fn derandomize(&self) -> (Self, E::Scalar) {
    let (aux, r_W) = self.aux.derandomize();
    (
      Self {
        aux,
        pre_committed: self.pre_committed.clone(),
      },
      r_W,
    )
  }
}

impl<E> SplitR1CSInstance<E>
where
  E: Engine,
{
  /// Create a new instance of [`SplitR1CSInstance`].
  pub fn new(aux: R1CSInstance<E>, pre_committed: (Commitment<E>, Commitment<E>)) -> Self {
    Self { aux, pre_committed }
  }
}
