//! This module defines R1CS related types and a folding scheme for Relaxed R1CS
mod sparse;
pub(crate) mod util;

use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  digest::SimpleDigestible,
  errors::NovaError,
  gadgets::{f_to_nat, nat_to_limbs, scalar_as_base},
  spartan::polys::multilinear::MultilinearPolynomial,
  traits::{
    commitment::CommitmentEngineTrait, AbsorbInROTrait, Engine, ROTrait, TranscriptReprTrait,
  },
  zip_with, Commitment, CommitmentKey, DerandKey, CE,
};
use core::cmp::max;
use ff::Field;
use once_cell::sync::OnceCell;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub(crate) use sparse::SparseMatrix;
use split::{LR1CSInstance, SplitR1CSInstance, SplitR1CSWitness};
use util::fold_witness;
pub mod split;
/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSShape<E: Engine> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_precommitted: (usize, usize),
  pub(crate) num_io: usize,
  pub(crate) A: SparseMatrix<E::Scalar>,
  pub(crate) B: SparseMatrix<E::Scalar>,
  pub(crate) C: SparseMatrix<E::Scalar>,
  #[serde(skip, default = "OnceCell::new")]
  pub(crate) digest: OnceCell<E::Scalar>,
}

impl<E: Engine> SimpleDigestible for R1CSShape<E> {}

/// A type that holds the result of a R1CS multiplication
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSResult<E: Engine> {
  pub(crate) AZ: Vec<E::Scalar>,
  pub(crate) BZ: Vec<E::Scalar>,
  pub(crate) CZ: Vec<E::Scalar>,
}

/// A type that holds a witness for a given R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSWitness<E: Engine> {
  pub(crate) W: Vec<E::Scalar>,
  pub(crate) r_W: E::Scalar,
}

/// A type that holds an R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSInstance<E: Engine> {
  /// commitment to witness
  pub comm_W: Commitment<E>,
  /// public inputs
  pub(crate) X: Vec<E::Scalar>,
}

/// A type that holds a witness for a given Relaxed R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelaxedR1CSWitness<E: Engine> {
  pub(crate) W: Vec<E::Scalar>,
  pub(crate) r_W: E::Scalar,
  pub(crate) E: Vec<E::Scalar>,
  pub(crate) r_E: E::Scalar,
}

/// A type that holds a Relaxed R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSInstance<E: Engine> {
  /// commitment to witness
  pub comm_W: Commitment<E>,
  /// commitment to error
  pub comm_E: Commitment<E>,
  /// public inputs
  pub X: Vec<E::Scalar>,
  /// relaxation
  pub u: E::Scalar,
}

/// A type for functions that hints commitment key sizing by returning the floor of the number of required generators.
pub type CommitmentKeyHint<E> = dyn Fn(&R1CSShape<E>) -> usize;

/// Generates public parameters for a Rank-1 Constraint System (R1CS).
///
/// This function takes into consideration the shape of the R1CS matrices and a hint function
/// for the number of generators. It returns a `CommitmentKey`.
///
/// # Arguments
///
/// * `S`: The shape of the R1CS matrices.
/// * `ck_floor`: A function that provides a floor for the number of generators. A good function to
///   provide is the `commitment_key_floor` field in the trait `RelaxedR1CSSNARKTrait`.
///
#[tracing::instrument(level = "debug", skip_all name="generate ck")]
pub(crate) fn commitment_key<E: Engine>(
  S: &R1CSShape<E>,
  ck_floor: &CommitmentKeyHint<E>,
) -> CommitmentKey<E> {
  let size = commitment_key_size(S, ck_floor);
  E::CE::setup(b"ck", size)
}

/// Computes the number of generators required for the commitment key corresponding to shape `S`.
pub(crate) fn commitment_key_size<E: Engine>(
  S: &R1CSShape<E>,
  ck_floor: &CommitmentKeyHint<E>,
) -> usize {
  let num_cons = S.num_cons;
  let num_vars = S.total_num_vars();
  let ck_hint = ck_floor(S);
  max(max(num_cons, num_vars), ck_hint)
}

impl<E: Engine> R1CSShape<E> {
  /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
  pub(crate) fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    num_precommitted: (usize, usize),
    A: SparseMatrix<E::Scalar>,
    B: SparseMatrix<E::Scalar>,
    C: SparseMatrix<E::Scalar>,
  ) -> Result<Self, NovaError> {
    let is_valid = |M: &SparseMatrix<E::Scalar>| -> Result<Vec<()>, NovaError> {
      M.iter()
        .map(|(row, col, _val)| {
          if row >= num_cons || col > num_io + num_vars + num_precommitted.0 + num_precommitted.1 {
            Err(NovaError::InvalidIndex)
          } else {
            Ok(())
          }
        })
        .collect::<Result<Vec<()>, NovaError>>()
    };

    is_valid(&A)?;
    is_valid(&B)?;
    is_valid(&C)?;

    // We require the number of public inputs/outputs to be even
    if num_io % 2 != 0 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    Ok(Self {
      num_cons,
      num_vars,
      num_io,
      num_precommitted,
      A,
      B,
      C,
      digest: OnceCell::new(),
    })
  }

  /// Get the total number of variables in the R1CS instance
  pub(crate) fn total_num_vars(&self) -> usize {
    self.num_vars + self.num_precommitted.0 + self.num_precommitted.1
  }

  // Checks regularity conditions on the R1CSShape, required in Spartan-class SNARKs
  // Returns false if num_cons or num_vars are not powers of two, or if num_io > num_vars
  #[inline]
  pub(crate) fn is_regular_shape(&self) -> bool {
    let num_vars = self.total_num_vars();
    let cons_valid = self.num_cons.next_power_of_two() == self.num_cons;
    let vars_valid = num_vars.next_power_of_two() == num_vars;
    let io_lt_vars = self.num_io < num_vars;
    cons_valid && vars_valid && io_lt_vars
  }

  pub(crate) fn multiply_vec(
    &self,
    z: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), NovaError> {
    if z.len()
      != self.num_io + self.num_vars + 1 + self.num_precommitted.0 + self.num_precommitted.1
    {
      return Err(NovaError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_vec(z),
      || rayon::join(|| self.B.multiply_vec(z), || self.C.multiply_vec(z)),
    );

    Ok((Az, Bz, Cz))
  }

  pub(crate) fn multiply_witness(
    &self,
    W: &[E::Scalar],
    u: &E::Scalar,
    X: &[E::Scalar],
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), NovaError> {
    if X.len() != self.num_io
      || W.len() != self.num_vars + self.num_precommitted.0 + self.num_precommitted.1
    {
      return Err(NovaError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_witness(W, u, X),
      || {
        rayon::join(
          || self.B.multiply_witness(W, u, X),
          || self.C.multiply_witness(W, u, X),
        )
      },
    );

    Ok((Az, Bz, Cz))
  }

  /// Computes the error term E = Az * Bz - u*Cz.
  fn compute_E(
    &self,
    W: &[E::Scalar],
    u: &E::Scalar,
    X: &[E::Scalar],
  ) -> Result<Vec<E::Scalar>, NovaError> {
    let num_vars = self.num_vars + self.num_precommitted.0 + self.num_precommitted.1;
    if X.len() != self.num_io || W.len() != num_vars {
      return Err(NovaError::InvalidWitnessLength);
    }

    let (Az, (Bz, Cz)) = rayon::join(
      || self.A.multiply_witness(W, u, X),
      || {
        rayon::join(
          || self.B.multiply_witness(W, u, X),
          || self.C.multiply_witness(W, u, X),
        )
      },
    );

    let E = zip_with!(
      (Az.into_par_iter(), Bz.into_par_iter(), Cz.into_par_iter()),
      |a, b, c| a * b - c * u
    )
    .collect::<Vec<E::Scalar>>();

    Ok(E)
  }

  /// Checks if the Relaxed R1CS instance is satisfiable given a witness and its shape
  pub(crate) fn is_sat_relaxed(
    &self,
    ck: &CommitmentKey<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
  ) -> Result<(), NovaError> {
    let num_vars = self.num_vars + self.num_precommitted.0 + self.num_precommitted.1;
    assert_eq!(W.W.len(), num_vars);
    assert_eq!(W.E.len(), self.num_cons);
    assert_eq!(U.X.len(), self.num_io);

    // verify if Az * Bz - u*Cz = E
    let E = self.compute_E(&W.W, &U.u, &U.X)?;
    W.E
      .par_iter()
      .zip_eq(E.into_par_iter())
      .enumerate()
      .try_for_each(|(i, (we, e))| {
        if *we != e {
          // constraint failed, retrieve constraint name
          Err(NovaError::UnSatIndex(i))
        } else {
          Ok(())
        }
      })?;

    // verify if comm_E and comm_W are commitments to E and W
    let res_comm = {
      let (comm_W, comm_E) = rayon::join(
        || {
          CE::<E>::commit_at(
            ck,
            &W.W,
            &W.r_W,
            self.num_precommitted.0 + self.num_precommitted.1,
          )
        },
        || CE::<E>::commit(ck, &W.E, &W.r_E),
      );
      U.comm_W == comm_W && U.comm_E == comm_E
    };

    if !res_comm {
      return Err(NovaError::UnSat);
    }

    Ok(())
  }

  /// Checks if the R1CS instance is satisfiable given a witness and its shape
  pub(crate) fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    U: &R1CSInstance<E>,
    W: &R1CSWitness<E>,
  ) -> Result<(), NovaError> {
    let num_vars = self.num_vars + self.num_precommitted.0 + self.num_precommitted.1;
    assert_eq!(W.W.len(), num_vars);
    assert_eq!(U.X.len(), self.num_io);

    // verify if Az * Bz - u*Cz = 0
    let E = self.compute_E(&W.W, &E::Scalar::ONE, &U.X)?;
    E.into_par_iter().enumerate().try_for_each(|(i, e)| {
      if e != E::Scalar::ZERO {
        Err(NovaError::UnSatIndex(i))
      } else {
        Ok(())
      }
    })?;

    // verify if comm_W is a commitment to W
    if U.comm_W
      != CE::<E>::commit_at(
        ck,
        &W.W,
        &W.r_W,
        self.num_precommitted.0 + self.num_precommitted.1,
      )
    {
      return Err(NovaError::UnSat);
    }
    Ok(())
  }

  /// Checks if the R1CS instance is satisfiable given a witness and its shape
  pub(crate) fn is_sat_split(
    &self,
    ck: &CommitmentKey<E>,
    U: &SplitR1CSInstance<E>,
    W: &SplitR1CSWitness<E>,
  ) -> Result<(), NovaError> {
    let num_vars = self.num_vars + self.num_precommitted.0 + self.num_precommitted.1;
    assert_eq!(W.W().len(), num_vars);
    assert_eq!(U.aux.X.len(), self.num_io);

    // verify if Az * Bz - u*Cz = 0
    let E = self.compute_E(&W.W(), &E::Scalar::ONE, &U.aux.X)?;
    E.into_par_iter().enumerate().try_for_each(|(i, e)| {
      if e != E::Scalar::ZERO {
        Err(NovaError::UnSatIndex(i))
      } else {
        Ok(())
      }
    })?;

    // verify if comm_W is a commitment to W
    if U.aux.comm_W
      != CE::<E>::commit_at(
        ck,
        &W.aux.W,
        &W.aux.r_W,
        self.num_precommitted.0 + self.num_precommitted.1,
      )
      || U.pre_committed
        != (
          CE::<E>::commit(ck, &W.pre_committed.0, &E::Scalar::ZERO),
          CE::<E>::commit_at(
            ck,
            &W.pre_committed.1,
            &E::Scalar::ZERO,
            self.num_precommitted.0,
          ),
        )
    {
      return Err(NovaError::UnSat);
    }
    Ok(())
  }

  /// Checks if the R1CS instance is satisfiable given a witness and its shape
  pub(crate) fn is_sat_linearized(
    &self,
    ck: &CommitmentKey<E>,
    U: &LR1CSInstance<E>,
    W: &SplitR1CSWitness<E>,
  ) -> Result<(), NovaError> {
    let num_vars = self.num_vars + self.num_precommitted.0 + self.num_precommitted.1;
    assert_eq!(W.W().len(), num_vars);
    assert_eq!(U.X.len(), self.num_io);
    let (Az, Bz, Cz) = self.multiply_witness(&W.W(), &U.u, &U.X)?;

    // Helper functions for resizing polynomials and evaluating them
    let eval_padded_poly = |mut vec: Vec<E::Scalar>| {
      vec.resize(self.num_cons.next_power_of_two(), E::Scalar::ZERO);
      MultilinearPolynomial::new(vec).evaluate(&U.rx)
    };
    assert_eq!(U.vs[0], eval_padded_poly(Az));
    assert_eq!(U.vs[1], eval_padded_poly(Bz));
    assert_eq!(U.vs[2], eval_padded_poly(Cz));

    // verify if comm_W is a commitment to W
    if U.comm_W
      != CE::<E>::commit_at(
        ck,
        &W.aux.W,
        &W.aux.r_W,
        self.num_precommitted.0 + self.num_precommitted.1,
      )
      || U.pre_committed
        != (
          CE::<E>::commit(ck, &W.pre_committed.0, &E::Scalar::ZERO),
          CE::<E>::commit_at(
            ck,
            &W.pre_committed.1,
            &E::Scalar::ZERO,
            self.num_precommitted.0,
          ),
        )
    {
      return Err(NovaError::UnSat);
    }
    Ok(())
  }

  /// A method to compute a commitment to the cross-term `T` given a
  /// Relaxed R1CS instance-witness pair and an R1CS instance-witness pair
  pub(crate) fn commit_T(
    &self,
    ck: &CommitmentKey<E>,
    U1: &RelaxedR1CSInstance<E>,
    W1: &RelaxedR1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
    r_T: &E::Scalar,
  ) -> Result<(Vec<E::Scalar>, Commitment<E>), NovaError> {
    let (AZ_1, BZ_1, CZ_1) = tracing::trace_span!("AZ_1, BZ_1, CZ_1")
      .in_scope(|| self.multiply_witness(&W1.W, &U1.u, &U1.X))?;

    let (AZ_2, BZ_2, CZ_2) = tracing::trace_span!("AZ_2, BZ_2, CZ_2")
      .in_scope(|| self.multiply_witness(&W2.W, &E::Scalar::ONE, &U2.X))?;

    let (AZ_1_circ_BZ_2, AZ_2_circ_BZ_1, u_1_cdot_CZ_2, u_2_cdot_CZ_1) =
      tracing::trace_span!("cross terms").in_scope(|| {
        let AZ_1_circ_BZ_2 = (0..AZ_1.len())
          .into_par_iter()
          .map(|i| AZ_1[i] * BZ_2[i])
          .collect::<Vec<E::Scalar>>();
        let AZ_2_circ_BZ_1 = (0..AZ_2.len())
          .into_par_iter()
          .map(|i| AZ_2[i] * BZ_1[i])
          .collect::<Vec<E::Scalar>>();
        let u_1_cdot_CZ_2 = (0..CZ_2.len())
          .into_par_iter()
          .map(|i| U1.u * CZ_2[i])
          .collect::<Vec<E::Scalar>>();
        let u_2_cdot_CZ_1 = (0..CZ_1.len())
          .into_par_iter()
          .map(|i| CZ_1[i])
          .collect::<Vec<E::Scalar>>();
        (AZ_1_circ_BZ_2, AZ_2_circ_BZ_1, u_1_cdot_CZ_2, u_2_cdot_CZ_1)
      });

    let T = tracing::trace_span!("T").in_scope(|| {
      AZ_1_circ_BZ_2
        .par_iter()
        .zip_eq(&AZ_2_circ_BZ_1)
        .zip_eq(&u_1_cdot_CZ_2)
        .zip_eq(&u_2_cdot_CZ_1)
        .map(|(((a, b), c), d)| *a + *b - *c - *d)
        .collect::<Vec<E::Scalar>>()
    });

    let comm_T = CE::<E>::commit(ck, &T, r_T);

    Ok((T, comm_T))
  }

  /// A method to compute a commitment to the cross-term `T` given two
  /// Relaxed R1CS instance-witness pair
  pub(crate) fn commit_T_relaxed(
    &self,
    ck: &CommitmentKey<E>,
    U1: &RelaxedR1CSInstance<E>,
    W1: &RelaxedR1CSWitness<E>,
    U2: &RelaxedR1CSInstance<E>,
    W2: &RelaxedR1CSWitness<E>,
    r_T: &E::Scalar,
  ) -> Result<(Vec<E::Scalar>, Commitment<E>), NovaError> {
    let Z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
    let Z2 = [W2.W.clone(), vec![U2.u], U2.X.clone()].concat();

    // The following code uses the optimization suggested in
    // Section 5.2 of [Mova](https://eprint.iacr.org/2024/1220.pdf)
    let Z = Z1
      .into_par_iter()
      .zip_eq(Z2.into_par_iter())
      .map(|(z1, z2)| z1 + z2)
      .collect::<Vec<E::Scalar>>();
    let u = U1.u + U2.u;

    let (AZ, BZ, CZ) = self.multiply_vec(&Z)?;

    let T = AZ
      .par_iter()
      .zip_eq(BZ.par_iter())
      .zip_eq(CZ.par_iter())
      .zip_eq(W1.E.par_iter())
      .zip_eq(W2.E.par_iter())
      .map(|((((az, bz), cz), e1), e2)| *az * *bz - u * *cz - *e1 - *e2)
      .collect::<Vec<E::Scalar>>();

    let comm_T = CE::<E>::commit(ck, &T, r_T);

    Ok((T, comm_T))
  }

  /// Pads the `R1CSShape` so that the shape passes `is_regular_shape`
  /// Renumbers variables to accommodate padded variables
  pub(crate) fn pad(&self) -> Self {
    let num_vars = self.total_num_vars();
    // check if the provided R1CSShape is already as required
    if self.is_regular_shape() {
      return self.clone();
    }

    // equalize the number of variables, constraints, and public IO
    let m = max(max(num_vars, self.num_cons), self.num_io).next_power_of_two();

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if num_vars == m {
      return Self {
        num_cons: m,
        num_vars: self.num_vars,
        num_precommitted: self.num_precommitted,
        num_io: self.num_io,
        A: self.A.clone(),
        B: self.B.clone(),
        C: self.C.clone(),
        digest: OnceCell::new(),
      };
    }

    // otherwise, we need to pad the number of variables and renumber variable accesses
    let num_vars_padded = m;
    let num_cons_padded = m;

    let apply_pad = |mut M: SparseMatrix<E::Scalar>| -> SparseMatrix<E::Scalar> {
      M.indices.par_iter_mut().for_each(|c| {
        if *c >= self.total_num_vars() {
          *c += num_vars_padded - num_vars
        }
      });

      M.cols += num_vars_padded - num_vars;

      let ex = {
        let nnz = M.indptr.last().unwrap();
        vec![*nnz; num_cons_padded - self.num_cons]
      };
      M.indptr.extend(ex);
      M
    };

    let A_padded = apply_pad(self.A.clone());
    let B_padded = apply_pad(self.B.clone());
    let C_padded = apply_pad(self.C.clone());

    Self {
      num_cons: num_cons_padded,
      num_vars: num_vars_padded - self.num_precommitted.0 - self.num_precommitted.1,
      num_precommitted: self.num_precommitted,
      num_io: self.num_io,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest: OnceCell::new(),
    }
  }

  /// Samples a new random `RelaxedR1CSInstance`/`RelaxedR1CSWitness` pair
  pub(crate) fn sample_random_instance_witness(
    &self,
    ck: &CommitmentKey<E>,
  ) -> Result<(RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>), NovaError> {
    // sample Z = (W, u, X)
    let num_vars = self.num_vars + self.num_precommitted.0 + self.num_precommitted.1;
    let Z = (0..num_vars + self.num_io + 1)
      .into_par_iter()
      .map(|_| E::Scalar::random(&mut OsRng))
      .collect::<Vec<E::Scalar>>();

    let r_W = E::Scalar::random(&mut OsRng);
    let r_E = E::Scalar::random(&mut OsRng);

    let u = Z[num_vars];

    // compute E <- AZ o BZ - u * CZ
    let (AZ, BZ, CZ) = self.multiply_vec(&Z)?;

    let E = AZ
      .par_iter()
      .zip(BZ.par_iter())
      .zip(CZ.par_iter())
      .map(|((az, bz), cz)| *az * *bz - u * *cz)
      .collect::<Vec<E::Scalar>>();

    // compute commitments to W,E in parallel
    let (comm_W, comm_E) = rayon::join(
      || CE::<E>::commit(ck, &Z[..num_vars], &r_W),
      || CE::<E>::commit(ck, &E, &r_E),
    );

    Ok((
      RelaxedR1CSInstance {
        comm_W,
        comm_E,
        u,
        X: Z[num_vars + 1..].to_vec(),
      },
      RelaxedR1CSWitness {
        W: Z[..num_vars].to_vec(),
        r_W,
        E,
        r_E,
      },
    ))
  }
}

impl<E: Engine> R1CSWitness<E> {
  /// Produces a default `R1CSWitness` given an `R1CSShape`
  pub fn default(S: &R1CSShape<E>) -> Self {
    Self {
      W: vec![E::Scalar::ZERO; S.num_vars],
      r_W: E::Scalar::ZERO,
    }
  }

  /// A method to create a witness object using a vector of scalars
  pub(crate) fn new(S: &R1CSShape<E>, W: Vec<E::Scalar>) -> Result<Self, NovaError> {
    if S.num_vars != W.len() {
      Err(NovaError::InvalidWitnessLength)
    } else {
      Ok(Self {
        W,
        r_W: E::Scalar::random(&mut OsRng),
      })
    }
  }

  /// Derandomizes the `R1CSWitness` using a `DerandKey`
  pub(crate) fn derandomize(&self) -> (Self, E::Scalar) {
    (
      R1CSWitness {
        W: self.W.clone(),
        r_W: E::Scalar::ZERO,
      },
      self.r_W,
    )
  }

  /// Commits to the witness using the supplied generators
  pub(crate) fn commit_at(&self, ck: &CommitmentKey<E>, idx: usize) -> Commitment<E> {
    CE::<E>::commit_at(ck, &self.W, &self.r_W, idx)
  }

  /// Folds an incoming `R1CSWitness` into the current one
  pub(crate) fn fold(&self, W2: &R1CSWitness<E>, rho: E::Scalar) -> Result<Self, NovaError> {
    let (W1, r_W1) = (&self.W, &self.r_W);
    let (W2, r_W2) = (&W2.W, &W2.r_W);
    let W = fold_witness(W1, W2, rho)?;
    let r_W = *r_W1 + rho * r_W2;
    Ok(Self { W, r_W })
  }
}

impl<E: Engine> R1CSInstance<E> {
  /// Produces a default `R1CSInstance` given an `R1CSShape`
  pub fn default(S: &R1CSShape<E>) -> Self {
    let comm_W = Commitment::<E>::default();
    Self {
      comm_W,
      X: vec![E::Scalar::ZERO; S.num_io],
    }
  }
  /// A method to create an instance object using constituent elements
  pub(crate) fn new(
    S: &R1CSShape<E>,
    comm_W: Commitment<E>,
    X: Vec<E::Scalar>,
  ) -> Result<Self, NovaError> {
    if S.num_io != X.len() {
      Err(NovaError::InvalidInputLength)
    } else {
      Ok(Self { comm_W, X })
    }
  }
}

impl<E: Engine> AbsorbInROTrait<E> for R1CSInstance<E> {
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    self.comm_W.absorb_in_ro(ro);
    for x in &self.X {
      ro.absorb(scalar_as_base::<E>(*x));
    }
  }
}

impl<E: Engine> RelaxedR1CSWitness<E> {
  /// Produces a default `RelaxedR1CSWitness` given an `R1CSShape`
  pub(crate) fn default(S: &R1CSShape<E>) -> Self {
    Self {
      W: vec![E::Scalar::ZERO; S.num_vars],
      r_W: E::Scalar::ZERO,
      E: vec![E::Scalar::ZERO; S.num_cons],
      r_E: E::Scalar::ZERO,
    }
  }

  /// Folds an incoming `R1CSWitness` into the current one
  pub(crate) fn fold(
    &self,
    W2: &R1CSWitness<E>,
    T: &[E::Scalar],
    r_T: &E::Scalar,
    r: &E::Scalar,
  ) -> Result<Self, NovaError> {
    let (W1, r_W1, E1, r_E1) = (&self.W, &self.r_W, &self.E, &self.r_E);
    let (W2, r_W2) = (&W2.W, &W2.r_W);

    if W1.len() != W2.len() {
      return Err(NovaError::InvalidWitnessLength);
    }

    let W = zip_with!((W1.par_iter(), W2), |a, b| *a + *r * *b).collect::<Vec<E::Scalar>>();
    let E = zip_with!((E1.par_iter(), T), |a, b| *a + *r * *b).collect::<Vec<E::Scalar>>();

    let r_W = *r_W1 + *r * r_W2;
    let r_E = *r_E1 + *r * r_T;
    Ok(Self { W, r_W, E, r_E })
  }

  /// Folds an incoming `R1CSWitness` into the current one
  pub(crate) fn fold_relaxed(
    &self,
    W2: &RelaxedR1CSWitness<E>,
    T: &[E::Scalar],
    r_T: &E::Scalar,
    r: &E::Scalar,
  ) -> Result<Self, NovaError> {
    let (W1, r_W1, E1, r_E1) = (&self.W, &self.r_W, &self.E, &self.r_E);
    let (W2, r_W2, E2, r_E2) = (&W2.W, &W2.r_W, &W2.E, &W2.r_E);

    if W1.len() != W2.len() {
      return Err(NovaError::InvalidWitnessLength);
    }
    let r_squared = r.square();

    let W = zip_with!((W1.par_iter(), W2), |a, b| *a + *r * *b).collect::<Vec<E::Scalar>>();
    let E = zip_with!((E1.par_iter(), E2.par_iter(), T), |e1, e2, t| *e1
      + *r * *t
      + r_squared * *e2)
    .collect::<Vec<E::Scalar>>();

    let r_W = *r_W1 + *r * r_W2;
    let r_E = *r_E1 + *r * r_T + *r * *r * *r_E2;

    Ok(Self { W, r_W, E, r_E })
  }

  /// Pads the provided witness to the correct length
  pub(crate) fn pad(&self, S: &R1CSShape<E>) -> Self {
    let mut W = self.W.clone();
    W.extend(vec![E::Scalar::ZERO; S.num_vars - W.len()]);

    let mut E = self.E.clone();
    E.extend(vec![E::Scalar::ZERO; S.num_cons - E.len()]);

    Self {
      W,
      r_W: self.r_W,
      E,
      r_E: self.r_E,
    }
  }

  /// Derandomizes the `R1CSWitness` using a `DerandKey`
  pub(crate) fn derandomize(&self) -> (Self, E::Scalar, E::Scalar) {
    (
      RelaxedR1CSWitness {
        W: self.W.clone(),
        r_W: E::Scalar::ZERO,
        E: self.E.clone(),
        r_E: E::Scalar::ZERO,
      },
      self.r_W,
      self.r_E,
    )
  }
}

impl<E: Engine> RelaxedR1CSInstance<E> {
  /// Produces a default `RelaxedR1CSInstance` given `R1CSGens` and `R1CSShape`
  pub(crate) fn default(_ck: &CommitmentKey<E>, S: &R1CSShape<E>) -> Self {
    let (comm_W, comm_E) = (Commitment::<E>::default(), Commitment::<E>::default());
    Self {
      comm_W,
      comm_E,
      u: E::Scalar::ZERO,
      X: vec![E::Scalar::ZERO; S.num_io],
    }
  }

  /// Folds an incoming `RelaxedR1CSInstance` into the current one
  pub(crate) fn fold(&self, U2: &R1CSInstance<E>, comm_T: &Commitment<E>, r: &E::Scalar) -> Self {
    let (X1, u1, comm_W_1, comm_E_1) =
      (&self.X, &self.u, &self.comm_W.clone(), &self.comm_E.clone());
    let (X2, comm_W_2) = (&U2.X, &U2.comm_W);

    // weighted sum of X, comm_W, comm_E, and u
    let X = zip_with!((X1.par_iter(), X2), |a, b| *a + *r * *b).collect::<Vec<E::Scalar>>();
    let comm_W = *comm_W_1 + *comm_W_2 * *r;
    let comm_E = *comm_E_1 + *comm_T * *r;
    let u = *u1 + *r;

    Self {
      comm_W,
      comm_E,
      X,
      u,
    }
  }

  /// Folds an incoming `RelaxedR1CSInstance` into the current one
  pub(crate) fn fold_relaxed(&self, U2: &Self, comm_T: &Commitment<E>, r: &E::Scalar) -> Self {
    let (X1, u1, comm_W_1, comm_E_1) =
      (&self.X, &self.u, &self.comm_W.clone(), &self.comm_E.clone());
    let (X2, u2, comm_W_2, comm_E_2) = (&U2.X, &U2.u, &U2.comm_W, &U2.comm_E);

    // weighted sum of X, comm_W, comm_E, and u
    let X = zip_with!((X1.par_iter(), X2), |a, b| *a + *r * *b).collect::<Vec<E::Scalar>>();
    let comm_W = *comm_W_1 + *comm_W_2 * *r;

    let r_squared = r.square();
    let comm_E = *comm_E_1 + *comm_T * *r + *comm_E_2 * r_squared;
    let u = *u1 + *r * u2;

    Self {
      comm_W,
      comm_E,
      X,
      u,
    }
  }

  /// Derandomizes the `RelaxedR1CSInstance` using a `DerandKey`
  pub(crate) fn derandomize(
    &self,
    dk: &DerandKey<E>,
    r_W: &E::Scalar,
    r_E: &E::Scalar,
  ) -> RelaxedR1CSInstance<E> {
    RelaxedR1CSInstance {
      comm_W: CE::<E>::derandomize(dk, &self.comm_W, r_W),
      comm_E: CE::<E>::derandomize(dk, &self.comm_E, r_E),
      X: self.X.clone(),
      u: self.u,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for RelaxedR1CSInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_W.to_transcript_bytes(),
      self.comm_E.to_transcript_bytes(),
      self.u.to_transcript_bytes(),
      self.X.as_slice().to_transcript_bytes(),
    ]
    .concat()
  }
}

impl<E: Engine> AbsorbInROTrait<E> for RelaxedR1CSInstance<E> {
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    self.comm_W.absorb_in_ro(ro);
    self.comm_E.absorb_in_ro(ro);
    ro.absorb(scalar_as_base::<E>(self.u));

    // absorb each element of self.X in bignum format
    for x in &self.X {
      let limbs: Vec<E::Scalar> = nat_to_limbs(&f_to_nat(x), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
      for limb in limbs {
        ro.absorb(scalar_as_base::<E>(limb));
      }
    }
  }
}

#[cfg(test)]
pub(crate) mod tests {
  use ff::Field;

  use super::*;
  use crate::{
    provider::{Bn256EngineKZG, PallasEngine, Secp256k1Engine},
    r1cs::sparse::SparseMatrix,
    traits::Engine,
  };

  pub(crate) fn tiny_r1cs<E: Engine>(num_vars: usize) -> R1CSShape<E> {
    let one = <E::Scalar as Field>::ONE;
    let (num_cons, num_vars, num_io, A, B, C) = {
      let num_cons = 4;
      let num_io = 2;

      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      // The R1CS for this problem consists of the following constraints:
      // `I0 * I0 - Z0 = 0`
      // `Z0 * I0 - Z1 = 0`
      // `(Z1 + I0) * 1 - Z2 = 0`
      // `(Z2 + 5) * 1 - I1 = 0`

      // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
      // constraint and a column for every entry in z = (vars, u, inputs)
      // An R1CS instance is satisfiable iff:
      // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
      let mut A: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut B: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut C: Vec<(usize, usize, E::Scalar)> = Vec::new();

      // constraint 0 entries in (A,B,C)
      // `I0 * I0 - Z0 = 0`
      A.push((0, num_vars + 1, one));
      B.push((0, num_vars + 1, one));
      C.push((0, 0, one));

      // constraint 1 entries in (A,B,C)
      // `Z0 * I0 - Z1 = 0`
      A.push((1, 0, one));
      B.push((1, num_vars + 1, one));
      C.push((1, 1, one));

      // constraint 2 entries in (A,B,C)
      // `(Z1 + I0) * 1 - Z2 = 0`
      A.push((2, 1, one));
      A.push((2, num_vars + 1, one));
      B.push((2, num_vars, one));
      C.push((2, 2, one));

      // constraint 3 entries in (A,B,C)
      // `(Z2 + 5) * 1 - I1 = 0`
      A.push((3, 2, one));
      A.push((3, num_vars, one + one + one + one + one));
      B.push((3, num_vars, one));
      C.push((3, num_vars + 2, one));

      (num_cons, num_vars, num_io, A, B, C)
    };

    // create a shape object
    let rows = num_cons;
    let cols = num_vars + num_io + 1;

    R1CSShape::new(
      num_cons,
      num_vars,
      num_io,
      (0, 0),
      SparseMatrix::new(&A, rows, cols),
      SparseMatrix::new(&B, rows, cols),
      SparseMatrix::new(&C, rows, cols),
    )
    .unwrap()
  }

  fn test_pad_tiny_r1cs_with<E: Engine>() {
    let padded_r1cs = tiny_r1cs::<E>(3).pad();
    assert!(padded_r1cs.is_regular_shape());

    let expected_r1cs = tiny_r1cs::<E>(4);

    assert_eq!(padded_r1cs, expected_r1cs);
  }

  #[test]
  fn test_pad_tiny_r1cs() {
    test_pad_tiny_r1cs_with::<PallasEngine>();
    test_pad_tiny_r1cs_with::<Bn256EngineKZG>();
    test_pad_tiny_r1cs_with::<Secp256k1Engine>();
  }
}
