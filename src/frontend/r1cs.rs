//! Support for generating R1CS using bellpepper.

#![allow(non_snake_case)]

use super::{shape_cs::ShapeCS, solver::SatisfyingAssignment, test_shape_cs::TestShapeCS};
use crate::{
  errors::NovaError,
  frontend::{Index, LinearCombination},
  r1cs::{
    commitment_key,
    split::{SplitR1CSInstance, SplitR1CSWitness},
    CommitmentKeyHint, R1CSInstance, R1CSShape, R1CSWitness, SparseMatrix,
  },
  traits::Engine,
  CommitmentKey,
};
use ff::PrimeField;

/// `NovaWitness` provide a method for acquiring an `R1CSInstance` and `R1CSWitness` from implementers.
pub trait NovaWitness<E: Engine> {
  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
  ) -> Result<(R1CSInstance<E>, R1CSWitness<E>), NovaError>;

  /// Return an instance and witness, given a shape and ck.
  fn split_r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
  ) -> Result<(SplitR1CSInstance<E>, SplitR1CSWitness<E>), NovaError>;
}

/// `NovaShape` provides methods for acquiring `R1CSShape` and `CommitmentKey` from implementers.
pub trait NovaShape<E: Engine> {
  /// Return an appropriate `R1CSShape` and `CommitmentKey` structs.
  /// A `CommitmentKeyHint` should be provided to help guide the construction of the `CommitmentKey`.
  /// This parameter is documented in `r1cs::R1CS::commitment_key`.
  fn r1cs_shape_and_key(&self, ck_hint: &CommitmentKeyHint<E>) -> (R1CSShape<E>, CommitmentKey<E>) {
    let S = self.r1cs_shape();
    let ck = commitment_key(&S, ck_hint);

    (S, ck)
  }
  /// Return an appropriate `R1CSShape` and `CommitmentKey` structs.
  /// A `CommitmentKeyHint` should be provided to help guide the construction of the `CommitmentKey`.
  /// This parameter is documented in `r1cs::R1CS::commitment_key`.
  fn r1cs_shape(&self) -> R1CSShape<E>;
}

impl<E: Engine> NovaWitness<E> for SatisfyingAssignment<E> {
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
  ) -> Result<(R1CSInstance<E>, R1CSWitness<E>), NovaError> {
    let W = R1CSWitness::<E>::new(shape, self.aux_assignment().to_vec())?;
    let X = &self.input_assignment()[1..];
    let comm_W = W.commit_at(
      ck,
      self.precommitted_assignment().len() + self.precommitted1_assignment().len(),
    );
    let instance = R1CSInstance::<E>::new(shape, comm_W, X.to_vec())?;
    Ok((instance, W))
  }

  fn split_r1cs_instance_and_witness(
    &self,
    S: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
  ) -> Result<(SplitR1CSInstance<E>, SplitR1CSWitness<E>), NovaError> {
    let (aux_U, aux_W) = self.r1cs_instance_and_witness(S, ck)?;
    let pre_committed_witness = (
      self.precommitted_assignment().to_vec(),
      self.precommitted1_assignment().to_vec(),
    );
    let W = SplitR1CSWitness::new(aux_W, pre_committed_witness);
    let pre_commits = W.commit(ck);
    let instance = SplitR1CSInstance::new(aux_U, pre_commits);
    Ok((instance, W))
  }
}

macro_rules! impl_nova_shape {
  ( $name:ident) => {
    impl<E: Engine> NovaShape<E> for $name<E>
    where
      E::Scalar: PrimeField,
    {
      fn r1cs_shape(&self) -> R1CSShape<E> {
        let mut A = SparseMatrix::<E::Scalar>::empty();
        let mut B = SparseMatrix::<E::Scalar>::empty();
        let mut C = SparseMatrix::<E::Scalar>::empty();

        let mut num_cons_added = 0;
        let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);
        let num_inputs = self.num_inputs();
        let num_constraints = self.num_constraints();
        let num_vars = self.num_aux();
        let num_precommitted = self.num_precommitted();
        let num_precommitted1 = self.num_precommitted1();

        for constraint in self.constraints.iter() {
          add_constraint(
            &mut X,
            num_vars,
            num_precommitted,
            num_precommitted1,
            &constraint.0,
            &constraint.1,
            &constraint.2,
          );
        }
        assert_eq!(num_cons_added, num_constraints);
        let num_cols = num_vars + num_inputs + num_precommitted + num_precommitted1;
        A.cols = num_cols;
        B.cols = num_cols;
        C.cols = num_cols;

        // Don't count One as an input for shape's purposes.
        let S = R1CSShape::new(
          num_constraints,
          num_vars,
          num_inputs - 1,
          (num_precommitted, num_precommitted1),
          A,
          B,
          C,
        )
        .unwrap();

        S
      }
    }
  };
}

impl_nova_shape!(ShapeCS);
impl_nova_shape!(TestShapeCS);

fn add_constraint<S: PrimeField>(
  X: &mut (
    &mut SparseMatrix<S>,
    &mut SparseMatrix<S>,
    &mut SparseMatrix<S>,
    &mut usize,
  ),
  num_vars: usize,
  num_precommitted: usize,
  num_precommitted1: usize,
  a_lc: &LinearCombination<S>,
  b_lc: &LinearCombination<S>,
  c_lc: &LinearCombination<S>,
) {
  let (A, B, C, nn) = X;
  let n = **nn;
  assert_eq!(n + 1, A.indptr.len(), "A: invalid shape");
  assert_eq!(n + 1, B.indptr.len(), "B: invalid shape");
  assert_eq!(n + 1, C.indptr.len(), "C: invalid shape");

  let add_constraint_component = |index: Index, coeff: &S, M: &mut SparseMatrix<S>| {
    // we add constraints to the matrix only if the associated coefficient is non-zero
    if *coeff != S::ZERO {
      match index {
        Index::Input(idx) => {
          // Inputs come last, with input 0, representing 'one',
          // at position num_vars within the witness vector.
          let idx = idx + num_vars + num_precommitted + num_precommitted1;
          M.data.push(*coeff);
          M.indices.push(idx);
        }
        Index::Aux(idx) => {
          let idx = idx + num_precommitted + num_precommitted1;
          M.data.push(*coeff);
          M.indices.push(idx);
        }
        Index::Precommitted(idx) => {
          M.data.push(*coeff);
          M.indices.push(idx);
        }
        Index::Precommitted1(idx) => {
          let idx = idx + num_precommitted;
          M.data.push(*coeff);
          M.indices.push(idx);
        }
      }
    }
  };

  for (index, coeff) in a_lc.iter() {
    add_constraint_component(index.0, coeff, A);
  }
  A.indptr.push(A.indices.len());

  for (index, coeff) in b_lc.iter() {
    add_constraint_component(index.0, coeff, B)
  }
  B.indptr.push(B.indices.len());

  for (index, coeff) in c_lc.iter() {
    add_constraint_component(index.0, coeff, C)
  }
  C.indptr.push(C.indices.len());

  **nn += 1;
}
