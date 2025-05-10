//! Support for generating R1CS shape using bellpepper.

use crate::{
  frontend::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable},
  traits::Engine,
};
use ff::PrimeField;

use super::Split;

/// `ShapeCS` is a `ConstraintSystem` for creating `R1CSShape`s for a circuit.
pub struct ShapeCS<E: Engine>
where
  E::Scalar: PrimeField,
{
  /// All constraints added to the `ShapeCS`.
  pub constraints: Vec<(
    LinearCombination<E::Scalar>,
    LinearCombination<E::Scalar>,
    LinearCombination<E::Scalar>,
  )>,
  inputs: usize,
  aux: usize,
  precommitted: usize,
  precommitted1: usize,
}

impl<E: Engine> ShapeCS<E> {
  /// Create a new, default `ShapeCS`,
  pub fn new() -> Self {
    ShapeCS::default()
  }

  /// Returns the number of constraints defined for this `ShapeCS`.
  pub fn num_constraints(&self) -> usize {
    self.constraints.len()
  }

  /// Returns the number of inputs defined for this `ShapeCS`.
  pub fn num_inputs(&self) -> usize {
    self.inputs
  }

  /// Returns the number of aux inputs defined for this `ShapeCS`.
  pub fn num_aux(&self) -> usize {
    self.aux
  }

  /// Returns the number of precommitted inputs defined for this `ShapeCS`.
  pub fn num_precommitted(&self) -> usize {
    self.precommitted
  }

  /// Returns the number of precommitted1 inputs defined for this `ShapeCS`.
  pub fn num_precommitted1(&self) -> usize {
    self.precommitted1
  }
}

impl<E: Engine> Default for ShapeCS<E> {
  fn default() -> Self {
    ShapeCS {
      constraints: vec![],
      inputs: 1,
      aux: 0,
      precommitted: 0,
      precommitted1: 0,
    }
  }
}

impl<E: Engine> ConstraintSystem<E::Scalar> for ShapeCS<E> {
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.aux += 1;

    Ok(Variable::new_unchecked(Index::Aux(self.aux - 1)))
  }

  fn alloc_precommitted<F, A, AR>(
    &mut self,
    _annotation: A,
    _f: F,
    idx: Split,
  ) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    match idx {
      Split::ZERO => alloc_precommitted_generic(&mut self.precommitted, Index::Precommitted),
      Split::ONE => alloc_precommitted_generic(&mut self.precommitted1, Index::Precommitted1),
    }
  }

  fn alloc_input<F, A, AR>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.inputs += 1;

    Ok(Variable::new_unchecked(Index::Input(self.inputs - 1)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LB: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LC: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
  {
    let a = a(LinearCombination::zero());
    let b = b(LinearCombination::zero());
    let c = c(LinearCombination::zero());

    self.constraints.push((a, b, c));
  }

  fn push_namespace<NR, N>(&mut self, _name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
  }

  fn pop_namespace(&mut self) {}

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }
}

fn alloc_precommitted_generic(
  num: &mut usize,
  index_fn: impl FnOnce(usize) -> Index,
) -> Result<Variable, SynthesisError> {
  *num += 1;
  Ok(Variable::new_unchecked(index_fn(*num - 1)))
}
