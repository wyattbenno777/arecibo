use std::marker::PhantomData;

use bellpepper_core::{
  num::{self, AllocatedNum},
  ConstraintSystem, SynthesisError,
};

use crate::{nebula::rs::StepCircuit, traits::CurveCycleEquipped, Commitment, CommitmentKey};

#[derive(Clone, Debug)]
pub struct VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  _engine: PhantomData<E>,
}

impl<E> StepCircuit<E::Scalar> for VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  fn arity(&self) -> usize {
    todo!()
  }

  fn non_deterministic_advice(&self) -> Vec<E::Scalar> {
    todo!()
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[num::AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    todo!()
  }
}
