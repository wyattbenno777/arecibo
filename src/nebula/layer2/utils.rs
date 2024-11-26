use serde::{Deserialize, Serialize};

use crate::{
  cyclefold::util::absorb_primary_commitment,
  r1cs::RelaxedR1CSInstance,
  traits::{CurveCycleEquipped, Dual, Engine, ROTrait},
  Commitment,
};

pub(crate) fn absorb_U<E1>(U: &RelaxedR1CSInstance<E1>, ro: &mut impl ROTrait<E1::Scalar, E1::Base>)
where
  E1: CurveCycleEquipped,
{
  absorb_primary_commitment::<E1, Dual<E1>>(&U.comm_W, ro);
  absorb_primary_commitment::<E1, Dual<E1>>(&U.comm_E, ro);
  ro.absorb(U.u);

  for x in &U.X {
    ro.absorb(*x);
  }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub(crate) struct RelaxedFoldingData<E: Engine> {
  pub U1: RelaxedR1CSInstance<E>,
  pub U2: RelaxedR1CSInstance<E>,
  pub T: Commitment<E>,
}

impl<E: Engine> RelaxedFoldingData<E> {
  pub fn new(U1: RelaxedR1CSInstance<E>, U2: RelaxedR1CSInstance<E>, T: Commitment<E>) -> Self {
    Self { U1, U2, T }
  }
}
