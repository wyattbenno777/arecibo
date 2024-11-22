use crate::{
  cyclefold::util::absorb_primary_commitment,
  r1cs::RelaxedR1CSInstance,
  traits::{CurveCycleEquipped, Dual, ROTrait},
};

pub(crate) fn absorb_U<E1>(U: &RelaxedR1CSInstance<E1>, ro: &mut impl ROTrait<E1::Scalar, E1::Base>)
where
  E1: CurveCycleEquipped,
{
  absorb_primary_commitment::<E1, Dual<E1>>(&U.comm_W, ro);
  absorb_primary_commitment::<E1, Dual<E1>>(&U.comm_E, ro);
  for x in &U.X {
    ro.absorb(*x);
  }
  ro.absorb(U.u);
}
