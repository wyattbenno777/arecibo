use crate::{
  cyclefold::util::absorb_primary_commitment,
  r1cs::split::SplitR1CSInstance,
  traits::{CurveCycleEquipped, Dual, Engine, ROTrait},
};

pub(crate) fn absorb_split_instance<E>(u: &SplitR1CSInstance<E>, ro: &mut <Dual<E> as Engine>::RO)
where
  E: CurveCycleEquipped,
{
  absorb_primary_commitment::<E, Dual<E>>(&u.aux.comm_W, ro);
  for x in &u.aux.X {
    ro.absorb(*x);
  }
  absorb_primary_commitment::<E, Dual<E>>(&u.pre_committed.0, ro);
  absorb_primary_commitment::<E, Dual<E>>(&u.pre_committed.1, ro);
}
