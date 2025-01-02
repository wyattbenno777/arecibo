//! A CycleFold influenced NIFS for folding IVC proofs.

use std::marker::PhantomData;

use crate::traits::CurveCycleEquipped;

/// A non-interactive folding scheme for IVC proofs.
pub struct NIFS<E>
where
  E: CurveCycleEquipped,
{
  _p: PhantomData<E>,
}
