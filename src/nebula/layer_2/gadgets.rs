use std::marker::PhantomData;

use crate::traits::CurveCycleEquipped;

/// Verifer gadget used to fold IVC proofs.
pub struct NIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  _engine: PhantomData<E>,
}
