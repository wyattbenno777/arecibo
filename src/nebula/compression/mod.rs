//! Implements components to enable the compression-step for IVC proofs

use std::marker::PhantomData;

use crate::traits::CurveCycleEquipped;

use super::traits::{Layer1PPTrait, Layer1RSTrait};

/// A SNARK that proves the knowledge of a valid Nebula proof
pub struct CompressedSNARK<E>
where
  E: CurveCycleEquipped,
{
  _engine: PhantomData<E>,
}

/// A type that holds the prover key for [`CompressedSNARK`]
#[derive(Debug)]
pub struct ProverKey<E>
where
  E: CurveCycleEquipped,
{
  _engine: PhantomData<E>,
}

impl<E> CompressedSNARK<E>
where
  E: CurveCycleEquipped,
{
  /// Create a new [`CompressedSNARK`]
  pub fn prove(pp: &impl Layer1PPTrait<E>, pk: &ProverKey<E>, rs: &impl Layer1RSTrait<E>) {}
}
