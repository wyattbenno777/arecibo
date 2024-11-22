use abomonation_derive::Abomonation;
use serde::{Deserialize, Serialize};

use crate::{
  cyclefold::util::FoldingData,
  traits::{CurveCycleEquipped, Dual, ROConstantsCircuit},
  Commitment,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Abomonation)]
pub struct FinalCircuitParams {
  limb_width: usize,
  n_limbs: usize,
}

impl FinalCircuitParams {
  pub const fn new(limb_width: usize, n_limbs: usize) -> Self {
    Self {
      limb_width,
      n_limbs,
    }
  }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FinalCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  data_F: Option<FoldingData<E1>>,

  E_F: Option<Commitment<E1>>,
  W_F: Option<Commitment<E1>>,
}

impl<E1> FinalCircuitInputs<E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    data_F: Option<FoldingData<E1>>,
    E_F: Option<Commitment<E1>>,
    W_F: Option<Commitment<E1>>,
  ) -> Self {
    Self { data_F, E_F, W_F }
  }
}

pub struct FinalCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  params: &'a FinalCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E1>>,
  inputs: Option<FinalCircuitInputs<E1>>,
}

impl<'a, E1> FinalCircuit<'a, E1>
where
  E1: CurveCycleEquipped,
{
  pub fn new(
    params: &'a FinalCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E1>>,
    inputs: Option<FinalCircuitInputs<E1>>,
  ) -> Self {
    Self {
      params,
      ro_consts,
      inputs,
    }
  }
}
