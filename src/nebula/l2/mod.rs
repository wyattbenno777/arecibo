use crate::errors::NovaError;
use crate::gadgets::scalar_as_base;
use crate::nebula::rs::{PublicParams, RecursiveSNARK};
use crate::r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
use crate::traits::commitment::CommitmentTrait;
use crate::traits::snark::default_ck_hint;
use crate::traits::CurveCycleEquipped;
use crate::Commitment;
use ff::Field;
use nifs::RelaxedNIFS;
use serde::{Deserialize, Serialize};
use utils::RelaxedFoldingData;

use final_circuit::{FinalCircuit, FinalCircuitInputs, FinalCircuitParams};

mod final_circuit;
mod gadgets;
mod nifs;
#[cfg(test)]
mod tests;
mod utils;

fn public_params<E1>(node_pp: &PublicParams<E1>) -> PublicParams<E1>
where
  E1: CurveCycleEquipped,
{
  let circuit_params = FinalCircuitParams::from(&node_pp.augmented_circuit_params);
  let final_circuit = FinalCircuit::<E1>::new(
    &circuit_params,
    node_pp.ro_consts_circuit_primary.clone(),
    None,
  );
  PublicParams::setup(&final_circuit, &*default_ck_hint(), &*default_ck_hint())
}
#[derive(Debug, Clone, Deserialize, Serialize)]
struct L2<E>
where
  E: CurveCycleEquipped,
{
  r_U: RelaxedR1CSInstance<E>,
  r_W: RelaxedR1CSWitness<E>,
  rs: RecursiveSNARK<E>,
  IC_i: E::Scalar,
  i: usize,
}

impl<E> L2<E>
where
  E: CurveCycleEquipped,
{
  /// Create a new instance of [`L2`]
  pub fn new(
    final_pp: &PublicParams<E>,
    node_pp: &PublicParams<E>,
    node_rs: &RecursiveSNARK<E>,
  ) -> Result<L2<E>, NovaError> {
    let r_U = RelaxedR1CSInstance::default(
      &node_pp.ck_primary,
      &node_pp.circuit_shape_primary.r1cs_shape,
    );
    let r_W = RelaxedR1CSWitness::default(&node_pp.circuit_shape_primary.r1cs_shape);

    let node_pp_digest = node_pp.digest();
    let (l_U, l_W) = node_rs.U_W();
    let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
      &node_pp.ck_primary,
      &node_pp.ro_consts_primary,
      &node_pp_digest,
      &node_pp.circuit_shape_primary.r1cs_shape,
      &r_U,
      &r_W,
      l_U,
      l_W,
    )?;

    let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

    let data_p = RelaxedFoldingData::new(r_U.clone(), l_U.clone(), comm_T);

    // Do calculation's outside circuit, to be passed in as advice to F'
    let r_squared = r * r;
    let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
    let W_new = r_U.comm_W + l_U.comm_W * r;

    let final_circuit_inputs = FinalCircuitInputs::<E>::new(
      Some(scalar_as_base::<E>(node_pp_digest)),
      Some(data_p),
      Some(E_new),
      Some(W_new),
    );

    let circuit_params = FinalCircuitParams::from(&node_pp.augmented_circuit_params);

    let final_circuit = FinalCircuit::new(
      &circuit_params,
      node_pp.ro_consts_circuit_primary.clone(),
      Some(final_circuit_inputs),
    );

    let z0 = vec![E::Scalar::ZERO];
    let mut IC_i = E::Scalar::ZERO;

    let mut rs = RecursiveSNARK::new(final_pp, &final_circuit, &z0)?;

    rs.prove_step(final_pp, &final_circuit, IC_i)?;
    IC_i = rs.increment_commitment(final_pp, &final_circuit);

    Ok(Self {
      r_U,
      r_W,
      rs,
      IC_i,
      i: 0,
    })
  }

  /// updates the provided [`L2`] by executing a step of the incremental computation
  #[tracing::instrument(skip_all, name = "Layer2::prove_step")]
  pub fn prove_step(
    &mut self,
    final_pp: &PublicParams<E>,
    node_pp: &PublicParams<E>,
    node_rs: &RecursiveSNARK<E>,
  ) -> Result<(), NovaError> {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }
    let node_pp_digest = node_pp.digest();
    let (l_U, l_W) = node_rs.U_W();
    let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
      &node_pp.ck_primary,
      &node_pp.ro_consts_primary,
      &node_pp_digest,
      &node_pp.circuit_shape_primary.r1cs_shape,
      &self.r_U,
      &self.r_W,
      l_U,
      l_W,
    )?;

    let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

    let data_p = RelaxedFoldingData::new(self.r_U.clone(), l_U.clone(), comm_T);

    // Do calculation's outside circuit, to be passed in as advice to F'
    let r_squared = r * r;
    let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
    let W_new = self.r_U.comm_W + l_U.comm_W * r;

    let final_circuit_inputs = FinalCircuitInputs::<E>::new(
      Some(scalar_as_base::<E>(node_pp_digest)),
      Some(data_p),
      Some(E_new),
      Some(W_new),
    );

    let circuit_params = FinalCircuitParams::from(&node_pp.augmented_circuit_params);

    let final_circuit = FinalCircuit::new(
      &circuit_params,
      node_pp.ro_consts_circuit_primary.clone(),
      Some(final_circuit_inputs),
    );

    self.rs.prove_step(final_pp, &final_circuit, self.IC_i)?;
    self.IC_i = self.rs.increment_commitment(final_pp, &final_circuit);
    self.r_U = r_U;
    self.r_W = r_W;

    Ok(())
  }

  /// Verifies the [`L2`] instance
  pub fn verify(&self, final_pp: &PublicParams<E>) -> Result<(), NovaError> {
    self
      .rs
      .verify(final_pp, self.rs.num_steps(), &[E::Scalar::ZERO], self.IC_i)?;

    Ok(())
  }
}
