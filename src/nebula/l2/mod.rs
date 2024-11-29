use serde::{Deserialize, Serialize};
use utils::RelaxedFoldingData;

use crate::errors::NovaError;
use crate::gadgets::scalar_as_base;
use crate::r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
use crate::traits::commitment::CommitmentTrait;
use crate::traits::commitment::Len;
use crate::{
  nebula::rs::{PublicParams, RecursiveSNARK},
  traits::{snark::default_ck_hint, CurveCycleEquipped},
  R1CSWithArity,
};
use crate::{Commitment, CommitmentKey};
use ff::Field;
use final_circuit::{FinalCircuit, FinalCircuitInputs};
use gadgets::NIFSVerifierCircuitInputs;
use nifs::RelaxedNIFS;

mod final_circuit;
mod gadgets;
mod nifs;
#[cfg(test)]
mod tests;
mod utils;

pub trait Layer1RSTrait<E>
where
  E: CurveCycleEquipped,
{
  fn F(&self) -> &RecursiveSNARK<E>;
  fn ops(&self) -> &RecursiveSNARK<E>;
  fn scan(&self) -> &RecursiveSNARK<E>;
}

pub trait Layer1PP<E: CurveCycleEquipped> {
  fn into_parts(self) -> (PublicParams<E>, PublicParams<E>, PublicParams<E>);
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct L2AggregationPublicParams<E>
where
  E: CurveCycleEquipped,
{
  pp: PublicParams<E>,
  circuit_shape_F: R1CSWithArity<E>,
  circuit_shape_ops: R1CSWithArity<E>,
  circuit_shape_scan: R1CSWithArity<E>,
  digest_F: E::Scalar,
  digest_ops: E::Scalar,
  digest_scan: E::Scalar,
  ck: CommitmentKey<E>,
}

impl<E> L2AggregationPublicParams<E>
where
  E: CurveCycleEquipped,
{
  pub fn setup<'a, PP1>(node_pp: PP1) -> Self
  where
    PP1: Layer1PP<E>,
  {
    let (pp_F, pp_ops, pp_scan) = node_pp.into_parts();
    let params = pp_F.augmented_circuit_params.clone();
    let final_circuit =
      FinalCircuit::<E>::new(&params, pp_F.ro_consts_circuit_primary.clone(), None);
    let (circuit_shape_F, ck_F, digest_F) = pp_F.into_shape_ck_digest();
    let (circuit_shape_ops, ck_ops, digest_ops) = pp_ops.into_shape_ck_digest();
    let (circuit_shape_scan, ck_scan, digest_scan) = pp_scan.into_shape_ck_digest();

    // choose ck with biggest size
    let ck = {
      let mut ck = ck_F;
      if ck_ops.length() > ck.length() {
        ck = ck_ops;
      }
      if ck_scan.length() > ck.length() {
        ck = ck_scan;
      }
      ck
    };

    let pp = PublicParams::setup(&final_circuit, &*default_ck_hint(), &*default_ck_hint());
    Self {
      pp,
      circuit_shape_F,
      circuit_shape_ops,
      circuit_shape_scan,
      digest_F,
      digest_ops,
      digest_scan,
      ck,
    }
  }
}

pub struct L2AggregationEngine<E>
where
  E: CurveCycleEquipped,
{
  r_U_F: RelaxedR1CSInstance<E>,
  r_W_F: RelaxedR1CSWitness<E>,
  r_U_ops: RelaxedR1CSInstance<E>,
  r_W_ops: RelaxedR1CSWitness<E>,
  r_U_scan: RelaxedR1CSInstance<E>,
  r_W_scan: RelaxedR1CSWitness<E>,
  rs: RecursiveSNARK<E>,
  IC_i: E::Scalar,
  i: usize,
}

impl<E> L2AggregationEngine<E>
where
  E: CurveCycleEquipped,
{
  pub fn new<RS1>(pp: &L2AggregationPublicParams<E>, l1_rs: &RS1) -> Result<Self, NovaError>
  where
    RS1: Layer1RSTrait<E>,
  {
    let (inputs_F, r_U_F, r_W_F) = {
      let F_shape = &pp.circuit_shape_F.r1cs_shape;
      let r_U = RelaxedR1CSInstance::default(&pp.ck, F_shape);
      let r_W = RelaxedR1CSWitness::default(F_shape);
      let rs_F = l1_rs.F();
      let (l_U, l_W) = rs_F.U_W();

      let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
        &pp.ck,
        &pp.pp.ro_consts_primary,
        &pp.digest_F,
        F_shape,
        &r_U,
        &r_W,
        l_U,
        l_W,
      )?;

      let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

      let data = RelaxedFoldingData::new(r_U.clone(), l_U.clone(), comm_T);

      // Do calculation's outside circuit, to be passed in as advice to F'
      let r_squared = r * r;
      let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
      let W_new = r_U.comm_W + l_U.comm_W * r;

      (
        NIFSVerifierCircuitInputs::<E>::new(
          Some(scalar_as_base::<E>(pp.digest_F)),
          Some(data),
          Some(E_new),
          Some(W_new),
        ),
        r_U,
        r_W,
      )
    };

    let (inputs_ops, r_U_ops, r_W_ops) = {
      let ops_shape = &pp.circuit_shape_ops.r1cs_shape;
      let r_U = RelaxedR1CSInstance::default(&pp.ck, ops_shape);
      let r_W = RelaxedR1CSWitness::default(ops_shape);
      let rs_ops = l1_rs.ops();
      let (l_U, l_W) = rs_ops.U_W();

      let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
        &pp.ck,
        &pp.pp.ro_consts_primary,
        &pp.digest_ops,
        ops_shape,
        &r_U,
        &r_W,
        l_U,
        l_W,
      )?;

      let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

      let data = RelaxedFoldingData::new(r_U.clone(), l_U.clone(), comm_T);

      // Do calculation's outside circuit, to be passed in as advice to F'
      let r_squared = r * r;
      let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
      let W_new = r_U.comm_W + l_U.comm_W * r;

      (
        NIFSVerifierCircuitInputs::<E>::new(
          Some(scalar_as_base::<E>(pp.digest_ops)),
          Some(data),
          Some(E_new),
          Some(W_new),
        ),
        r_U,
        r_W,
      )
    };

    let (inputs_scan, r_U_scan, r_W_scan) = {
      let scan_shape = &pp.circuit_shape_scan.r1cs_shape;
      let r_U = RelaxedR1CSInstance::default(&pp.ck, scan_shape);
      let r_W = RelaxedR1CSWitness::default(scan_shape);
      let rs_scan = l1_rs.scan();
      let (l_U, l_W) = rs_scan.U_W();

      let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
        &pp.ck,
        &pp.pp.ro_consts_primary,
        &pp.digest_scan,
        scan_shape,
        &r_U,
        &r_W,
        l_U,
        l_W,
      )?;

      let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

      let data = RelaxedFoldingData::new(r_U.clone(), l_U.clone(), comm_T);

      // Do calculation's outside circuit, to be passed in as advice to F'
      let r_squared = r * r;
      let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
      let W_new = r_U.comm_W + l_U.comm_W * r;

      (
        NIFSVerifierCircuitInputs::<E>::new(
          Some(scalar_as_base::<E>(pp.digest_scan)),
          Some(data),
          Some(E_new),
          Some(W_new),
        ),
        r_U,
        r_W,
      )
    };

    let final_circuit_inputs =
      FinalCircuitInputs::new(Some(inputs_F), Some(inputs_ops), Some(inputs_scan));

    let final_circuit = FinalCircuit::<E>::new(
      &pp.pp.augmented_circuit_params,
      pp.pp.ro_consts_circuit_primary.clone(),
      Some(final_circuit_inputs),
    );

    let z0 = vec![E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO];
    let mut IC_i = E::Scalar::ZERO;

    let mut rs = RecursiveSNARK::new(&pp.pp, &final_circuit, &z0)?;

    rs.prove_step(&pp.pp, &final_circuit, IC_i)?;
    IC_i = rs.increment_commitment(&pp.pp, &final_circuit);

    Ok(Self {
      r_U_F,
      r_W_F,
      r_U_ops,
      r_W_ops,
      r_U_scan,
      r_W_scan,
      rs,
      IC_i,
      i: 0,
    })
  }

  /// updates the provided [`L2`] by executing a step of the incremental computation
  #[tracing::instrument(skip_all, name = "Layer2::prove_step")]
  pub fn prove_step<RS1>(
    &mut self,
    pp: &L2AggregationPublicParams<E>,
    l1_rs: &RS1,
  ) -> Result<(), NovaError>
  where
    RS1: Layer1RSTrait<E>,
  {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    let (inputs_F, r_U_F, r_W_F) = {
      let F_shape = &pp.circuit_shape_F.r1cs_shape;
      let r_U = RelaxedR1CSInstance::default(&pp.ck, F_shape);
      let r_W = RelaxedR1CSWitness::default(F_shape);
      let rs_F = l1_rs.F();
      let (l_U, l_W) = rs_F.U_W();

      let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
        &pp.ck,
        &pp.pp.ro_consts_primary,
        &pp.digest_F,
        F_shape,
        &r_U,
        &r_W,
        l_U,
        l_W,
      )?;

      let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

      let data = RelaxedFoldingData::new(r_U.clone(), l_U.clone(), comm_T);

      // Do calculation's outside circuit, to be passed in as advice to F'
      let r_squared = r * r;
      let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
      let W_new = r_U.comm_W + l_U.comm_W * r;

      (
        NIFSVerifierCircuitInputs::<E>::new(
          Some(scalar_as_base::<E>(pp.digest_F)),
          Some(data),
          Some(E_new),
          Some(W_new),
        ),
        r_U,
        r_W,
      )
    };

    let (inputs_ops, r_U_ops, r_W_ops) = {
      let ops_shape = &pp.circuit_shape_ops.r1cs_shape;
      let r_U = RelaxedR1CSInstance::default(&pp.ck, ops_shape);
      let r_W = RelaxedR1CSWitness::default(ops_shape);
      let rs_ops = l1_rs.ops();
      let (l_U, l_W) = rs_ops.U_W();

      let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
        &pp.ck,
        &pp.pp.ro_consts_primary,
        &pp.digest_ops,
        ops_shape,
        &r_U,
        &r_W,
        l_U,
        l_W,
      )?;

      let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

      let data = RelaxedFoldingData::new(r_U.clone(), l_U.clone(), comm_T);

      // Do calculation's outside circuit, to be passed in as advice to F'
      let r_squared = r * r;
      let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
      let W_new = r_U.comm_W + l_U.comm_W * r;

      (
        NIFSVerifierCircuitInputs::<E>::new(
          Some(scalar_as_base::<E>(pp.digest_ops)),
          Some(data),
          Some(E_new),
          Some(W_new),
        ),
        r_U,
        r_W,
      )
    };

    let (inputs_scan, r_U_scan, r_W_scan) = {
      let scan_shape = &pp.circuit_shape_scan.r1cs_shape;
      let r_U = RelaxedR1CSInstance::default(&pp.ck, scan_shape);
      let r_W = RelaxedR1CSWitness::default(scan_shape);
      let rs_scan = l1_rs.scan();
      let (l_U, l_W) = rs_scan.U_W();

      let (nifs_primary, (r_U, r_W), r) = RelaxedNIFS::<E>::prove(
        &pp.ck,
        &pp.pp.ro_consts_primary,
        &pp.digest_scan,
        scan_shape,
        &r_U,
        &r_W,
        l_U,
        l_W,
      )?;

      let comm_T = Commitment::<E>::decompress(&nifs_primary.comm_T)?;

      let data = RelaxedFoldingData::new(r_U.clone(), l_U.clone(), comm_T);

      // Do calculation's outside circuit, to be passed in as advice to F'
      let r_squared = r * r;
      let E_new = r_U.comm_E + comm_T * r + l_U.comm_E * r_squared;
      let W_new = r_U.comm_W + l_U.comm_W * r;

      (
        NIFSVerifierCircuitInputs::<E>::new(
          Some(scalar_as_base::<E>(pp.digest_scan)),
          Some(data),
          Some(E_new),
          Some(W_new),
        ),
        r_U,
        r_W,
      )
    };

    let final_circuit_inputs =
      FinalCircuitInputs::new(Some(inputs_F), Some(inputs_ops), Some(inputs_scan));

    let final_circuit = FinalCircuit::<E>::new(
      &pp.pp.augmented_circuit_params,
      pp.pp.ro_consts_circuit_primary.clone(),
      Some(final_circuit_inputs),
    );

    self.rs.prove_step(&pp.pp, &final_circuit, self.IC_i)?;
    self.IC_i = self.rs.increment_commitment(&pp.pp, &final_circuit);

    // update the state
    self.r_U_F = r_U_F;
    self.r_W_F = r_W_F;
    self.r_U_ops = r_U_ops;
    self.r_W_ops = r_W_ops;
    self.r_U_scan = r_U_scan;
    self.r_W_scan = r_W_scan;
    self.i += 1;

    Ok(())
  }

  /// Verifies the [`L2`] instance
  pub fn verify(&self, pp: &L2AggregationPublicParams<E>) -> Result<(), NovaError> {
    self.rs.verify(
      &pp.pp,
      self.rs.num_steps(),
      &[E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO],
      self.IC_i,
    )?;

    Ok(())
  }
}
