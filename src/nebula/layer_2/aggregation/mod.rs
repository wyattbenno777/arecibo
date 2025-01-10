//! Module containing components to enable aggregation of IVC proofs.

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

use serde::{Deserialize, Serialize};
use verifier_circuit::VerifierCircuit;

use super::traits::Layer1PP;

mod verifier_circuit;

/// Defines the public parameters for the Aggregation layer
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AggregationPublicParams<E>
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

impl<E> AggregationPublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// Produce the setup material for the Aggregation layer
  pub fn setup<PP1>(node_pp: impl Layer1PP<E>) -> Self {
    // Get already setup public params from layer 1
    let (pp_F, pp_ops, pp_scan) = node_pp.into_parts();

    // Public Params for Verifier Circuit
    let aug_params = pp_F.augmented_circuit_params;
    let ro_consts = pp_F.ro_consts_circuit.clone();

    // Get Layer 1 circuit shapes, commitment keys and pp.digests.
    // We need circuit shapes to and commitment keys to construct the default R1CS instance's and witness's.
    // And we use the digests in out NIFS.
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

    // Get Public Params for Verifier Circuit
    let verifier_circuit: VerifierCircuit<E> =
      VerifierCircuit::new(aug_params, ro_consts, None, None, None, None);
    let pp: PublicParams<E> =
      PublicParams::setup(&verifier_circuit, &*default_ck_hint(), &*default_ck_hint());

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
