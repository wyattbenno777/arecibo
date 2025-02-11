//! Module containing components to enable aggregation of IVC proofs.

use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS, NUM_FE_IN_EMULATED_POINT},
  errors::NovaError,
  gadgets::scalar_as_base,
  nebula::{
    augmented_circuit::AugmentedCircuitParams,
    layer_2::utils::{absorb_U, absorb_U_bn, random_fold_and_derandom},
    nifs::{CycleFoldRelaxedNIFS, PrimaryNIFS, PrimaryRelaxedNIFS},
    rs::{PublicParams, RecursiveSNARK},
  },
  r1cs::{CommitmentKeyHint, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    commitment::{CommitmentEngineTrait, Len},
    CurveCycleEquipped, Dual, Engine, ROTrait,
  },
  CommitmentKey, R1CSWithArity,
};
use ff::Field;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use verifier_circuit::VerifierCircuit;

use super::{nifs::NIFS, utils::Layer2FoldingData};
use crate::nebula::traits::{Layer1PPTrait, Layer1RSTrait, MemoryCommitmentsTraits};

pub mod compression;
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
  ck: Arc<CommitmentKey<E>>,
}

impl<E> AggregationPublicParams<E>
where
  E: CurveCycleEquipped,
{
  /// Produce the setup material for the Aggregation layer
  #[tracing::instrument(level = "info", name = "AggregationPublicParams::setup", skip_all)]
  pub fn setup(
    node_pp: impl Layer1PPTrait<E>,
    ck_hint_primary: &CommitmentKeyHint<E>,
    ck_hint_cyclefold: &CommitmentKeyHint<Dual<E>>,
  ) -> Self {
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

    // Get Public Params for Verifier Circuit
    let verifier_circuit: VerifierCircuit<E> =
      VerifierCircuit::new(aug_params, ro_consts, None, None, None, None);
    let pp: PublicParams<E> =
      PublicParams::setup(&verifier_circuit, ck_hint_primary, ck_hint_cyclefold);

    // choose ck with biggest size
    let ck = {
      let mut ck = ck_F;
      if ck_ops.length() > ck.length() {
        ck = ck_ops;
      }
      if ck_scan.length() > ck.length() {
        ck = ck_scan;
      }
      if pp.ck().length() > ck.length() {
        ck = pp.ck().clone();
      }
      ck
    };

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

  #[inline]
  fn r1cs_shape_cyclefold(&self) -> &R1CSShape<Dual<E>> {
    &self.pp.circuit_shape_cyclefold.r1cs_shape
  }

  fn ck_cyclefold(&self) -> &CommitmentKey<Dual<E>> {
    &self.pp.ck_cyclefold
  }

  fn ck(&self) -> &CommitmentKey<E> {
    &self.ck
  }

  fn augmented_circuit_params(&self) -> AugmentedCircuitParams {
    self.pp.augmented_circuit_params
  }

  fn default_cyclefold_instance(
    &self,
  ) -> (RelaxedR1CSInstance<Dual<E>>, RelaxedR1CSWitness<Dual<E>>) {
    (
      RelaxedR1CSInstance::default(self.ck_cyclefold(), self.r1cs_shape_cyclefold()),
      RelaxedR1CSWitness::default(self.r1cs_shape_cyclefold()),
    )
  }

  fn primary_r1cs_shapes(&self) -> Vec<&R1CSShape<E>> {
    vec![
      &self.circuit_shape_F.r1cs_shape,
      &self.circuit_shape_ops.r1cs_shape,
      &self.circuit_shape_scan.r1cs_shape,
      &self.pp.circuit_shape_primary.r1cs_shape,
    ]
  }
}

/// Layer 2 Recursive SNARK
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AggregationRecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  // F
  r_W_F: RelaxedR1CSWitness<E>,
  r_U_F: RelaxedR1CSInstance<E>,
  // ops
  r_W_ops: RelaxedR1CSWitness<E>,
  r_U_ops: RelaxedR1CSInstance<E>,
  // scan
  r_W_scan: RelaxedR1CSWitness<E>,
  r_U_scan: RelaxedR1CSInstance<E>,
  // secondary
  r_W_cyclefold: RelaxedR1CSWitness<Dual<E>>,
  r_U_cyclefold: RelaxedR1CSInstance<Dual<E>>,
  rs: RecursiveSNARK<E>,
  IC_i: E::Scalar,
  i: usize,
  z0: Vec<E::Scalar>,
}

impl<E> AggregationRecursiveSNARK<E>
where
  E: CurveCycleEquipped,
{
  #[tracing::instrument(skip_all, name = "AggregationRecursiveSNARK::new")]
  /// Constructs a new AggregationRecursiveSNARK instance
  pub fn new(
    pp: &AggregationPublicParams<E>,
    layer1_rs: &impl Layer1RSTrait<E>,
    _U: &impl MemoryCommitmentsTraits<E>,
  ) -> Result<Self, NovaError> {
    let (r_U_cyclefold, r_W_cyclefold) = pp.default_cyclefold_instance();

    /*
     * ********************************  F fold ********************************
     */
    let F_shape = &pp.circuit_shape_F.r1cs_shape;
    let r_U_F = RelaxedR1CSInstance::default(&*pp.ck, F_shape);
    let r_W_F = RelaxedR1CSWitness::default(F_shape);
    let (U2_F, W2_F, U2_secondary_F, W2_secondary_F) = layer1_rs.F().primary_secondary_U_W();
    let (nifs_F, (new_r_U_F, new_r_W_F), (r_U_cyclefold_temp1, r_W_cyclefold_temp1)) = NIFS::prove(
      (pp.ck(), pp.ck_cyclefold()),
      &pp.pp.ro_consts,
      &pp.digest_F,
      (
        &pp.circuit_shape_F.r1cs_shape,
        &pp.pp.circuit_shape_cyclefold.r1cs_shape,
      ),
      (&r_U_F, &r_W_F),
      (U2_F, W2_F),
      (&r_U_cyclefold, &r_W_cyclefold),
      (U2_secondary_F, W2_secondary_F),
    )?;
    let E_new_F = new_r_U_F.comm_E;
    let W_new_F = new_r_U_F.comm_W;
    let folding_data_F = Layer2FoldingData::new(
      Some(pp.digest_F),
      Some(nifs_F),
      Some(r_U_F.clone()),
      Some(U2_F.clone()),
      Some(E_new_F),
      Some(W_new_F),
      Some(U2_secondary_F.clone()),
    );

    /*
     * ********************************  ops fold ********************************
     */
    let shape_ops = &pp.circuit_shape_ops.r1cs_shape;
    let r_U_ops = RelaxedR1CSInstance::default(&*pp.ck, shape_ops);
    let r_W_ops = RelaxedR1CSWitness::default(shape_ops);
    let (U2_ops, W2_ops, U2_secondary_ops, W2_secondary_ops) =
      layer1_rs.ops().primary_secondary_U_W();
    let (nifs_ops, (new_r_U_ops, new_r_W_ops), (r_U_cyclefold_temp2, r_W_cyclefold_temp2)) =
      NIFS::prove(
        (pp.ck(), pp.ck_cyclefold()),
        &pp.pp.ro_consts,
        &pp.digest_ops,
        (
          &pp.circuit_shape_ops.r1cs_shape,
          &pp.pp.circuit_shape_cyclefold.r1cs_shape,
        ),
        (&r_U_ops, &r_W_ops),
        (U2_ops, W2_ops),
        (&r_U_cyclefold_temp1, &r_W_cyclefold_temp1),
        (U2_secondary_ops, W2_secondary_ops),
      )?;
    let E_new_ops = new_r_U_ops.comm_E;
    let W_new_ops = new_r_U_ops.comm_W;
    let folding_data_ops = Layer2FoldingData::new(
      Some(pp.digest_ops),
      Some(nifs_ops),
      Some(r_U_ops.clone()),
      Some(U2_ops.clone()),
      Some(E_new_ops),
      Some(W_new_ops),
      Some(U2_secondary_ops.clone()),
    );

    /*
     * ********************************  scan fold ********************************
     */
    let shape_scan = &pp.circuit_shape_scan.r1cs_shape;
    let r_U_scan = RelaxedR1CSInstance::default(&*pp.ck, shape_scan);
    let r_W_scan = RelaxedR1CSWitness::default(shape_scan);
    let (U2_scan, W2_scan, U2_secondary_scan, W2_secondary_scan) =
      layer1_rs.scan().primary_secondary_U_W();
    let (nifs_scan, (new_r_U_scan, new_r_W_scan), (new_r_U_cyclefold, new_r_W_cyclefold)) =
      NIFS::prove(
        (pp.ck(), pp.ck_cyclefold()),
        &pp.pp.ro_consts,
        &pp.digest_scan,
        (
          &pp.circuit_shape_scan.r1cs_shape,
          &pp.pp.circuit_shape_cyclefold.r1cs_shape,
        ),
        (&r_U_scan, &r_W_scan),
        (U2_scan, W2_scan),
        (&r_U_cyclefold_temp2, &r_W_cyclefold_temp2),
        (U2_secondary_scan, W2_secondary_scan),
      )?;
    let E_new_scan = new_r_U_scan.comm_E;
    let W_new_scan = new_r_U_scan.comm_W;
    let folding_data_scan = Layer2FoldingData::new(
      Some(pp.digest_scan),
      Some(nifs_scan),
      Some(r_U_scan.clone()),
      Some(U2_scan.clone()),
      Some(E_new_scan),
      Some(W_new_scan),
      Some(U2_secondary_scan.clone()),
    );

    /*
     * ********************************  Run verifier circuit ********************************
     */
    let verifier_circuit: VerifierCircuit<E> = VerifierCircuit::new(
      pp.augmented_circuit_params(),
      pp.pp.ro_consts_circuit.clone(),
      Some(folding_data_F),
      Some(folding_data_ops),
      Some(folding_data_scan),
      Some(r_U_cyclefold.clone()),
    );

    let z0 = {
      let mut ro = <Dual<E> as Engine>::RO::new(
        pp.pp.ro_consts.clone(),
        3 * (2 * NUM_FE_IN_EMULATED_POINT + 3) + (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS), // 3 * (U.W + U.comm_E + U.X + U.u) + U_cyclefold
      );
      absorb_U::<E>(&r_U_F, &mut ro);
      absorb_U::<E>(&r_U_ops, &mut ro);
      absorb_U::<E>(&r_U_scan, &mut ro);
      absorb_U_bn(&r_U_cyclefold, &mut ro);
      let hash_U = scalar_as_base::<Dual<E>>(ro.squeeze(NUM_CHALLENGE_BITS));
      vec![hash_U]
    };
    let mut IC_i = E::Scalar::ZERO;
    let mut rs = RecursiveSNARK::new(&pp.pp, &verifier_circuit, &z0)?;
    rs.prove_step(&pp.pp, &verifier_circuit, IC_i)?;
    IC_i = rs.increment_commitment(&pp.pp, &verifier_circuit);
    Ok(Self {
      r_W_F: new_r_W_F,
      r_U_F: new_r_U_F,
      r_W_ops: new_r_W_ops,
      r_U_ops: new_r_U_ops,
      r_W_scan: new_r_W_scan,
      r_U_scan: new_r_U_scan,
      r_W_cyclefold: new_r_W_cyclefold,
      r_U_cyclefold: new_r_U_cyclefold,
      rs,
      IC_i,
      i: 0,
      z0,
    })
  }

  #[tracing::instrument(skip_all, name = "AggregationRecursiveSNARK::prove_step")]
  /// Proves a step in the aggregation proof
  pub fn prove_step(
    &mut self,
    pp: &AggregationPublicParams<E>,
    layer1_rs: &impl Layer1RSTrait<E>,
    _U: &impl MemoryCommitmentsTraits<E>,
  ) -> Result<(), NovaError> {
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }
    /*
     * ********************************  F fold ********************************
     */
    let (U2_F, W2_F, U2_secondary_F, W2_secondary_F) = layer1_rs.F().primary_secondary_U_W();
    let (nifs_F, (new_r_U_F, new_r_W_F), (r_U_cyclefold_temp1, r_W_cyclefold_temp1)) = NIFS::prove(
      (pp.ck(), pp.ck_cyclefold()),
      &pp.pp.ro_consts,
      &pp.digest_F,
      (
        &pp.circuit_shape_F.r1cs_shape,
        &pp.pp.circuit_shape_cyclefold.r1cs_shape,
      ),
      (&self.r_U_F, &self.r_W_F),
      (U2_F, W2_F),
      (&self.r_U_cyclefold, &self.r_W_cyclefold),
      (U2_secondary_F, W2_secondary_F),
    )?;
    let E_new_F = new_r_U_F.comm_E;
    let W_new_F = new_r_U_F.comm_W;
    let folding_data_F = Layer2FoldingData::new(
      Some(pp.digest_F),
      Some(nifs_F),
      Some(self.r_U_F.clone()),
      Some(U2_F.clone()),
      Some(E_new_F),
      Some(W_new_F),
      Some(U2_secondary_F.clone()),
    );

    /*
     * ********************************  ops fold ********************************
     */
    let (U2_ops, W2_ops, U2_secondary_ops, W2_secondary_ops) =
      layer1_rs.ops().primary_secondary_U_W();
    let (nifs_ops, (new_r_U_ops, new_r_W_ops), (r_U_cyclefold_temp2, r_W_cyclefold_temp2)) =
      NIFS::prove(
        (pp.ck(), pp.ck_cyclefold()),
        &pp.pp.ro_consts,
        &pp.digest_ops,
        (
          &pp.circuit_shape_ops.r1cs_shape,
          &pp.pp.circuit_shape_cyclefold.r1cs_shape,
        ),
        (&self.r_U_ops, &self.r_W_ops),
        (U2_ops, W2_ops),
        (&r_U_cyclefold_temp1, &r_W_cyclefold_temp1),
        (U2_secondary_ops, W2_secondary_ops),
      )?;
    let E_new_ops = new_r_U_ops.comm_E;
    let W_new_ops = new_r_U_ops.comm_W;
    let folding_data_ops = Layer2FoldingData::new(
      Some(pp.digest_ops),
      Some(nifs_ops),
      Some(self.r_U_ops.clone()),
      Some(U2_ops.clone()),
      Some(E_new_ops),
      Some(W_new_ops),
      Some(U2_secondary_ops.clone()),
    );

    /*
     * ********************************  scan fold ********************************
     */
    let (U2_scan, W2_scan, U2_secondary_scan, W2_secondary_scan) =
      layer1_rs.scan().primary_secondary_U_W();
    let (nifs_scan, (new_r_U_scan, new_r_W_scan), (new_r_U_cyclefold, new_r_W_cyclefold)) =
      NIFS::prove(
        (pp.ck(), pp.ck_cyclefold()),
        &pp.pp.ro_consts,
        &pp.digest_scan,
        (
          &pp.circuit_shape_scan.r1cs_shape,
          &pp.pp.circuit_shape_cyclefold.r1cs_shape,
        ),
        (&self.r_U_scan, &self.r_W_scan),
        (U2_scan, W2_scan),
        (&r_U_cyclefold_temp2, &r_W_cyclefold_temp2),
        (U2_secondary_scan, W2_secondary_scan),
      )?;
    let E_new_scan = new_r_U_scan.comm_E;
    let W_new_scan = new_r_U_scan.comm_W;
    let folding_data_scan = Layer2FoldingData::new(
      Some(pp.digest_scan),
      Some(nifs_scan),
      Some(self.r_U_scan.clone()),
      Some(U2_scan.clone()),
      Some(E_new_scan),
      Some(W_new_scan),
      Some(U2_secondary_scan.clone()),
    );

    let verifier_circuit: VerifierCircuit<E> = VerifierCircuit::new(
      pp.augmented_circuit_params(),
      pp.pp.ro_consts_circuit.clone(),
      Some(folding_data_F),
      Some(folding_data_ops),
      Some(folding_data_scan),
      Some(self.r_U_cyclefold.clone()),
    );
    self.rs.prove_step(&pp.pp, &verifier_circuit, self.IC_i)?;
    self.IC_i = self.rs.increment_commitment(&pp.pp, &verifier_circuit);
    self.r_W_F = new_r_W_F;
    self.r_U_F = new_r_U_F;
    self.r_W_ops = new_r_W_ops;
    self.r_U_ops = new_r_U_ops;
    self.r_W_scan = new_r_W_scan;
    self.r_U_scan = new_r_U_scan;
    self.r_U_cyclefold = new_r_U_cyclefold;
    self.r_W_cyclefold = new_r_W_cyclefold;
    self.i += 1;
    Ok(())
  }

  #[tracing::instrument(skip_all, name = "AggregationRecursiveSNARK::verify")]
  /// Verifies the aggregation proof
  pub fn verify(&self, pp: &AggregationPublicParams<E>) -> Result<(), NovaError> {
    self
      .rs
      .verify(&pp.pp, self.rs.num_steps(), &self.z0, self.IC_i)?;
    let (res_r_F, (res_r_ops, (res_r_scan, res_r_cyclefold))) = rayon::join(
      || {
        pp.circuit_shape_F
          .r1cs_shape
          .is_sat_relaxed(&pp.ck, &self.r_U_F, &self.r_W_F)
      },
      || {
        rayon::join(
          || {
            pp.circuit_shape_ops
              .r1cs_shape
              .is_sat_relaxed(&pp.ck, &self.r_U_ops, &self.r_W_ops)
          },
          || {
            rayon::join(
              || {
                pp.circuit_shape_scan.r1cs_shape.is_sat_relaxed(
                  &pp.ck,
                  &self.r_U_scan,
                  &self.r_W_scan,
                )
              },
              || {
                pp.r1cs_shape_cyclefold().is_sat_relaxed(
                  pp.ck_cyclefold(),
                  &self.r_U_cyclefold,
                  &self.r_W_cyclefold,
                )
              },
            )
          },
        )
      },
    );
    res_r_F?;
    res_r_ops?;
    res_r_scan?;
    res_r_cyclefold?;
    Ok(())
  }

  fn fold_derandom(
    &self,
    pp: &AggregationPublicParams<E>,
  ) -> Result<
    (
      // rs
      RelaxedR1CSInstance<E>,
      RelaxedR1CSWitness<E>,
      PrimaryNIFS<E>,
      PrimaryRelaxedNIFS<E>,
      E::Scalar,
      E::Scalar,
      RelaxedR1CSInstance<E>,
      // F
      RelaxedR1CSInstance<E>,
      RelaxedR1CSWitness<E>,
      PrimaryRelaxedNIFS<E>,
      E::Scalar,
      E::Scalar,
      RelaxedR1CSInstance<E>,
      // ops
      RelaxedR1CSInstance<E>,
      RelaxedR1CSWitness<E>,
      PrimaryRelaxedNIFS<E>,
      E::Scalar,
      E::Scalar,
      RelaxedR1CSInstance<E>,
      // scan
      RelaxedR1CSInstance<E>,
      RelaxedR1CSWitness<E>,
      PrimaryRelaxedNIFS<E>,
      E::Scalar,
      E::Scalar,
      RelaxedR1CSInstance<E>,
    ),
    NovaError,
  > {
    // Primary RS fold
    let (
      U_verifier,
      W_verifier,
      nifs_verifier,
      nifs_r_verfier,
      wit_blind_verifer,
      err_blind_verifier,
      random_U_verifier,
    ) = self.rs.fold_ivc_compression_step(&pp.pp)?;

    // Randomize the rest of the running instances
    let (derandom_U_F, derandom_W_F, nifs_r_F, wit_blind_F, err_blind_F, random_U_F) =
      random_fold_and_derandom(
        &pp.circuit_shape_F.r1cs_shape,
        &pp.ck,
        &pp.pp.ro_consts,
        pp.digest_F,
        &self.r_U_F,
        &self.r_W_F,
      )?;
    let (derandom_U_ops, derandom_W_ops, nifs_r_ops, wit_blind_ops, err_blind_ops, random_U_ops) =
      random_fold_and_derandom(
        &pp.circuit_shape_ops.r1cs_shape,
        &pp.ck,
        &pp.pp.ro_consts,
        pp.digest_ops,
        &self.r_U_ops,
        &self.r_W_ops,
      )?;
    let (
      derandom_U_scan,
      derandom_W_scan,
      nifs_r_scan,
      wit_blind_scan,
      err_blind_scan,
      random_U_scan,
    ) = random_fold_and_derandom(
      &pp.circuit_shape_scan.r1cs_shape,
      &pp.ck,
      &pp.pp.ro_consts,
      pp.digest_scan,
      &self.r_U_scan,
      &self.r_W_scan,
    )?;

    Ok((
      // rs
      U_verifier,
      W_verifier,
      nifs_verifier,
      nifs_r_verfier,
      wit_blind_verifer,
      err_blind_verifier,
      random_U_verifier,
      // F
      derandom_U_F,
      derandom_W_F,
      nifs_r_F,
      wit_blind_F,
      err_blind_F,
      random_U_F,
      // ops
      derandom_U_ops,
      derandom_W_ops,
      nifs_r_ops,
      wit_blind_ops,
      err_blind_ops,
      random_U_ops,
      // scan
      derandom_U_scan,
      derandom_W_scan,
      nifs_r_scan,
      wit_blind_scan,
      err_blind_scan,
      random_U_scan,
    ))
  }

  fn fold_derandom_secondary(
    &self,
    pp: &AggregationPublicParams<E>,
  ) -> Result<
    (
      RelaxedR1CSInstance<Dual<E>>,
      RelaxedR1CSWitness<Dual<E>>,
      CycleFoldRelaxedNIFS<E>,
      CycleFoldRelaxedNIFS<E>,
      RelaxedR1CSInstance<Dual<E>>,
      <Dual<E> as Engine>::Scalar,
      <Dual<E> as Engine>::Scalar,
    ),
    NovaError,
  > {
    let ck = pp.ck_cyclefold();
    let ro_consts = &pp.pp.ro_consts;
    let S = pp.r1cs_shape_cyclefold();

    let (r_U_secondary_verifier, r_W_secondary_verifier) = self.rs.secondary_rs_part();
    let (nifs_1, (U_temp_1, W_temp_1), _) = CycleFoldRelaxedNIFS::<E>::prove(
      ck,
      ro_consts,
      S,
      r_U_secondary_verifier,
      r_W_secondary_verifier,
      &self.r_U_cyclefold,
      &self.r_W_cyclefold,
    )?;
    // Sample random U and W
    let (U_random, W_random) = S.sample_random_instance_witness(ck)?;

    // Random Fold
    let (nifs_final, (U, W), _) = CycleFoldRelaxedNIFS::<E>::prove(
      ck, ro_consts, S, &U_temp_1, &W_temp_1, &U_random, &W_random,
    )?;

    // Derandomize
    let (derandom_W, wit_blind, err_blind) = W.derandomize();
    let derandom_U = U.derandomize(
      &<Dual<E> as Engine>::CE::derand_key(ck),
      &wit_blind,
      &err_blind,
    );
    Ok((
      derandom_U, derandom_W, nifs_1, nifs_final, U_random, wit_blind, err_blind,
    ))
  }
}

#[cfg(test)]
mod test {
  use super::{
    compression::CompressedSNARK, AggregationPublicParams, AggregationRecursiveSNARK,
    Layer1PPTrait, Layer1RSTrait,
  };
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    nebula::{
      audit_rs::{AuditPublicParams, AuditRecursiveSNARK, AuditStepCircuit},
      rs::{PublicParams, RecursiveSNARK, StepCircuit},
      traits::MemoryCommitmentsTraits,
    },
    provider::{ipa_pc, Bn256EngineIPA},
    spartan,
    traits::{snark::default_ck_hint, CurveCycleEquipped, Dual, Engine},
  };
  use ff::{Field, PrimeField};
  use tracing_subscriber::{fmt, prelude::__tracing_subscriber_SubscriberExt, EnvFilter, Registry};
  use tracing_texray::TeXRayLayer;

  type E1 = Bn256EngineIPA;
  type F = <E1 as Engine>::Scalar;
  type EE1 = ipa_pc::EvaluationEngine<E1>;
  type EE2 = ipa_pc::EvaluationEngine<Dual<E1>>;
  type S1 = spartan::batched::BatchedRelaxedR1CSSNARK<E1, EE1>;
  type S2 = spartan::snark::RelaxedR1CSSNARK<Dual<E1>, EE2>;

  type TestMemoryComms<F> = (F, F);

  impl<E> MemoryCommitmentsTraits<E> for TestMemoryComms<E::Scalar>
  where
    E: CurveCycleEquipped,
  {
    fn C_FS(&self) -> <E>::Scalar {
      self.1
    }

    fn C_IS(&self) -> <E>::Scalar {
      self.0
    }
  }

  #[test]
  fn test_ivc_folding() {
    tracing_init();
    let (node_pp, nodes_rs) = node_nw(10);
    tracing_texray::examine(tracing::info_span!("aggregation"))
      .in_scope(|| aggregation_node(node_pp, &nodes_rs));
  }

  fn aggregation_node(node_pp: NodePP, nodes_rs: &[NodeRS]) {
    let aggregation_pp =
      AggregationPublicParams::<E1>::setup(node_pp, &*default_ck_hint(), &*default_ck_hint());
    let mut aggregation_engine =
      AggregationRecursiveSNARK::new(&aggregation_pp, &nodes_rs[0], &(F::ZERO, F::ZERO)).unwrap();

    for node_rs in nodes_rs.iter() {
      aggregation_engine
        .prove_step(&aggregation_pp, node_rs, &(F::ZERO, F::ZERO))
        .unwrap();
    }

    aggregation_engine.verify(&aggregation_pp).unwrap();

    let (pk, vk) = CompressedSNARK::<E1, S1, S2>::setup(&aggregation_pp).unwrap();
    let snark =
      CompressedSNARK::<E1, S1, S2>::prove(&aggregation_pp, &pk, &aggregation_engine).unwrap();
    snark.verify(&aggregation_pp, &vk).unwrap();
  }

  struct NodePP {
    pp1: PublicParams<E1>,
    pp2: PublicParams<E1>,
    pp3: AuditPublicParams<E1>,
  }

  impl Layer1PPTrait<E1> for NodePP {
    fn into_parts(self) -> (PublicParams<E1>, PublicParams<E1>, AuditPublicParams<E1>) {
      (self.pp1, self.pp2, self.pp3)
    }

    fn F(&self) -> &PublicParams<E1> {
      &self.pp1
    }

    fn ops(&self) -> &PublicParams<E1> {
      &self.pp2
    }

    fn scan(&self) -> &AuditPublicParams<E1> {
      &self.pp3
    }
  }

  struct NodeRS {
    rs1: RecursiveSNARK<E1>,
    rs2: RecursiveSNARK<E1>,
    rs3: AuditRecursiveSNARK<E1>,
  }

  impl Layer1RSTrait<E1> for NodeRS {
    fn F(&self) -> &RecursiveSNARK<E1> {
      &self.rs1
    }

    fn ops(&self) -> &RecursiveSNARK<E1> {
      &self.rs2
    }

    fn scan(&self) -> &AuditRecursiveSNARK<E1> {
      &self.rs3
    }
  }

  fn node_nw(num_proofs: usize) -> (NodePP, Vec<NodeRS>) {
    let pp1 = tracing::info_span!("node_pp").in_scope(|| {
      PublicParams::<E1>::setup(
        &LineCircuit::default(),
        &*default_ck_hint(),
        &*default_ck_hint(),
      )
    });

    let pp2 = tracing::info_span!("node_pp").in_scope(|| {
      PublicParams::<E1>::setup(
        &Add32Circuit::default(),
        &*default_ck_hint(),
        &*default_ck_hint(),
      )
    });

    let pp3 = tracing::info_span!("node_pp").in_scope(|| {
      AuditPublicParams::<E1>::setup(
        &Mul32Circuit::default(),
        &*default_ck_hint(),
        &*default_ck_hint(),
      )
    });

    let line_circuits = [
      LineCircuit { x: 4 },
      LineCircuit { x: 10 },
      LineCircuit { x: 100 },
      LineCircuit { x: 1231 },
    ];
    let add32_circuits = [
      Add32Circuit { a: 12, b: 10 },
      Add32Circuit { a: 8, b: 4 },
      Add32Circuit { a: 13, b: 1145 },
      Add32Circuit { a: 134, b: 562 },
    ];
    let mul32_circuits = [
      Mul32Circuit { a: 10, b: 4 },
      Mul32Circuit { a: 20, b: 7 },
      Mul32Circuit { a: 158, b: 143 },
      Mul32Circuit { a: 573, b: 520 },
    ];

    let proofs = (0..num_proofs)
      .map(|_| {
        let line_rs = RSNARK(&pp1, &line_circuits);
        let add32_rs = RSNARK(&pp2, &add32_circuits);
        let mul32_rs = AuditRSNARK(&pp3, &mul32_circuits);

        NodeRS {
          rs1: line_rs,
          rs2: add32_rs,
          rs3: mul32_rs,
        }
      })
      .collect::<Vec<_>>();

    (NodePP { pp1, pp2, pp3 }, proofs)
  }

  fn RSNARK(pp: &PublicParams<E1>, C: &[impl StepCircuit<F>]) -> RecursiveSNARK<E1>
  where
    F: PrimeField,
  {
    let z0 = vec![F::from(0u64)];

    let mut recursive_snark = RecursiveSNARK::new(pp, &C[0], &z0).unwrap();
    let mut IC_i = F::ZERO;

    for circuit in C {
      recursive_snark.prove_step(pp, circuit, IC_i).unwrap();

      IC_i = recursive_snark.increment_commitment(pp, circuit);
    }
    recursive_snark
      .verify(pp, recursive_snark.num_steps(), &z0, IC_i)
      .unwrap();

    recursive_snark
  }

  fn AuditRSNARK(
    pp: &AuditPublicParams<E1>,
    C: &[impl AuditStepCircuit<F>],
  ) -> AuditRecursiveSNARK<E1>
  where
    F: PrimeField,
  {
    let z0 = vec![F::from(0u64)];

    let mut recursive_snark = AuditRecursiveSNARK::new(pp, &C[0], &z0).unwrap();
    let mut IC_i = (F::ZERO, F::ZERO);

    for circuit in C {
      recursive_snark.prove_step(pp, circuit, IC_i).unwrap();

      IC_i = recursive_snark.increment_commitment(pp, circuit);
    }
    recursive_snark
      .verify(pp, recursive_snark.num_steps(), &z0, IC_i)
      .unwrap();

    recursive_snark
  }

  #[derive(Clone, Debug, Default)]
  struct LineCircuit {
    x: u32,
  }

  impl<F> StepCircuit<F> for LineCircuit
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn non_deterministic_advice(&self) -> Vec<F> {
      vec![]
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      // y = 2x + 1
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(F::from(self.x as u64)))?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        let two = F::from(2u64);
        let one = F::from(1u64);
        x.get_value()
          .map(|x| x * two + one)
          .ok_or(SynthesisError::AssignmentMissing)
      })?;

      // 2x = y - 1
      cs.enforce(
        || "2x = y - 1",
        |lc| lc + x.get_variable(),
        |lc| lc + (F::from(2u64), CS::one()),
        |lc| lc + y.get_variable() - CS::one(),
      );

      Ok(z.to_vec())
    }
  }

  #[derive(Clone, Debug, Default)]
  struct Add32Circuit {
    a: u32,
    b: u32,
  }

  impl<F> StepCircuit<F> for Add32Circuit
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn non_deterministic_advice(&self) -> Vec<F> {
      vec![]
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let zero = F::ZERO;
      let O = F::from(0x100000000u64);
      let ON: F = zero - O;

      let (c, of) = self.a.overflowing_add(self.b);
      let o = if of { ON } else { zero };

      // construct witness
      let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(F::from(self.a as u64)))?;
      let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(F::from(self.b as u64)))?;
      let c = AllocatedNum::alloc(cs.namespace(|| "c"), || Ok(F::from(c as u64)))?;

      // note, this is "advice"
      let o = AllocatedNum::alloc(cs.namespace(|| "o"), || Ok(o))?;
      let O = AllocatedNum::alloc(cs.namespace(|| "O"), || Ok(O))?;

      // check o * (o + O) == 0
      cs.enforce(
        || "check o * (o + O) == 0",
        |lc| lc + o.get_variable(),
        |lc| lc + o.get_variable() + O.get_variable(),
        |lc| lc,
      );

      // a + b + o = c
      cs.enforce(
        || "x + y + o = z",
        |lc| lc + a.get_variable() + b.get_variable() + o.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + c.get_variable(),
      );

      Ok(z.to_vec())
    }
  }

  #[derive(Clone, Debug, Default)]
  struct Mul32Circuit {
    a: u32,
    b: u32,
  }

  impl<F> AuditStepCircuit<F> for Mul32Circuit
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn FS_advice(&self) -> Vec<F> {
      vec![]
    }

    fn IS_advice(&self) -> Vec<F> {
      vec![]
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let zero = F::ZERO;
      let O = F::from(0x100000000u64);
      let ON: F = zero - O;

      let (c, of) = self.a.overflowing_mul(self.b);
      let o = if of { ON } else { zero };

      // construct witness
      let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(F::from(self.a as u64)))?;
      let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(F::from(self.b as u64)))?;
      let c = AllocatedNum::alloc(cs.namespace(|| "c"), || Ok(F::from(c as u64)))?;

      // note, this is "advice"
      let o = AllocatedNum::alloc(cs.namespace(|| "o"), || Ok(o))?;
      let O = AllocatedNum::alloc(cs.namespace(|| "O"), || Ok(O))?;

      // check o * (o + O) == 0
      cs.enforce(
        || "check o * (o + O) == 0",
        |lc| lc + o.get_variable(),
        |lc| lc + o.get_variable() + O.get_variable(),
        |lc| lc,
      );

      // a * b = c - o
      cs.enforce(
        || "a * b = c - o",
        |lc| lc + a.get_variable(),
        |lc| lc + b.get_variable(),
        |lc| lc + c.get_variable() - o.get_variable(),
      );

      Ok(z.to_vec())
    }
  }

  fn tracing_init() {
    // Create an EnvFilter that filters out spans below the 'info' level
    let filter = EnvFilter::new("arecibo=info");

    // Create a TeXRayLayer
    let texray_layer = TeXRayLayer::new(); // Optional: Only show spans longer than 100ms

    // Set up the global subscriber
    let subscriber = Registry::default()
      .with(filter)
      .with(fmt::layer())
      .with(texray_layer);
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
  }
}
