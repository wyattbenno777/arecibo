//! Defines behavior of Layer 1 Nebula proofs

use crate::{
  errors::NovaError,
  nebula::{
    audit_rs::{AuditPublicParams, AuditRecursiveSNARK},
    rs::{PublicParams, RecursiveSNARK},
  },
  r1cs::{R1CSInstance, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    commitment::{CommitmentEngineTrait, Len},
    CurveCycleEquipped, Dual, Engine, ROConstants,
  },
  CommitmentKey,
};
use std::sync::Arc;

/// Defines how a Layer1 RecursiveSNARK should be structured
pub trait Layer1RSTrait<E>
where
  E: CurveCycleEquipped,
{
  /// Returns the F instance
  fn F(&self) -> &RecursiveSNARK<E>;

  /// Returns the ops instance
  fn ops(&self) -> &RecursiveSNARK<E>;

  /// Returns the scan instance
  fn scan(&self) -> &AuditRecursiveSNARK<E>;

  /// Fold the CycleFold instances and derandomize
  fn fold_cyclefold_derandom(
    &self,
    pp: &impl Layer1PPTrait<E>,
  ) -> Result<
    (
      CycleFoldRelaxedNIFS<E>,
      CycleFoldRelaxedNIFS<E>,
      CycleFoldRelaxedNIFS<E>,
      RelaxedR1CSInstance<Dual<E>>,
      RelaxedR1CSWitness<Dual<E>>,
      RelaxedR1CSInstance<Dual<E>>,
      <Dual<E> as Engine>::Scalar,
      <Dual<E> as Engine>::Scalar,
    ),
    NovaError,
  > {
    let ck = pp.ck_secondary();
    let ro_consts = pp.ro_consts();
    let S = pp.cyclefold_r1cs_shape();
    let (r_U_cyclefold_F, r_W_cyclefold_F) = self.F().r_cyclefold();
    let (r_U_cyclefold_ops, r_W_cyclefold_ops) = self.ops().r_cyclefold();
    let (r_U_cyclefold_scan, r_W_cyclefold_scan) = self.scan().r_cyclefold();

    // First Fold
    let (nifs_1, (U_temp_1, W_temp_1), _) = CycleFoldRelaxedNIFS::<E>::prove(
      ck,
      ro_consts,
      S,
      r_U_cyclefold_F,
      r_W_cyclefold_F,
      r_U_cyclefold_ops,
      r_W_cyclefold_ops,
    )?;

    // Second Fold
    let (nifs_2, (U_temp_2, W_temp_2), _) = CycleFoldRelaxedNIFS::<E>::prove(
      ck,
      ro_consts,
      S,
      &U_temp_1,
      &W_temp_1,
      r_U_cyclefold_scan,
      r_W_cyclefold_scan,
    )?;

    // Sample random U and W
    let (U_random, W_random) = S.sample_random_instance_witness(ck)?;

    // Random Fold
    let (nifs_final, (U, W), _) = CycleFoldRelaxedNIFS::<E>::prove(
      ck, ro_consts, S, &U_temp_2, &W_temp_2, &U_random, &W_random,
    )?;

    // Derandomize
    let (derandom_W, wit_blind, err_blind) = W.derandomize();
    let derandom_U = U.derandomize(
      &<Dual<E> as Engine>::CE::derand_key(ck),
      &wit_blind,
      &err_blind,
    );

    Ok((
      nifs_1, nifs_2, nifs_final, derandom_U, derandom_W, U_random, wit_blind, err_blind,
    ))
  }

  /// Clone U secondary
  fn r_U_secondary_clone(&self) -> Vec<RelaxedR1CSInstance<Dual<E>>> {
    vec![
      self.F().r_U_cyclefold.clone(),
      self.ops().r_U_cyclefold.clone(),
      self.scan().r_U_cyclefold.clone(),
    ]
  }

  /// Clone primary running U
  fn r_U_clone(&self) -> Vec<RelaxedR1CSInstance<E>> {
    vec![
      self.F().r_U_primary.clone(),
      self.ops().r_U_primary.clone(),
      self.scan().r_U_primary.clone(),
    ]
  }

  /// Get fresh R1CSInstance
  fn l_u_clone(&self) -> Vec<R1CSInstance<E>> {
    vec![
      self.F().l_u_primary.clone(),
      self.ops().l_u_primary.clone(),
      self.scan().l_u_primary.clone(),
    ]
  }
}

/// Defines how a Layer 1 Nebula PublicParams should be structured
pub trait Layer1PPTrait<E: CurveCycleEquipped> {
  /// Splits the PublicParams into three parts (F, ops, scan)
  fn into_parts(self) -> (PublicParams<E>, PublicParams<E>, AuditPublicParams<E>);

  /// Returns the F public params
  fn F(&self) -> &PublicParams<E>;

  /// Returns the ops public params
  fn ops(&self) -> &PublicParams<E>;

  /// Returns the scan public params
  fn scan(&self) -> &AuditPublicParams<E>;

  /// Get the biggest commitmentkey
  fn biggest_ck<'a>(&'a self) -> &'a Arc<CommitmentKey<E>>
  where
    E: 'a,
  {
    let ck_F = self.F().ck();
    let ck_ops = self.ops().ck();
    let ck_scan = self.scan().ck();
    let mut ck = ck_F;
    if ck_ops.length() > ck.length() {
      ck = ck_ops;
    }
    if ck_scan.length() > ck.length() {
      ck = ck_scan;
    }
    ck
  }

  /// Get the primary R1CS shapes
  fn primary_r1cs_shapes(&self) -> Vec<&R1CSShape<E>> {
    vec![
      &self.F().circuit_shape_primary.r1cs_shape,
      &self.ops().circuit_shape_primary.r1cs_shape,
      &self.scan().circuit_shape_primary.r1cs_shape,
    ]
  }

  /// Get the secondary R1CS shapes
  fn secondary_r1cs_shapes<'a>(&'a self) -> Vec<&'a R1CSShape<Dual<E>>>
  where
    E: 'a,
  {
    vec![
      &self.F().circuit_shape_cyclefold.r1cs_shape,
      &self.ops().circuit_shape_cyclefold.r1cs_shape,
      &self.scan().circuit_shape_cyclefold.r1cs_shape,
    ]
  }

  /// Get the secondary R1CS shape
  fn cyclefold_r1cs_shape<'a>(&'a self) -> &'a R1CSShape<Dual<E>>
  where
    E: 'a,
  {
    &self.F().circuit_shape_cyclefold.r1cs_shape
  }

  /// Get the secondary ck
  fn ck_secondary<'a>(&'a self) -> &'a Arc<CommitmentKey<Dual<E>>>
  where
    E: 'a,
  {
    &self.F().ck_cyclefold
  }

  /// Get ro constants
  fn ro_consts<'a>(&'a self) -> &'a ROConstants<Dual<E>>
  where
    E: 'a,
  {
    &self.F().ro_consts
  }
}

/// Get the scan commitments from the statement the Layer 1 proof is proving
pub trait MemoryCommitmentsTraits<E>
where
  E: CurveCycleEquipped,
{
  /// Get commitment to C_is
  fn C_IS(&self) -> E::Scalar;
  /// commitment to C_fs
  fn C_FS(&self) -> E::Scalar;
}

/// Defines the fields that a RecursiveSNARK should have
pub trait RecursiveSNARKFieldsTrait<E>
where
  E: CurveCycleEquipped,
{
  /// Get r_U_cyclefold and r_W_cyclefold
  fn r_cyclefold(&self) -> (&RelaxedR1CSInstance<Dual<E>>, &RelaxedR1CSWitness<Dual<E>>);
  /// Get the random value
  fn r_i(&self) -> E::Scalar;
}

macro_rules! impl_rs_fields_trait {
  ($name:ident) => {
    impl<E> RecursiveSNARKFieldsTrait<E> for $name<E>
    where
      E: CurveCycleEquipped,
    {
      fn r_cyclefold(&self) -> (&RelaxedR1CSInstance<Dual<E>>, &RelaxedR1CSWitness<Dual<E>>) {
        (&self.r_U_cyclefold, &self.r_W_cyclefold)
      }

      fn r_i(&self) -> E::Scalar {
        self.r_i
      }
    }
  };
}

pub(crate) use impl_rs_fields_trait;

use super::nifs::CycleFoldRelaxedNIFS;
