use super::nifs::CycleFoldNIFS;
use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS},
  cyclefold::gadgets::{
    emulated::{self},
    AllocatedCycleFoldInstance,
  },
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{
    alloc_bignat_constant, emulated::AllocatedEmulPoint, le_bits_to_num, AllocatedPoint, BigNat,
    Num,
  },
  nebula::nifs::{CycleFoldRelaxedNIFS, PrimaryRelaxedNIFS},
  traits::{
    commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, Group, ROCircuitTrait,
    ROConstantsCircuit,
  },
};
use itertools::Itertools;
use r1cs::AllocatedRelaxedR1CSInstanceBn;

pub mod r1cs;
#[cfg(test)]
mod test;

/// Verifer gadget used to fold IVC proofs.
pub struct NIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  // proof from primary fold
  pub(super) nifs_primary: PrimaryNIFSVerifierGadget<E>,

  // proof from first cyclefold fold
  pub(super) nifs_E1: CycleFoldNIFSVerifierGadget<E>,
  pub(super) l_u_cyclefold_E1: AllocatedCycleFoldInstance<Dual<E>>,

  // proof from second cyclefold fold
  pub(super) nifs_E2: CycleFoldNIFSVerifierGadget<E>,
  pub(super) l_u_cyclefold_E2: AllocatedCycleFoldInstance<Dual<E>>,

  // proof from third cyclefold fold
  pub(super) nifs_W: CycleFoldNIFSVerifierGadget<E>,
  pub(super) l_u_cyclefold_W: AllocatedCycleFoldInstance<Dual<E>>,

  // proof from fourth cyclefold fold
  pub(super) nifs_final_cyclefold: CycleFoldRelaxedNIFSVerifierGadget<E>,
}

impl<E> NIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  pub fn verify<CS>(
    &self,
    mut cs: CS,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    limb_width: usize,
    n_limbs: usize,
    U1: &emulated::AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
    U2: &emulated::AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
    U1_secondary: &AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>,
    U2_secondary: &AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>,
    pp_digest: &AllocatedNum<E::Scalar>,
    E_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    W_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  ) -> Result<
    (
      emulated::AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
      AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>,
    ),
    SynthesisError,
  >
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let U = self.nifs_primary.verify(
      cs.namespace(|| "primary fold"),
      ro_consts.clone(),
      U1,
      U2,
      pp_digest,
      E_new,
      W_new,
    )?;

    let U_secondary_temp = self.nifs_E1.verify(
      cs.namespace(|| "verify first cyclefold"),
      ro_consts.clone(),
      limb_width,
      n_limbs,
      U1_secondary,
      &self.l_u_cyclefold_E1,
    )?;

    let U_secondary_temp_1 = self.nifs_E2.verify(
      cs.namespace(|| "verify second cyclefold"),
      ro_consts.clone(),
      limb_width,
      n_limbs,
      &U_secondary_temp,
      &self.l_u_cyclefold_E2,
    )?;

    let U_secondary_temp_2 = self.nifs_W.verify(
      cs.namespace(|| "verify third cyclefold"),
      ro_consts.clone(),
      limb_width,
      n_limbs,
      &U_secondary_temp_1,
      &self.l_u_cyclefold_W,
    )?;

    let U_secondary = self.nifs_final_cyclefold.verify(
      cs.namespace(|| "verify fourth cyclefold"),
      ro_consts.clone(),
      limb_width,
      n_limbs,
      &U_secondary_temp_2,
      U2_secondary,
    )?;

    Ok((U, U_secondary))
  }
}

/// Verifier gadget for primary fold
pub struct PrimaryNIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  comm_T: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
}

impl<E> PrimaryNIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    nifs: Option<&PrimaryRelaxedNIFS<E>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let comm_T = AllocatedEmulPoint::alloc(
      cs.namespace(|| "allocate T"),
      nifs.map(|nifs| nifs.comm_T.to_coordinates()),
      limb_width,
      n_limbs,
    )?;
    Ok(Self { comm_T })
  }

  pub fn verify<CS>(
    &self,
    mut cs: CS,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    U1: &emulated::AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
    U2: &emulated::AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
    pp_digest: &AllocatedNum<E::Scalar>,
    E_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    W_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  ) -> Result<emulated::AllocatedEmulRelaxedR1CSInstance<Dual<E>>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    U1.fold_with_relaxed_r1cs(
      cs.namespace(|| "fold with relaxed r1cs"),
      pp_digest,
      U2,
      W_new,
      E_new,
      &self.comm_T,
      ro_consts,
    )
  }
}

/// Verifier gadget to fold CycleFold instance pairs
pub struct CycleFoldNIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  comm_T: AllocatedPoint<<Dual<E> as Engine>::GE>,
}

impl<E> CycleFoldNIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(mut cs: CS, nifs: Option<&CycleFoldNIFS<E>>) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let comm_T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      nifs.map(|nifs| nifs.comm_T.to_coordinates()),
    )?;
    Ok(Self { comm_T })
  }
  pub fn verify<CS>(
    &self,
    mut cs: CS,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    limb_width: usize,
    n_limbs: usize,
    U: &AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>,
    u: &AllocatedCycleFoldInstance<Dual<E>>,
  ) -> Result<AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      ro_consts,
      (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS) + (3 + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // (U) + (u) + T
    );
    U.absorb_in_ro(
      cs.namespace(|| "absorb cyclefold running instance"),
      &mut ro,
    )?;
    u.absorb_in_ro(cs.namespace(|| "absorb cyclefold instance"), &mut ro)?;
    ro.absorb(&self.comm_T.x);
    ro.absorb(&self.comm_T.y);
    ro.absorb(&self.comm_T.is_infinity);
    let r_bits = ro.squeeze(cs.namespace(|| "r bits"), NUM_CHALLENGE_BITS)?;
    let r = le_bits_to_num(cs.namespace(|| "r"), &r_bits)?;

    // W_fold = self.W + r * u.W
    let rW = u.W.scalar_mul(cs.namespace(|| "r * u.W"), &r_bits)?;
    let W_fold = U.W.add(cs.namespace(|| "self.W + r * u.W"), &rW)?;

    // E_fold = self.E + r * T
    let rT = self.comm_T.scalar_mul(cs.namespace(|| "r * T"), &r_bits)?;
    let E_fold = U.E.add(cs.namespace(|| "self.E + r * T"), &rT)?;

    // Analyze r into limbs
    let r_bn = BigNat::from_num(
      cs.namespace(|| "allocate r_bn"),
      &Num::from(r),
      limb_width,
      n_limbs,
    )?;

    // Allocate the order of the non-native field as a constant
    let m_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc m"),
      &<Dual<E> as Engine>::GE::group_params().2,
      limb_width,
      n_limbs,
    )?;

    // u_fold = u_r + r
    let u_fold = U
      .u
      .add(&r_bn)?
      .red_mod(cs.namespace(|| "reduce u_fold"), &m_bn)?;

    // Fold the IO:
    let mut X_fold = vec![];

    // Calculate the folded io variables
    for (idx, (X, x)) in U.X.iter().zip_eq(u.X.iter()).enumerate() {
      let (_, r) = x.mult_mod(cs.namespace(|| format!("r*u.X[{idx}]")), &r_bn, &m_bn)?;
      let r_new = X.add(&r)?;
      let X_i_fold = r_new.red_mod(cs.namespace(|| format!("reduce folded X[{idx}]")), &m_bn)?;
      X_fold.push(X_i_fold);
    }
    let X_fold = X_fold.try_into().map_err(|err: Vec<_>| {
      SynthesisError::IncompatibleLengthVector(format!("{} != {NIO_CYCLE_FOLD}", err.len()))
    })?;

    Ok(AllocatedRelaxedR1CSInstanceBn {
      W: W_fold,
      E: E_fold,
      u: u_fold,
      X: X_fold,
    })
  }
}

/// Verifier gadget to fold CycleFold relaxed instance pairs
pub struct CycleFoldRelaxedNIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  comm_T: AllocatedPoint<<Dual<E> as Engine>::GE>,
}

impl<E> CycleFoldRelaxedNIFSVerifierGadget<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    nifs: Option<&CycleFoldRelaxedNIFS<E>>,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let comm_T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      nifs.map(|nifs| nifs.comm_T.to_coordinates()),
    )?;
    Ok(Self { comm_T })
  }

  pub fn verify<CS>(
    &self,
    mut cs: CS,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    limb_width: usize,
    n_limbs: usize,
    U1: &AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>,
    U2: &AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>,
  ) -> Result<AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      ro_consts,
      2 * (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS) + 3, // 2* (U) + T
    );
    U1.absorb_in_ro(
      cs.namespace(|| "absorb cyclefold running instance"),
      &mut ro,
    )?;
    U2.absorb_in_ro(cs.namespace(|| "absorb U2"), &mut ro)?;
    ro.absorb(&self.comm_T.x);
    ro.absorb(&self.comm_T.y);
    ro.absorb(&self.comm_T.is_infinity);
    let r_bits = ro.squeeze(cs.namespace(|| "r bits"), NUM_CHALLENGE_BITS)?;
    let r = le_bits_to_num(cs.namespace(|| "r"), &r_bits)?;

    // Analyze r into limbs
    let r_bn = BigNat::from_num(
      cs.namespace(|| "allocate r_bn"),
      &Num::from(r.clone()),
      limb_width,
      n_limbs,
    )?;

    // Allocate the order of the non-native field as a constant
    let m_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc m"),
      &<Dual<E> as Engine>::GE::group_params().2,
      limb_width,
      n_limbs,
    )?;

    let (_, r_squared_bn) = r_bn.mult_mod(cs.namespace(|| "r*r"), &r_bn, &m_bn)?;
    let r_squared_bits = r_squared_bn.to_bits_le(cs.namespace(|| "r_squared_bits"))?;

    // W_fold = self.W + r * u.W
    let rW = U2.W.scalar_mul(cs.namespace(|| "r * u.W"), &r_bits)?;
    let W_fold = U1.W.add(cs.namespace(|| "self.W + r * u.W"), &rW)?;

    // E_fold = U1.E + r * T + r^2 * U2.E
    let rT = self.comm_T.scalar_mul(cs.namespace(|| "r * T"), &r_bits)?;
    let E_fold_term_1 = U1.E.add(cs.namespace(|| "self.E + r * T"), &rT)?;

    let r2E = U2
      .E
      .scalar_mul(cs.namespace(|| "r^2 * U2.E"), &r_squared_bits)?;
    let E_fold = E_fold_term_1.add(cs.namespace(|| "E_fold_term_1 + r^2 * U2.E"), &r2E)?;

    // u_fold = U.u + r * u.u
    let u_fold = {
      let (_, r_u2) = U2.u.mult_mod(cs.namespace(|| "r * u2"), &r_bn, &m_bn)?;
      U1.u
        .add(&r_u2)?
        .red_mod(cs.namespace(|| "reduce u_fold"), &m_bn)?
    };

    // Fold the IO:
    let mut X_fold = vec![];

    // Calculate the folded io variables
    for (idx, (X, x)) in U1.X.iter().zip_eq(U2.X.iter()).enumerate() {
      let (_, r) = x.mult_mod(cs.namespace(|| format!("r*u.X[{idx}]")), &r_bn, &m_bn)?;
      let r_new = X.add(&r)?;
      let X_i_fold = r_new.red_mod(cs.namespace(|| format!("reduce folded X[{idx}]")), &m_bn)?;
      X_fold.push(X_i_fold);
    }
    let X_fold = X_fold.try_into().map_err(|err: Vec<_>| {
      SynthesisError::IncompatibleLengthVector(format!("{} != {NIO_CYCLE_FOLD}", err.len()))
    })?;

    Ok(AllocatedRelaxedR1CSInstanceBn {
      W: W_fold,
      E: E_fold,
      u: u_fold,
      X: X_fold,
    })
  }
}
