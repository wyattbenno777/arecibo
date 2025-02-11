use super::emulated::AllocatedEmulPoint;
use crate::constants::NUM_MATRICES;
use crate::frontend::SynthesisError;
use crate::traits::commitment::CommitmentTrait;
use crate::Commitment;
use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem},
  r1cs::LR1CSInstance,
  traits::{CurveCycleEquipped, Dual, Engine, ROConstantsCircuit},
};
use ff::{Field, PrimeField};
use itertools::Itertools;
use std::marker::PhantomData;

pub struct NIFSGadget<E>
where
  E: CurveCycleEquipped,
{
  _engine: PhantomData<E>,
}

impl<E> NIFSGadget<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(mut cs: CS) -> Self
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    Self {
      _engine: PhantomData,
    }
  }

  pub fn verify<CS>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    ro_consts: &ROConstantsCircuit<Dual<E>>,
    U: &LR1CSInstanceGadget<E>,
    u: &R1CSInstanceGadget<E>,
    W_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  ) where
    CS: ConstraintSystem<E::Scalar>,
  {
    // Verify the NIFS
  }
}

pub struct LR1CSInstanceGadget<E>
where
  E: CurveCycleEquipped,
{
  pub comm_W: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  pub u: AllocatedNum<E::Scalar>,
  pub x0: AllocatedNum<E::Scalar>,
  pub x1: AllocatedNum<E::Scalar>,
  pub rx: Vec<AllocatedNum<E::Scalar>>,
  pub vs: Vec<AllocatedNum<E::Scalar>>,
}

impl<E> LR1CSInstanceGadget<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    inst: Option<&LR1CSInstance<E>>,
    limb_width: usize,
    n_limbs: usize,
    num_rounds: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let (comm_W, x0, x1) = alloc_instance_witness::<_, E>(
      cs.namespace(|| "allocate instance witness"),
      inst.map(|x| x.comm_W),
      inst.map(|x| &x.X),
      limb_width,
      n_limbs,
    )?;
    let u = alloc_scalar(cs.namespace(|| "allocate u"), inst.map(|x| x.u))?;
    let rx = alloc_sized_vec(cs.namespace(|| "rx"), inst.map(|x| &x.rx), num_rounds)?;
    let vs = alloc_sized_vec(cs.namespace(|| "vs"), inst.map(|x| &x.vs), NUM_MATRICES)?;
    Ok(Self {
      comm_W,
      u,
      x0,
      x1,
      rx,
      vs,
    })
  }
}

pub struct R1CSInstanceGadget<E>
where
  E: CurveCycleEquipped,
{
  pub comm_W: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  pub x0: AllocatedNum<E::Scalar>,
  pub x1: AllocatedNum<E::Scalar>,
}

impl<E> R1CSInstanceGadget<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    inst: Option<&LR1CSInstance<E>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let (comm_W, x0, x1) = alloc_instance_witness::<_, E>(
      cs.namespace(|| "allocate instance witness"),
      inst.map(|x| x.comm_W),
      inst.map(|x| &x.X),
      limb_width,
      n_limbs,
    )?;
    Ok(Self { comm_W, x0, x1 })
  }
}

fn alloc_sized_vec<CS, F>(
  mut cs: CS,
  v: Option<&Vec<F>>,
  size: usize,
) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
where
  CS: ConstraintSystem<F>,
  F: PrimeField,
{
  (0..size)
    .map(|i| alloc_scalar(cs.namespace(|| format!("v[{}]", i)), v.map(|v| v[i])))
    .try_collect()
}

fn alloc_instance_witness<CS, E>(
  mut cs: CS,
  comm_W: Option<Commitment<E>>,
  X: Option<&Vec<E::Scalar>>,
  limb_width: usize,
  n_limbs: usize,
) -> Result<
  (
    AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    AllocatedNum<E::Scalar>,
    AllocatedNum<E::Scalar>,
  ),
  SynthesisError,
>
where
  CS: ConstraintSystem<E::Scalar>,
  E: CurveCycleEquipped,
{
  let comm_W = AllocatedEmulPoint::alloc(
    cs.namespace(|| "allocate comm_W"),
    comm_W.map(|x| x.to_coordinates()),
    limb_width,
    n_limbs,
  )?;
  let x0 = alloc_scalar(cs.namespace(|| "allocate x0"), X.map(|X| X[0]))?;
  let x1 = alloc_scalar(cs.namespace(|| "allocate x1"), X.map(|X| X[1]))?;
  Ok((comm_W, x0, x1))
}

fn alloc_scalar<CS, F>(mut cs: CS, s: Option<F>) -> Result<AllocatedNum<F>, SynthesisError>
where
  CS: ConstraintSystem<F>,
  F: PrimeField,
{
  AllocatedNum::alloc(cs.namespace(|| "scalar"), || {
    s.map_or(Ok(F::ZERO), |s| Ok(s))
  })
}
