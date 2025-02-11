use super::emulated::AllocatedEmulPoint;
use crate::constants::NUM_CHALLENGE_BITS;
use crate::gadgets::le_bits_to_num;
use crate::traits::ROCircuitTrait;
use crate::{
  constants::{DEFAULT_ABSORBS, NUM_MATRICES, NUM_UNIVARIATE_COEFFS},
  frontend::{gadgets::Assignment, num::AllocatedNum, ConstraintSystem, SynthesisError},
  hypernova::{nifs::NIFS, ro_sumcheck::ROSumcheckProof},
  map_field,
  r1cs::{LR1CSInstance, R1CSInstance},
  spartan::polys::univariate::UniPoly,
  traits::{commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROConstantsCircuit},
  Commitment,
};
use ff::{Field, PrimeField};
use itertools::Itertools;
use std::marker::PhantomData;

pub struct AllocatedNIFS<E>
where
  E: CurveCycleEquipped,
{
  sc: AllocatedSumcheckProof<E>,
  sigmas: Vec<AllocatedNum<E::Scalar>>,
  thetas: Vec<AllocatedNum<E::Scalar>>,
}

impl<E> AllocatedNIFS<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    inst: Option<&NIFS<E>>,
    num_rounds: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let sc = AllocatedSumcheckProof::alloc(
      cs.namespace(|| "sumcheck proof"),
      map_field!(inst, sc),
      num_rounds,
    )?;
    let sigmas = alloc_sized_vec(
      cs.namespace(|| "sigmas"),
      map_field!(inst, sigmas),
      NUM_MATRICES,
    )?;
    let thetas = alloc_sized_vec(
      cs.namespace(|| "thetas"),
      map_field!(inst, thetas),
      NUM_MATRICES,
    )?;
    Ok(Self { sc, sigmas, thetas })
  }

  pub fn verify<CS>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    ro_consts: &ROConstantsCircuit<Dual<E>>,
    U: &AllocatedLR1CSInstance<E>,
    u: &AllocatedR1CSInstance<E>,
    W_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    num_rounds: usize,
  ) -> Result<AllocatedLR1CSInstance<E>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    // squeeze rho, gamma, beta
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp_digest);
    u.absorb_in_ro(cs.namespace(|| "absorb u"), &mut ro)?;
    let rho_bits = ro.squeeze(cs.namespace(|| "rho bits"), NUM_CHALLENGE_BITS)?;
    let rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(&rho);
    let gamma_bits = ro.squeeze(cs.namespace(|| "gamma bits"), NUM_CHALLENGE_BITS)?;
    let gamma = le_bits_to_num(cs.namespace(|| "gamma"), &gamma_bits)?;
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(&gamma);
    let beta: Vec<AllocatedNum<E::Scalar>> = ro
      .squeeze_vec(cs.namespace(|| "beta"), NUM_CHALLENGE_BITS, num_rounds)?
      .into_iter()
      .enumerate()
      .map(|(i, bits)| le_bits_to_num(cs.namespace(|| format!("beta[{}]", i)), &bits))
      .try_collect()?;
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ro_consts.clone(), DEFAULT_ABSORBS);
    for b in beta.iter() {
      ro.absorb(b);
    }

    let claim = self.compute_claim(
      cs.namespace(|| "claim"),
      &gamma,
      &U.vs[0],
      &U.vs[1],
      &U.vs[2],
    )?;
    todo!()
  }

  fn compute_claim<CS>(
    &self,
    mut cs: CS,
    gamma: &AllocatedNum<E::Scalar>,
    v0: &AllocatedNum<E::Scalar>,
    v1: &AllocatedNum<E::Scalar>,
    v2: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    // v[2] * gamma^2
    let gamma_squared = gamma.square(cs.namespace(|| "gamma * gamma"))?;
    let term_2 = v2.mul(cs.namespace(|| "v2 * gamma^2"), &gamma_squared)?;

    // v[1] * gamma
    let term_1 = v1.mul(cs.namespace(|| "v1 * gamma"), gamma)?;

    // claim = v[0] + v[1] * gamma + v[2] * gamma^2
    //       = term_0 + term_1 + term_2
    v0.add(cs.namespace(|| "claim"), &term_1)?
      .add(cs.namespace(|| "claim"), &term_2)
  }
}

pub struct AllocatedSumcheckProof<E>
where
  E: CurveCycleEquipped,
{
  polys: Vec<AllocatedUniPoly<E::Scalar>>,
}

impl<E> AllocatedSumcheckProof<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    inst: Option<&ROSumcheckProof<E>>,
    num_rounds: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    Ok(Self {
      polys: (0..num_rounds)
        .map(|i| {
          AllocatedUniPoly::alloc(
            cs.namespace(|| format!("poly_{i}")),
            inst.map(|inst| &inst.polys[i]),
          )
        })
        .try_collect()?,
    })
  }
}

pub struct AllocatedUniPoly<F>
where
  F: PrimeField,
{
  coeffs: Vec<AllocatedNum<F>>,
}

impl<F> AllocatedUniPoly<F>
where
  F: PrimeField,
{
  pub fn alloc<CS>(mut cs: CS, inst: Option<&UniPoly<F>>) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<F>,
  {
    Ok(Self {
      coeffs: alloc_sized_vec(
        cs.namespace(|| "coeffs"),
        inst.map(|poly| &poly.coeffs),
        NUM_UNIVARIATE_COEFFS,
      )?,
    })
  }
}

pub struct AllocatedLR1CSInstance<E>
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

impl<E> AllocatedLR1CSInstance<E>
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

pub struct AllocatedR1CSInstance<E>
where
  E: CurveCycleEquipped,
{
  pub comm_W: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  pub x0: AllocatedNum<E::Scalar>,
  pub x1: AllocatedNum<E::Scalar>,
}

impl<E> AllocatedR1CSInstance<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    inst: Option<&R1CSInstance<E>>,
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

  pub fn absorb_in_ro<CS>(
    &self,
    mut cs: CS,
    ro: &mut impl ROCircuitTrait<E::Scalar>,
  ) -> Result<(), SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    self
      .comm_W
      .absorb_in_ro(cs.namespace(|| "absorb u_W"), ro)?;
    ro.absorb(&self.x0);
    ro.absorb(&self.x1);
    Ok(())
  }
}

pub fn alloc_sized_vec<CS, F>(
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

pub fn increment<CS, F>(mut cs: CS, i: &AllocatedNum<F>) -> Result<AllocatedNum<F>, SynthesisError>
where
  CS: ConstraintSystem<F>,
  F: PrimeField,
{
  let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
    Ok(*i.get_value().get()? + F::ONE)
  })?;
  cs.enforce(
    || "check i + 1",
    |lc| lc,
    |lc| lc,
    |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
  );
  Ok(i_new)
}
