use super::{
  alloc_bignat_constant, alloc_tuple_comms, alloc_zero, conditionally_select,
  emulated::AllocatedEmulPoint, int::enforce_equal, utils::alloc_one, AllocatedPoint,
  AllocatedRelaxedR1CSInstance, BigNat, Num,
};
use crate::{
  constants::{
    DEFAULT_ABSORBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS, NUM_MATRICES, NUM_UNIVARIATE_COEFFS,
  },
  cyclefold::gadgets::AllocatedCycleFoldInstance,
  frontend::{gadgets::Assignment, num::AllocatedNum, Boolean, ConstraintSystem, SynthesisError},
  gadgets::le_bits_to_num,
  hypernova::{
    nifs::{CycleFoldNIFS, NIFS},
    ro_sumcheck::ROSumcheckProof,
  },
  map_field,
  r1cs::split::{LR1CSInstance, SplitR1CSInstance},
  spartan::polys::univariate::UniPoly,
  traits::{
    commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, Group, ROCircuitTrait,
    ROConstantsCircuit,
  },
  Commitment,
};
use ff::PrimeField;
use itertools::Itertools;

pub struct AllocatedNIFS<E>
where
  E: CurveCycleEquipped,
{
  sc: AllocatedSumcheckProof<E>,
  sigmas: Vec<AllocatedNum<E::Scalar>>,
  thetas: Vec<AllocatedNum<E::Scalar>>,
  cyclefold_nifs: AllocatedCycleFoldNIFS<E>,
  cyclefold_nifs_1: AllocatedCycleFoldNIFS<E>,
  cyclefold_nifs_2: AllocatedCycleFoldNIFS<E>,
}

impl<E> AllocatedNIFS<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    inst: Option<&NIFS<E>>,
    num_rounds: usize,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let sc = AllocatedSumcheckProof::alloc(
      cs.namespace(|| "sumcheck proof"),
      map_field!(inst, partial_nifs.sc),
      num_rounds,
    )?;
    let sigmas = alloc_sized_vec(
      cs.namespace(|| "sigmas"),
      map_field!(inst, partial_nifs.sigmas),
      NUM_MATRICES,
    )?;
    let thetas = alloc_sized_vec(
      cs.namespace(|| "thetas"),
      map_field!(inst, partial_nifs.thetas),
      NUM_MATRICES,
    )?;
    let cyclefold_nifs = AllocatedCycleFoldNIFS::alloc(
      cs.namespace(|| "cyclefold_nifs"),
      map_field!(inst, cyclefold_nifs),
      limb_width,
      n_limbs,
    )?;
    let cyclefold_nifs_1 = AllocatedCycleFoldNIFS::alloc(
      cs.namespace(|| "cyclefold_nifs_1"),
      map_field!(inst, cyclefold_nifs_1),
      limb_width,
      n_limbs,
    )?;
    let cyclefold_nifs_2 = AllocatedCycleFoldNIFS::alloc(
      cs.namespace(|| "cyclefold_nifs_2"),
      map_field!(inst, cyclefold_nifs_2),
      limb_width,
      n_limbs,
    )?;
    Ok(Self {
      sc,
      sigmas,
      thetas,
      cyclefold_nifs,
      cyclefold_nifs_1,
      cyclefold_nifs_2,
    })
  }

  pub fn verify<CS>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    ro_consts: &ROConstantsCircuit<Dual<E>>,
    U1: &AllocatedLR1CSInstance<E>,
    u1: &AllocatedSplitR1CSInstance<E>,
    W_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    pre_committed: (
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    ),
    U1_cyclefold: &AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
    num_rounds: usize,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<
    (
      AllocatedLR1CSInstance<E>,
      AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
    ),
    SynthesisError,
  >
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    // Squeeze rho, gamma, beta
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ro_consts.clone(), DEFAULT_ABSORBS);
    ro.absorb(pp_digest);
    u1.absorb_in_ro(cs.namespace(|| "absorb u"), &mut ro)?;
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

    // --- Verify sum-check proof ---
    //
    // claim <- U.vs[0] + gamma * U.vs[1] + gamma^2 * U.vs[2]
    let claim = Self::claim(
      cs.namespace(|| "claim"),
      &gamma,
      &U1.vs[0],
      &U1.vs[1],
      &U1.vs[2],
    )?;
    // verify sum-check
    let (sub_claim, rx_p) = self.sc.verify(
      cs.namespace(|| "verify sumcheck"),
      ro_consts,
      ro,
      &claim,
      num_rounds,
    )?;

    // Compute cl, cr from thetas and sigmas
    let cl = self.cl(cs.namespace(|| "cl"), &gamma, &U1.rx, &rx_p, num_rounds)?;
    let cr = self.cr(cs.namespace(|| "cr"), &gamma, &beta, &rx_p, num_rounds)?;
    let cl_add_cr = cl.add(cs.namespace(|| "cl + cr"), &cr)?;

    // cl + cr == sub_claim
    enforce_equal(&mut cs, || "cl + cr = sub_claim", &cl_add_cr, &sub_claim);

    // Output the folded instance
    let U = U1.fold(
      cs.namespace(|| "folded instance"),
      u1,
      &rho,
      rx_p,
      &self.sigmas,
      &self.thetas,
      W_new,
      pre_committed,
    )?;
    let U_cyclefold_temp = self.cyclefold_nifs.verify(
      cs.namespace(|| "cyclefold_nifs"),
      ro_consts.clone(),
      U1_cyclefold,
      limb_width,
      n_limbs,
    )?;
    let U_cyclefold_temp_1 = self.cyclefold_nifs_1.verify(
      cs.namespace(|| "cyclefold_nifs_1"),
      ro_consts.clone(),
      &U_cyclefold_temp,
      limb_width,
      n_limbs,
    )?;
    let U_cyclefold = self.cyclefold_nifs_2.verify(
      cs.namespace(|| "cyclefold_nifs_2"),
      ro_consts.clone(),
      &U_cyclefold_temp_1,
      limb_width,
      n_limbs,
    )?;
    Ok((U, U_cyclefold))
  }

  fn claim<CS>(
    mut cs: CS,
    gamma: &AllocatedNum<E::Scalar>,
    v0: &AllocatedNum<E::Scalar>,
    v1: &AllocatedNum<E::Scalar>,
    v2: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    Self::random_linear_combination(
      cs.namespace(|| "random linear combination"),
      gamma,
      v0,
      v1,
      v2,
    )
  }

  fn cl<CS>(
    &self,
    mut cs: CS,
    gamma: &AllocatedNum<E::Scalar>,
    rx: &[AllocatedNum<E::Scalar>],
    rx_p: &[AllocatedNum<E::Scalar>],
    num_rounds: usize,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let sigma_term = Self::random_linear_combination(
      cs.namespace(|| "sigma_term"),
      gamma,
      &self.sigmas[0],
      &self.sigmas[1],
      &self.sigmas[2],
    )?;
    let eq_poly = AllocatedEqPolynomial::new(rx.to_vec());
    let e = eq_poly.evaluate(cs.namespace(|| "eq_poly"), rx_p, num_rounds)?;
    let cl = sigma_term.mul(cs.namespace(|| "sigma_term * e"), &e)?;
    Ok(cl)
  }

  fn cr<CS>(
    &self,
    mut cs: CS,
    gamma: &AllocatedNum<E::Scalar>,
    beta: &[AllocatedNum<E::Scalar>],
    rx_p: &[AllocatedNum<E::Scalar>],
    num_rounds: usize,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let gamma_cubed = gamma
      .mul(cs.namespace(|| "gamma^2"), gamma)?
      .mul(cs.namespace(|| "gamma^3"), gamma)?;
    let ab = self.thetas[0].mul(cs.namespace(|| "a * b"), &self.thetas[1])?;
    let ab_sub_c = sub(cs.namespace(|| "a * b - c"), &ab, &self.thetas[2])?;
    let eq_poly = AllocatedEqPolynomial::new(beta.to_vec());
    let e = eq_poly.evaluate(cs.namespace(|| "eq_poly"), rx_p, num_rounds)?;
    let e_mul_gamma_cubed = e.mul(cs.namespace(|| "e * gamma^3"), &gamma_cubed)?;
    let cr = ab_sub_c.mul(
      cs.namespace(|| "(a * b - c) * e gamma^3"),
      &e_mul_gamma_cubed,
    )?;
    Ok(cr)
  }

  fn random_linear_combination<CS>(
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
    v0.add(cs.namespace(|| "term 1 + term 2"), &term_1)?
      .add(cs.namespace(|| "claim"), &term_2)
  }
}

pub struct AllocatedCycleFoldNIFS<E>
where
  E: CurveCycleEquipped,
{
  comm_T: AllocatedPoint<<Dual<E> as Engine>::GE>,
  l_u: AllocatedCycleFoldInstance<Dual<E>>,
}

impl<E> AllocatedCycleFoldNIFS<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    inst: Option<&CycleFoldNIFS<E>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let comm_T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      map_field!(inst, comm_T.to_coordinates()),
    )?;
    let l_u = AllocatedCycleFoldInstance::alloc(
      cs.namespace(|| "l_u"),
      map_field!(inst, l_u),
      limb_width,
      n_limbs,
    )?;
    Ok(Self { comm_T, l_u })
  }

  /// The NIFS verifier which folds the CycleFold instance into a running relaxed R1CS instance.
  pub fn verify<CS>(
    &self,
    mut cs: CS,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    U: &AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<AllocatedRelaxedR1CSInstance<Dual<E>, NIO_CYCLE_FOLD>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(ro_consts, DEFAULT_ABSORBS);
    U.absorb_in_ro(
      cs.namespace(|| "absorb cyclefold running instance"),
      &mut ro,
    )?;
    self
      .l_u
      .absorb_in_ro(cs.namespace(|| "absorb cyclefold instance"), &mut ro)?;
    ro.absorb(&self.comm_T.x);
    ro.absorb(&self.comm_T.y);
    ro.absorb(&self.comm_T.is_infinity);
    let r_bits = ro.squeeze(cs.namespace(|| "r bits"), NUM_CHALLENGE_BITS)?;
    let r = le_bits_to_num(cs.namespace(|| "r"), &r_bits)?;

    // W_fold = self.W + r * u.W
    let rW = self.l_u.W.scalar_mul(cs.namespace(|| "r * u.W"), &r_bits)?;
    let W_fold = U.W.add(cs.namespace(|| "self.W + r * u.W"), &rW)?;

    // E_fold = self.E + r * T
    let rT = self.comm_T.scalar_mul(cs.namespace(|| "r * T"), &r_bits)?;
    let E_fold = U.E.add(cs.namespace(|| "self.E + r * T"), &rT)?;

    // u_fold = u_r + r
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "u_fold"), || {
      Ok(*U.u.get_value().get()? + r.get_value().get()?)
    })?;
    cs.enforce(
      || "Check u_fold",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_fold.get_variable() - U.u.get_variable() - r.get_variable(),
    );

    // Fold the IO:
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

    // fold io
    let mut X_fold = vec![];
    for (idx, (X, x)) in U.X.iter().zip_eq(self.l_u.X.iter()).enumerate() {
      let (_, r) = x.mult_mod(cs.namespace(|| format!("r*u.X[{idx}]")), &r_bn, &m_bn)?;
      let r_new = X.add(&r)?;
      let X_i_fold = r_new.red_mod(cs.namespace(|| format!("reduce folded X[{idx}]")), &m_bn)?;
      X_fold.push(X_i_fold);
    }
    let X_fold = X_fold.try_into().map_err(|err: Vec<_>| {
      SynthesisError::IncompatibleLengthVector(format!("{} != {NIO_CYCLE_FOLD}", err.len()))
    })?;
    Ok(AllocatedRelaxedR1CSInstance {
      W: W_fold,
      E: E_fold,
      u: u_fold,
      X: X_fold,
    })
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
            map_field!(inst, polys[i]),
          )
        })
        .try_collect()?,
    })
  }

  fn verify<CS>(
    &self,
    mut cs: CS,
    ro_consts: &ROConstantsCircuit<Dual<E>>,
    ro: <Dual<E> as Engine>::ROCircuit,
    claim: &AllocatedNum<E::Scalar>,
    num_rounds: usize,
  ) -> Result<(AllocatedNum<E::Scalar>, Vec<AllocatedNum<E::Scalar>>), SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let mut e = claim.clone();
    let mut ro = ro;
    let mut rx = Vec::with_capacity(num_rounds);
    for i in 0..num_rounds {
      let poly = &self.polys[i];
      let s0 = poly.eval_at_zero();
      let s1 = poly.eval_at_one(cs.namespace(|| format!("eval at one {i}")))?;
      let s0_s1 = s0.add(cs.namespace(|| format!("s0 + s1 {i}")), &s1)?;
      enforce_equal(
        &mut cs,
        || format!("poly(0) + poly(1) == e {i}"),
        &s0_s1,
        &e,
      );
      poly.absorb_in_ro(&mut ro)?;
      let r_i_bits = ro.squeeze(cs.namespace(|| format!("r_bits_{i}")), NUM_CHALLENGE_BITS)?;
      let r_i = le_bits_to_num(cs.namespace(|| format!("r_{i}")), &r_i_bits)?;
      ro = <Dual<E> as Engine>::ROCircuit::new(ro_consts.clone(), DEFAULT_ABSORBS);
      ro.absorb(&r_i);
      e = poly.eval(cs.namespace(|| format!("eval_{i}")), &r_i)?;
      rx.push(r_i);
    }
    Ok((e, rx))
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
        map_field!(inst, coeffs),
        NUM_UNIVARIATE_COEFFS,
      )?,
    })
  }

  pub fn eval_at_zero(&self) -> AllocatedNum<F> {
    self.coeffs[0].clone()
  }

  pub fn eval_at_one<CS>(&self, mut cs: CS) -> Result<AllocatedNum<F>, SynthesisError>
  where
    CS: ConstraintSystem<F>,
  {
    let mut eval = alloc_zero(cs.namespace(|| "eval"));
    for i in 0..NUM_UNIVARIATE_COEFFS {
      eval = eval.add(cs.namespace(|| format!("add_{i}")), &self.coeffs[i])?
    }
    Ok(eval)
  }

  pub fn eval<CS>(&self, mut cs: CS, r: &AllocatedNum<F>) -> Result<AllocatedNum<F>, SynthesisError>
  where
    CS: ConstraintSystem<F>,
  {
    let mut eval = self.coeffs[0].clone();
    let mut power = r.clone();
    for i in 1..NUM_UNIVARIATE_COEFFS {
      let term = self.coeffs[i].mul(cs.namespace(|| format!("r_{i} * coeff_{i}")), &power)?;
      eval = eval.add(cs.namespace(|| format!("eval+= r_{i} * coeff")), &term)?;
      power = power.mul(cs.namespace(|| format!("r_^{i}")), r)?;
    }
    Ok(eval)
  }

  pub fn absorb_in_ro(&self, ro: &mut impl ROCircuitTrait<F>) -> Result<(), SynthesisError> {
    for i in 0..NUM_UNIVARIATE_COEFFS {
      ro.absorb(&self.coeffs[i]);
    }
    Ok(())
  }
}

pub struct AllocatedEqPolynomial<F>
where
  F: PrimeField,
{
  r: Vec<AllocatedNum<F>>,
}

impl<F> AllocatedEqPolynomial<F>
where
  F: PrimeField,
{
  pub fn new(r: Vec<AllocatedNum<F>>) -> Self {
    Self { r }
  }

  pub fn evaluate<CS>(
    &self,
    mut cs: CS,
    e: &[AllocatedNum<F>],
    num_rounds: usize,
  ) -> Result<AllocatedNum<F>, SynthesisError>
  where
    CS: ConstraintSystem<F>,
  {
    let one = alloc_one(cs.namespace(|| "one"));
    let mut eval = one.clone();
    #[allow(clippy::needless_range_loop)]
    for i in 0..num_rounds {
      let x_e = self.r[i].mul(cs.namespace(|| format!("x * e {i}")), &e[i])?;
      let sub_x = sub(cs.namespace(|| format!("1 - x {i}")), &one, &self.r[i])?;
      let sub_e = sub(cs.namespace(|| format!("1 - e {i}")), &one, &e[i])?;
      let sub_circ_e_x = sub_x.mul(cs.namespace(|| format!("(1 - x)(1 - e) {i}")), &sub_e)?;
      let term = x_e.add(
        cs.namespace(|| format!("x * e + (1 - x)(1 - e) {i}")),
        &sub_circ_e_x,
      )?;
      eval = eval.mul(cs.namespace(|| format!("eval * term {i}")), &term)?;
    }
    Ok(eval)
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
  pub pre_committed: (
    AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  ),
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
    let (comm_W, x0, x1, pre_committed) = alloc_instance_witness::<_, E>(
      cs.namespace(|| "allocate instance witness"),
      map_field!(inst, comm_W).copied(),
      map_field!(inst, X),
      map_field!(inst, pre_committed).copied(),
      limb_width,
      n_limbs,
    )?;
    let u = alloc_scalar(cs.namespace(|| "allocate u"), map_field!(inst, u).copied())?;
    let rx = alloc_sized_vec(cs.namespace(|| "rx"), map_field!(inst, rx), num_rounds)?;
    let vs = alloc_sized_vec(cs.namespace(|| "vs"), map_field!(inst, vs), NUM_MATRICES)?;
    Ok(Self {
      comm_W,
      u,
      x0,
      x1,
      rx,
      vs,
      pre_committed,
    })
  }
  pub fn fold<CS>(
    &self,
    mut cs: CS,
    u: &AllocatedSplitR1CSInstance<E>,
    rho: &AllocatedNum<E::Scalar>,
    rx_p: Vec<AllocatedNum<E::Scalar>>,
    sigmas: &[AllocatedNum<E::Scalar>],
    thetas: &[AllocatedNum<E::Scalar>],
    W_new: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    pre_committed: (
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    ),
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let u_fold = self.u.add(cs.namespace(|| "u_fold = u + rho"), rho)?;
    let x0_fold = fold_scalar(cs.namespace(|| "folded_x0"), &self.x0, &u.x0, rho)?;
    let x1_fold = fold_scalar(cs.namespace(|| "folded_x1"), &self.x1, &u.x1, rho)?;
    let mut vs = Vec::with_capacity(NUM_MATRICES);
    for i in 0..NUM_MATRICES {
      vs.push(fold_scalar(
        cs.namespace(|| format!("folded_v_{i}")),
        &sigmas[i],
        &thetas[i],
        rho,
      )?);
    }
    Ok(Self {
      comm_W: W_new,
      u: u_fold,
      x0: x0_fold,
      x1: x1_fold,
      rx: rx_p,
      vs,
      pre_committed,
    })
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
    ro.absorb(&self.u);
    ro.absorb(&self.x0);
    ro.absorb(&self.x1);
    for r in self.rx.iter() {
      ro.absorb(r);
    }
    for v in self.vs.iter() {
      ro.absorb(v);
    }
    self
      .pre_committed
      .0
      .absorb_in_ro(cs.namespace(|| "pre_committed0"), ro)?;
    self
      .pre_committed
      .1
      .absorb_in_ro(cs.namespace(|| "pre_committed1"), ro)?;
    Ok(())
  }

  pub fn conditionally_select<CS>(
    &self,
    mut cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let comm_W = self.comm_W.conditionally_select(
      cs.namespace(|| "comm_W = cond ? self.comm_W : other.comm_W"),
      &other.comm_W,
      condition,
    )?;
    let u = conditionally_select(
      cs.namespace(|| "u = cond ? self.u : other.u"),
      &self.u,
      &other.u,
      condition,
    )?;
    let x0 = conditionally_select(
      cs.namespace(|| "x0 = cond ? self.x0 : other.x0"),
      &self.x0,
      &other.x0,
      condition,
    )?;
    let x1 = conditionally_select(
      cs.namespace(|| "x1 = cond ? self.x1 : other.x1"),
      &self.x1,
      &other.x1,
      condition,
    )?;
    let rx = self
      .rx
      .iter()
      .zip_eq(other.rx.iter())
      .enumerate()
      .map(|(i, (a, b))| {
        conditionally_select(cs.namespace(|| format!("rx[{}]", i)), a, b, condition)
      })
      .try_collect()?;
    let vs = self
      .vs
      .iter()
      .zip_eq(other.vs.iter())
      .enumerate()
      .map(|(i, (a, b))| {
        conditionally_select(cs.namespace(|| format!("vs[{}]", i)), a, b, condition)
      })
      .try_collect()?;
    let pre_committed0 = self.pre_committed.0.conditionally_select(
      cs.namespace(|| "pre_committed0 = cond ? self.pre_committed0 : other.pre_committed0"),
      &other.pre_committed.0,
      condition,
    )?;
    let pre_committed1 = self.pre_committed.1.conditionally_select(
      cs.namespace(|| "pre_committed1 = cond ? self.pre_committed1 : other.pre_committed1"),
      &other.pre_committed.1,
      condition,
    )?;
    Ok(Self {
      comm_W,
      u,
      x0,
      x1,
      rx,
      vs,
      pre_committed: (pre_committed0, pre_committed1),
    })
  }

  pub fn default<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    limb_width: usize,
    n_limbs: usize,
    num_rounds: usize,
  ) -> Result<Self, SynthesisError> {
    let default_point =
      AllocatedEmulPoint::default(cs.namespace(|| "default comm_W"), limb_width, n_limbs)?;
    let u = alloc_zero(cs.namespace(|| "u = 0"));
    let x0 = u.clone();
    let x1 = u.clone();
    let rx = (0..num_rounds).map(|_| u.clone()).collect_vec();
    let vs = (0..NUM_MATRICES).map(|_| u.clone()).collect_vec();
    Ok(Self {
      comm_W: default_point.clone(),
      u,
      x0,
      x1,
      rx,
      vs,
      pre_committed: (default_point.clone(), default_point),
    })
  }
}

pub struct AllocatedSplitR1CSInstance<E>
where
  E: CurveCycleEquipped,
{
  pub comm_W: AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  pub x0: AllocatedNum<E::Scalar>,
  pub x1: AllocatedNum<E::Scalar>,
  pub pre_committed: (
    AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
  ),
}

impl<E> AllocatedSplitR1CSInstance<E>
where
  E: CurveCycleEquipped,
{
  pub fn alloc<CS>(
    mut cs: CS,
    inst: Option<&SplitR1CSInstance<E>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let (comm_W, x0, x1, pre_committed) = alloc_instance_witness::<_, E>(
      cs.namespace(|| "allocate instance witness"),
      map_field!(inst, aux.comm_W).copied(),
      map_field!(inst, aux.X),
      map_field!(inst, pre_committed).copied(),
      limb_width,
      n_limbs,
    )?;
    Ok(Self {
      comm_W,
      x0,
      x1,
      pre_committed,
    })
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
    self
      .pre_committed
      .0
      .absorb_in_ro(cs.namespace(|| "pre_committed0"), ro)?;
    self
      .pre_committed
      .1
      .absorb_in_ro(cs.namespace(|| "pre_committed1"), ro)?;
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
  pre_committed: Option<(Commitment<E>, Commitment<E>)>,
  limb_width: usize,
  n_limbs: usize,
) -> Result<
  (
    AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    AllocatedNum<E::Scalar>,
    AllocatedNum<E::Scalar>,
    (
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,
    ),
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
  let pre_committed = alloc_tuple_comms::<_, E>(
    cs.namespace(|| "pre_committed"),
    pre_committed,
    limb_width,
    n_limbs,
  )?;
  Ok((comm_W, x0, x1, pre_committed))
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

/// Adds a constraint to CS, enforcing a difference relationship between the allocated numbers a, b, and difference.
///
/// a - b = difference
pub(crate) fn enforce_difference<F: PrimeField, A, AR, CS: ConstraintSystem<F>>(
  cs: &mut CS,
  annotation: A,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  difference: &AllocatedNum<F>,
) where
  A: FnOnce() -> AR,
  AR: Into<String>,
{
  //    difference = a-b
  // => difference + b = a
  // => (difference + b) * 1 = a
  cs.enforce(
    annotation,
    |lc| lc + difference.get_variable() + b.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + a.get_variable(),
  );
}

/// Compute difference and enforce it.
pub(crate) fn sub<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let res = AllocatedNum::alloc(cs.namespace(|| "sub_num"), || {
    let mut tmp = a.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    tmp.sub_assign(&b.get_value().ok_or(SynthesisError::AssignmentMissing)?);

    Ok(tmp)
  })?;

  // a - b = res
  enforce_difference(&mut cs, || "subtraction constraint", a, b, &res);
  Ok(res)
}

/// a + rho * b
pub(crate) fn fold_scalar<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  rho: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let folded_scalar = AllocatedNum::alloc(cs.namespace(|| "folded_scalar"), || {
    Ok(*a.get_value().get()? + *rho.get_value().get()? * *b.get_value().get()?)
  })?;
  cs.enforce(
    || "folded_scalar = a + rho * b",
    |lc| lc + rho.get_variable(),
    |lc| lc + b.get_variable(),
    |lc| lc + folded_scalar.get_variable() - a.get_variable(),
  );
  Ok(folded_scalar)
}
