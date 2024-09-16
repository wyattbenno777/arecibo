#![allow(dead_code)] // methods will be used later for batched_ppsnark aggregating
use std::marker::PhantomData;

use crate::gadgets::alloc_one;
use crate::gadgets::{alloc_negate, alloc_zero};
use crate::spartan::math::Math;

use crate::spartan::polys::univariate::UniPoly;

use bellpepper::gadgets::Assignment;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use ff::PrimeField;

use itertools::Itertools as _;

/// A multilinear extension of a polynomial $Z(\cdot)$, denote it as $\tilde{Z}(x_1, ..., x_m)$
/// where the degree of each variable is at most one.
///
/// This is the dense representation of a multilinear poynomial.
/// Let it be $\mathbb{G}(\cdot): \mathbb{F}^m \rightarrow \mathbb{F}$, it can be represented uniquely by the list of
/// evaluations of $\mathbb{G}(\cdot)$ over the Boolean hypercube $\{0, 1\}^m$.
///
/// For example, a 3 variables multilinear polynomial can be represented by evaluation
/// at points $[0, 2^3-1]$.
///
/// The implementation follows
/// $$
/// \tilde{Z}(x_1, ..., x_m) = \sum_{e\in {0,1}^m}Z(e) \cdot \prod_{i=1}^m(x_i \cdot e_i + (1-x_i) \cdot (1-e_i))
/// $$
///
/// Vector $Z$ indicates $Z(e)$ where $e$ ranges from $0$ to $2^m-1$.
#[derive(Debug, Clone)]
pub struct AllocMultilinearPolynomial<Scalar: PrimeField> {
  pub(crate) Z: Vec<AllocatedNum<Scalar>>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<Scalar: PrimeField> AllocMultilinearPolynomial<Scalar> {
  /// Creates a new `MultilinearPolynomial` from the given evaluations.
  ///
  /// # Panics
  /// The number of evaluations must be a power of two.
  pub fn new(Z: Vec<AllocatedNum<Scalar>>) -> Self {
    let num_vars = Z.len().log_2();
    assert_eq!(Z.len(), 1 << num_vars);
    Self { Z }
  }

  /// Evaluates the polynomial with the given evaluations and chi coefficients
  pub fn evaluate_with_chis<CS: ConstraintSystem<Scalar>>(
    mut cs: CS,
    Z: &[AllocatedNum<Scalar>],
    chis: &[AllocatedNum<Scalar>],
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let mut res = alloc_zero(cs.namespace(|| "res"));
    // zip_with!(par_iter, (chis, Z), |a, b| *a * b).sum()
    for (i, (chi, z)) in chis.iter().zip_eq(Z).enumerate() {
      let chi_times_z = chi.mul(cs.namespace(|| format!("chi * z_{i}")), z).unwrap();
      res = res.add(
        cs.namespace(|| format!("res + chi_times_z_{i}")),
        &chi_times_z,
      )?;
    }

    Ok(res)
  }

  /// evaluations of the polynomial in all the 2^num_vars Boolean inputs
  pub fn evaluations(&self) -> &[AllocatedNum<Scalar>] {
    &self.Z[..]
  }
}

/*
 * ******************
 * Univariate Polynomial
 * ******************
 */

/// A univariate polynomial allocated in a constraint system.
// ax^2 + bx + c stored as vec![c, b, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, b, a]
pub struct AllocatedUniPoly<F: PrimeField> {
  pub coeffs: Vec<AllocatedNum<F>>,
}

impl<F: PrimeField> AllocatedUniPoly<F> {
  /// Setup the univariate polynomial in the constraint system.
  pub fn alloc<CS: ConstraintSystem<F>>(
    mut cs: CS,
    uni_poly: &UniPoly<F>,
  ) -> Result<Self, SynthesisError> {
    let coeffs = uni_poly
      .coeffs
      .iter()
      .enumerate()
      .map(|(i, coeff)| AllocatedNum::alloc(cs.namespace(|| format!("coeff_{i}")), || Ok(*coeff)))
      .collect::<Result<Vec<_>, _>>()?;
    Ok(Self { coeffs })
  }
}

impl<F: PrimeField> AllocatedUniPoly<F> {
  /// Degree == number of coefficients - 1
  ///
  /// Example:
  /// ax^2 + bx + c
  /// coeffs_len == vec![c, b, a].len() == 3
  /// degree = 3 - 1 = 2
  pub fn degree<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    AllocatedNum::alloc(cs.namespace(|| "uni_poly degree"), || {
      Ok(F::from((self.coeffs.len() - 1) as u64))
    })
  }

  /// Evaluation at zero == constant term
  pub fn eval_at_zero<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    AllocatedNum::alloc(cs.namespace(|| "uni_poly eval at zero"), || {
      Ok(*self.coeffs[0].get_value().get()?)
    })
  }

  /// Evaluation at one == sum of all coefficients
  pub fn eval_at_one<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    let mut res = self.coeffs[0].clone();
    for (i, coeff) in self.coeffs.iter().enumerate().skip(1) {
      res = res.add(cs.namespace(|| format!("add coeff_{i}")), coeff)?;
    }
    Ok(res)
  }

  /// Evaluate polynomial at a point
  pub fn evaluate<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    r: &AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    let mut eval = self.coeffs[0].clone();
    let mut power = r.clone();

    for (i, coeff) in self.coeffs.iter().enumerate().skip(1) {
      let term = &power.mul(cs.namespace(|| format!("power * coeff_{i}")), coeff)?;
      eval = eval.add(cs.namespace(|| format!("add new term_{i}")), term)?;
      power = power.mul(cs.namespace(|| format!("update power for {i} step")), r)?;
    }

    Ok(eval)
  }
}

/*
 * ******************
 * Equality Polynomial
 * ******************
 */

/// Represents the Allocated multilinear extension polynomial (MLE) of the equality polynomial $eq(x,e)$, denoted as $\tilde{eq}(x, e)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{eq}(x, e) = \prod_{i=1}^m(e_i * x_i + (1 - e_i) * (1 - x_i))
/// $$
///
/// Each element in the vector `r` corresponds to a component $e_i$, representing a bit from the binary representation of an input value $e$.
/// This polynomial evaluates to 1 if every component $x_i$ equals its corresponding $e_i$, and 0 otherwise.
///
/// For instance, for e = 6 (with a binary representation of 0b110), the vector r would be [1, 1, 0].
#[derive(Debug)]
pub struct AllocatedEqPolynomial<F: PrimeField> {
  pub r: Vec<AllocatedNum<F>>,
}

impl<F: PrimeField> AllocatedEqPolynomial<F> {
  /// Creates a new `EqPolynomial` from a vector of Scalars `r`.
  ///
  /// Each Scalar in `r` corresponds to a bit from the binary representation of an input value `e`.
  pub const fn new(r: Vec<AllocatedNum<F>>) -> Self {
    Self { r }
  }

  /// Evaluates the `EqPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `r`.
  ///
  /// Panics if `rx` and `r` have different lengths.
  pub fn evaluate<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    rx: &[AllocatedNum<F>],
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    // assert_eq!(self.r.len(), rx.len());
    let r_len = AllocatedNum::alloc(cs.namespace(|| "r_len"), || {
      Ok(F::from(self.r.len() as u64))
    })?;

    let rx_len = AllocatedNum::alloc(cs.namespace(|| "rx_len"), || Ok(F::from(rx.len() as u64)))?;

    cs.enforce(
      || "r_len == rx_len",
      |lc| lc + r_len.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + rx_len.get_variable(),
    );

    // (0..rx.len())
    //   .map(|i| self.r[i] * rx[i] + (Scalar::ONE - self.r[i]) * (Scalar::ONE - rx[i]))
    //   .product()

    let mut res = AllocatedNum::alloc(cs.namespace(|| "res"), || Ok(F::ONE))?;

    let one = alloc_one(cs.namespace(|| "one"));

    for (i, r_i) in self.r.iter().enumerate() {
      let rx_i = &rx[i];

      let neg_r_val = alloc_negate(cs.namespace(|| format!("-r_{i}")), r_i)?;
      let neg_rx_i = alloc_negate(cs.namespace(|| format!("-rx_{i}")), rx_i)?;

      // (Scalar::ONE - self.r[i])
      let one_minus_r_i = one.add(cs.namespace(|| format!("1 - r_{i}")), &neg_r_val)?;

      // (Scalar::ONE - rx[i]))
      let one_minus_rx_i = one.add(cs.namespace(|| format!("1 - rx_{i}")), &neg_rx_i)?;

      // self.r[i] * rx[i]
      let r_times_rx = r_i.mul(cs.namespace(|| format!("r_times_rx_{i}")), rx_i)?;

      // (Scalar::ONE - self.r[i]) * (Scalar::ONE - rx[i]))
      let one_minus_r_times_one_minus_rx = one_minus_r_i.mul(
        cs.namespace(|| format!("one_minus_r_times_one_minus_rx_{i}")),
        &one_minus_rx_i,
      )?;

      // self.r[i] * rx[i] + (Scalar::ONE - self.r[i]) * (Scalar::ONE - rx[i]))
      let term = r_times_rx.add(
        cs.namespace(|| format!("r_times_rx_{i} + one_minus_r_times_one_minus_rx_{i}")),
        &one_minus_r_times_one_minus_rx,
      )?;

      // ... .product()
      res = res.mul(cs.namespace(|| format!("res_{i} * term_{i}")), &term)?;
    }
    Ok(res)
  }

  /// Evaluates the `EqPolynomial` from the `2^|r|` points in its domain, without creating an intermediate polynomial
  /// representation.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals_from_points<CS: ConstraintSystem<F>>(
    mut cs: CS,
    r: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let ell = r.len();
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let one = alloc_one(cs.namespace(|| "one"));
    let mut evals = (0..(2_usize).pow(ell as u32))
      .map(|_| zero.clone())
      .collect::<Vec<_>>();

    let mut size_int = 1;
    let mut size = one.clone();

    evals[0] = one.clone();

    for (j, r) in r.iter().rev().enumerate() {
      let (evals_left, evals_right) = evals.split_at_mut(size_int);
      let (evals_right, _) = evals_right.split_at_mut(size_int);

      evals_left
        .iter_mut()
        .zip_eq(evals_right.iter_mut())
        .enumerate()
        .for_each(|(i, (x, y))| {
          *y = x.mul(cs.namespace(|| format!("x * r {i} {j}")), r).unwrap();
          let neg_y = alloc_negate(cs.namespace(|| format!("-y_{i} {j}")), y).unwrap();
          *x = x
            .add(cs.namespace(|| format!("x -= y {i} {j}")), &neg_y)
            .unwrap();
        });

      size_int *= 2;
      size = size.add(cs.namespace(|| format!("size *= 2 {j}")), &size)?;
    }

    Ok(evals)
  }
}

/*
 * ******************
 * Power Polynomial
 * ******************
 */

/// `AllocatedPowPolynomial`: Represents multilinear extension of power polynomials
/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $pow(x,t)$, denoted as $\tilde{pow}(x, t)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{power}(x, t) = \prod_{i=1}^m(1 + (t^{2^i} - 1) * x_i)
/// $$
pub struct AllocatedPowPolynomial<F: PrimeField> {
  eq: AllocatedEqPolynomial<F>,
}

impl<F: PrimeField> AllocatedPowPolynomial<F> {
  /// Create a new instance of an AllocatedPowPolynomial from a vec of AllocatedNums
  pub fn new<CS: ConstraintSystem<F>>(
    mut cs: CS,
    t: &AllocatedNum<F>,
    ell: usize,
  ) -> Result<Self, SynthesisError> {
    // t_pow = [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
    let t_pow = Self::squares(cs.namespace(|| "get_squares"), t, ell)?;

    Ok(Self {
      eq: AllocatedEqPolynomial::new(t_pow),
    })
  }

  /// Create powers the following powers of `t`:
  /// [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
  pub fn squares<CS: ConstraintSystem<F>>(
    mut cs: CS,
    t: &AllocatedNum<F>,
    ell: usize,
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    // Resulting vector of powers
    let mut squares = Vec::new();

    // Push init
    // init = t^{2^0} = t
    squares.push(t.clone());

    // Calculate the rest of the powers
    // Easier to do with for loop instead of iterators for gadgets
    for i in 1..ell {
      let new_t = squares[i - 1].square(cs.namespace(|| format!("t_squared_{i}")))?;
      squares.push(new_t);
    }

    Ok(squares)
  }

  /// Coordinates for PowPolynomial equals the coordinates of the equality polynomial
  pub fn coordinates(self) -> Vec<AllocatedNum<F>> {
    self.eq.r
  }

  /// Creates the evals corresponding to a `PowPolynomial` from an already-existing vector of powers.
  /// `t_pow.len() > ell` must be true.
  pub(crate) fn evals_with_powers<CS: ConstraintSystem<F>>(
    mut cs: CS,
    powers: &[AllocatedNum<F>],
    ell: usize,
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let t_pow = powers[..ell].to_vec();
    AllocatedEqPolynomial::evals_from_points(cs.namespace(|| "EQ::evals_from_points"), &t_pow)
  }

  /// Evaluates the `AllocatedPowPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `t_pow`.
  ///
  /// Panics if `rx` and `t_pow` have different lengths.
  pub fn evaluate<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    rx: &[AllocatedNum<F>],
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    self.eq.evaluate(cs.namespace(|| "evaluate eq poly"), rx)
  }
}

/// Powers gadget
/// Creates a vector of the first `n` powers of `s`.
pub fn alloc_powers<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  s: &AllocatedNum<F>,
  n: usize,
) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
  let mut powers = Vec::new();
  let mut power = s.clone();

  // n^0 = 1
  let alloc_one = AllocatedNum::alloc(cs.namespace(|| "one"), || Ok(F::ONE))?;
  powers.push(alloc_one);

  for i in 1..n {
    powers.push(power.clone());
    power = power.mul(cs.namespace(|| format!("index {i} power * s")), s)?;
  }

  Ok(powers)
}
impl<Scalar: PrimeField> From<AllocatedPowPolynomial<Scalar>> for AllocatedEqPolynomial<Scalar> {
  fn from(polynomial: AllocatedPowPolynomial<Scalar>) -> Self {
    polynomial.eq
  }
}

/*
 * ******************
 * Masked Equality Polynomial
 * ******************
 */

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $eqₘ(x,r)$
/// over n variables, where the first 2^m evaluations are 0.
///
/// The polynomial is defined by the formula:
/// eqₘ(x,r) = eq(x,r) - ( ∏_{0 ≤ i < n-m} (1−rᵢ)(1−xᵢ) )⋅( ∏_{n-m ≤ i < n} (1−rᵢ)(1−xᵢ) + rᵢ⋅xᵢ )
#[derive(Debug)]
pub struct AllocatedMaskedEqPolynomial<'a, Scalar: PrimeField> {
  eq: &'a AllocatedEqPolynomial<Scalar>,
  num_masked_vars: usize,
}

impl<'a, Scalar: PrimeField> AllocatedMaskedEqPolynomial<'a, Scalar> {
  /// Creates a new `MaskedEqPolynomial` from a vector of Scalars `r` of size n, with the number of
  /// masked variables m = `num_masked_vars`.
  pub const fn new(eq: &'a AllocatedEqPolynomial<Scalar>, num_masked_vars: usize) -> Self {
    AllocatedMaskedEqPolynomial {
      eq,
      num_masked_vars,
    }
  }

  /// Evaluates the `MaskedEqPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `r`.
  ///
  /// Panics if `rx` and `r` have different lengths.
  pub fn evaluate<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    rx: &[AllocatedNum<Scalar>],
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let r = &self.eq.r;
    assert_eq!(r.len(), rx.len());
    let split_idx = r.len() - self.num_masked_vars;

    let (r_lo, r_hi) = r.split_at(split_idx);
    let (rx_lo, rx_hi) = rx.split_at(split_idx);

    // let eq_lo = zip_eq(r_lo, rx_lo)
    //   .map(|(r, rx)| *r * rx + (Scalar::ONE - r) * (Scalar::ONE - rx))
    //   .product::<Scalar>();
    let mut eq_lo = alloc_one(cs.namespace(|| "eq_lo"));
    let one = alloc_one(cs.namespace(|| "one"));

    for (i, (r, rx)) in r_lo.iter().zip_eq(rx_lo).enumerate() {
      let neg_r_val = alloc_negate(cs.namespace(|| format!("r_lo: -r_{i}")), r)?;
      let neg_rx_i = alloc_negate(cs.namespace(|| format!("r_lo: -rx_{i}")), rx)?;

      // (Scalar::ONE - self.r[i])
      let one_minus_r_i = one.add(cs.namespace(|| format!("r_lo: 1 - r_{i}")), &neg_r_val)?;

      // (Scalar::ONE - rx[i]))
      let one_minus_rx_i = one.add(cs.namespace(|| format!("r_lo: 1 - rx_{i}")), &neg_rx_i)?;

      // self.r[i] * rx[i]
      let r_times_rx = r.mul(cs.namespace(|| format!("r_lo: r_times_rx_{i}")), rx)?;

      // (Scalar::ONE - self.r[i]) * (Scalar::ONE - rx[i]))
      let one_minus_r_times_one_minus_rx = one_minus_r_i.mul(
        cs.namespace(|| format!("r_lo: one_minus_r_times_one_minus_rx_{i}")),
        &one_minus_rx_i,
      )?;

      // self.r[i] * rx[i] + (Scalar::ONE - self.r[i]) * (Scalar::ONE - rx[i]))
      let term = r_times_rx.add(
        cs.namespace(|| format!("r_lo: r_times_rx_{i} + one_minus_r_times_one_minus_rx_{i}")),
        &one_minus_r_times_one_minus_rx,
      )?;

      // ... .product()
      eq_lo = eq_lo
        .mul(
          cs.namespace(|| format!("r_lo: eq_lo_{i} * term_{i}")),
          &term,
        )
        .unwrap();
    }

    // let eq_hi = zip_eq(r_hi, rx_hi)
    //   .map(|(r, rx)| *r * rx + (Scalar::ONE - r) * (Scalar::ONE - rx))
    //   .product::<Scalar>();

    let mut eq_hi = alloc_one(cs.namespace(|| "eq_hi"));
    for (i, (r, rx)) in r_hi.iter().zip_eq(rx_hi).enumerate() {
      let neg_r_val = alloc_negate(cs.namespace(|| format!("r_hi: -r_{i}")), r)?;
      let neg_rx_i = alloc_negate(cs.namespace(|| format!("r_hi: -rx_{i}")), rx)?;

      // (Scalar::ONE - self.r[i])
      let one_minus_r_i = one.add(cs.namespace(|| format!("r_hi: 1 - r_{i}")), &neg_r_val)?;

      // (Scalar::ONE - rx[i]))
      let one_minus_rx_i = one.add(cs.namespace(|| format!("r_hi: 1 - rx_{i}")), &neg_rx_i)?;

      // self.r[i] * rx[i]
      let r_times_rx = r.mul(cs.namespace(|| format!("r_hi: r_times_rx_{i}")), rx)?;

      // (Scalar::ONE - self.r[i]) * (Scalar::ONE - rx[i]))
      let one_minus_r_times_one_minus_rx = one_minus_r_i.mul(
        cs.namespace(|| format!("r_hi: one_minus_r_times_one_minus_rx_{i}")),
        &one_minus_rx_i,
      )?;

      // self.r[i] * rx[i] + (Scalar::ONE - self.r[i]) * (Scalar::ONE - rx[i]))
      let term = r_times_rx.add(
        cs.namespace(|| format!("r_hi: r_times_rx_{i} + one_minus_r_times_one_minus_rx_{i}")),
        &one_minus_r_times_one_minus_rx,
      )?;

      // ... .product()
      eq_hi = eq_hi.mul(
        cs.namespace(|| format!("r_hi: eq_hi_{i} * term_{i}")),
        &term,
      )?;
    }

    // let mask_lo = zip_eq(r_lo, rx_lo)
    //   .map(|(r, rx)| (Scalar::ONE - r) * (Scalar::ONE - rx))
    //   .product::<Scalar>();

    let mut mask_lo = alloc_one(cs.namespace(|| "mask_lo"));

    for (i, (r, rx)) in r_lo.iter().zip_eq(rx_lo).enumerate() {
      let neg_r_val = alloc_negate(cs.namespace(|| format!("mask_lo: -r_{i}")), r)?;
      let neg_rx_i = alloc_negate(cs.namespace(|| format!("mask_lo: -rx_{i}")), rx)?;

      // (Scalar::ONE - self.r[i])
      let one_minus_r_i = one.add(cs.namespace(|| format!("mask_lo: 1 - r_{i}")), &neg_r_val)?;

      // (Scalar::ONE - rx[i]))
      let one_minus_rx_i = one.add(cs.namespace(|| format!("mask_lo: 1 - rx_{i}")), &neg_rx_i)?;

      // (Scalar::ONE - self.r[i]) * (Scalar::ONE - rx[i]))
      let term = one_minus_r_i.mul(
        cs.namespace(|| format!("mask_lo: one_minus_r_times_one_minus_rx_{i}")),
        &one_minus_rx_i,
      )?;

      // ... .product()
      mask_lo = mask_lo.mul(
        cs.namespace(|| format!("mask_lo: mask_lo_{i} * term_{i}")),
        &term,
      )?;
    }

    // (eq_lo - mask_lo) * eq_hi

    let neg_mask_lo = alloc_negate(cs.namespace(|| "neg_mask_lo"), &mask_lo)?;

    let eq_lo_minus_mask_lo = eq_lo
      .add(cs.namespace(|| "eq_lo + neg_mask_lo"), &neg_mask_lo)
      .unwrap();

    eq_lo_minus_mask_lo.mul(cs.namespace(|| " (eq_lo - mask_lo) * eq_hi"), &eq_hi)
  }
}

pub struct AllocatedIdentityPolynomial<Scalar> {
  ell: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField> AllocatedIdentityPolynomial<Scalar> {
  pub(crate) fn new(ell: usize) -> Self {
    Self {
      ell,
      _p: PhantomData,
    }
  }

  pub fn evaluate<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    r: &[AllocatedNum<Scalar>],
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    // assert_eq!(self.ell, r.len());
    let ell = AllocatedNum::alloc(cs.namespace(|| "ell"), || Ok(Scalar::from(self.ell as u64)))?;
    let r_len = AllocatedNum::alloc(cs.namespace(|| "r_len"), || {
      Ok(Scalar::from(r.len() as u64))
    })?;
    cs.enforce(
      || "ell == r_len",
      |lc| lc + ell.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + r_len.get_variable(),
    );

    // let mut power_of_two = 1_u64;
    let mut power_of_two = alloc_one(cs.namespace(|| "power_of_two"));

    let mut eval = alloc_zero(cs.namespace(|| "eval"));

    // (0..self.ell)
    //   .rev()
    //   .map(|i| {
    //     let result = Scalar::from(power_of_two) * r[i];
    //     power_of_two *= 2;
    //     result
    //   })
    //   .sum()
    for i in (0..self.ell).rev() {
      let result = power_of_two.mul(cs.namespace(|| format!("power_of_two * r_{i}")), &r[i])?;
      power_of_two = power_of_two.add(
        cs.namespace(|| format!("power_of_two double_{i}")),
        &power_of_two,
      )?;
      eval = eval.add(cs.namespace(|| format!("eval + result_{i}")), &result)?;
    }

    Ok(eval)
  }
}

/// Sparse multilinear polynomial, which means the $Z(\cdot)$ is zero at most points.
/// In our context, sparse polynomials are non-zeros over the hypercube at locations that map to "small" integers
/// We exploit this property to implement a time-optimal algorithm
pub struct AllocatedSparsePolynomial<Scalar: PrimeField> {
  num_vars: usize,
  Z: Vec<AllocatedNum<Scalar>>,
}

impl<Scalar: PrimeField> AllocatedSparsePolynomial<Scalar> {
  pub fn new(num_vars: usize, Z: Vec<AllocatedNum<Scalar>>) -> Self {
    Self { num_vars, Z }
  }

  // a time-optimal algorithm to evaluate sparse polynomials
  pub fn evaluate<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    r: &[AllocatedNum<Scalar>],
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    // assert_eq!(self.num_vars, r.len());
    let num_vars = AllocatedNum::alloc(cs.namespace(|| "num_vars"), || {
      Ok(Scalar::from(self.num_vars as u64))
    })?;

    let r_len = AllocatedNum::alloc(cs.namespace(|| "r_len"), || {
      Ok(Scalar::from(r.len() as u64))
    })?;

    cs.enforce(
      || "num_vars == r_len",
      |lc| lc + num_vars.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + r_len.get_variable(),
    );

    let num_vars_z = self.Z.len().next_power_of_two().log_2();
    let chis = AllocatedEqPolynomial::evals_from_points(
      cs.namespace(|| "chis"),
      &r[self.num_vars - 1 - num_vars_z..],
    )?;

    // let eval_partial: Scalar = self
    //   .Z
    //   .iter()
    //   .zip(chis.iter())
    //   .map(|(z, chi)| *z * *chi)
    //   .sum();
    let mut eval_partial = alloc_zero(cs.namespace(|| "eval_partial"));
    for (i, z) in self.Z.iter().enumerate() {
      let term = z.mul(cs.namespace(|| format!("z_{i} * chi_{i}")), &chis[i])?;
      eval_partial =
        eval_partial.add(cs.namespace(|| format!("eval_partial + term_{i}")), &term)?;
    }

    // let common = (0..self.num_vars - 1 - num_vars_z)
    //   .map(|i| (Scalar::ONE - r[i]))
    //   .product::<Scalar>();
    let mut common = alloc_one(cs.namespace(|| "common"));

    for i in 0..self.num_vars - 1 - num_vars_z {
      let one = alloc_one(cs.namespace(|| format!("one_{i}")));
      let neg_r_val = alloc_negate(cs.namespace(|| format!("-r_{i}")), &r[i])?;

      // (Scalar::ONE - self.r[i])
      let one_minus_r_i = one.add(cs.namespace(|| format!("1 - r_{i}")), &neg_r_val)?;

      common = common.mul(
        cs.namespace(|| format!("common_{i} * one_minus_r_{i}")),
        &one_minus_r_i,
      )?;
    }

    // common * eval_partial
    common.mul(cs.namespace(|| "common * eval_partial"), &eval_partial)
  }
}

#[cfg(test)]
mod tests {
  use bellpepper_core::test_cs::TestConstraintSystem;

  use crate::{
    provider::PallasEngine,
    spartan::{polys::power::PowPolynomial, powers},
    traits::Engine,
  };

  use super::*;

  type E = PallasEngine;
  type Fq = <E as Engine>::Scalar;

  /*
   * ******************
   * Power Polynomial
   * ******************
   */
  #[test]
  fn test_squares() -> Result<(), SynthesisError> {
    let mut cs = TestConstraintSystem::<Fq>::new();
    let t = AllocatedNum::alloc(cs.namespace(|| "t"), || Ok(Fq::from(2)))?;

    let squares =
      AllocatedPowPolynomial::<Fq>::squares(cs.namespace(|| "get_squares of t"), &t, 6)?;

    assert_eq!(squares.len(), 6);
    // 2^{2^0} = 2^1 = 2
    assert_eq!(*squares[0].get_value().get()?, Fq::from(2));
    // 2^{2^1} = 2^2 = 4
    assert_eq!(*squares[1].get_value().get()?, Fq::from(4));
    // 2^{2^2} = 2^4 = 16
    assert_eq!(*squares[2].get_value().get()?, Fq::from(16));
    // 2^{2^3} = 2^8 = 256
    assert_eq!(*squares[3].get_value().get()?, Fq::from(256));
    // 2^{2^4} = 2^16 = 65536
    assert_eq!(*squares[4].get_value().get()?, Fq::from(65536));
    // 2^{2^5} = 2^32 = 4294967296
    assert_eq!(*squares[5].get_value().get()?, Fq::from(4294967296));

    assert!(cs.is_satisfied());
    Ok(())
  }

  #[test]
  fn test_squares_with_pow_poly() -> Result<(), SynthesisError> {
    let mut cs = TestConstraintSystem::<Fq>::new();
    let t = AllocatedNum::alloc(cs.namespace(|| "t"), || Ok(Fq::from(2)))?;

    let squares =
      AllocatedPowPolynomial::<Fq>::squares(cs.namespace(|| "get_squares of t"), &t, 6)?;
    let fq_squares = PowPolynomial::<Fq>::squares(&Fq::from(2), 6);
    assert_eq!(squares.len(), 6);
    // 2^{2^0} = 2^1 = 2
    assert_eq!(*squares[0].get_value().get()?, fq_squares[0]);
    // 2^{2^1} = 2^2 = 4
    assert_eq!(*squares[1].get_value().get()?, fq_squares[1]);
    // 2^{2^2} = 2^4 = 16
    assert_eq!(*squares[2].get_value().get()?, fq_squares[2]);
    // 2^{2^3} = 2^8 = 256
    assert_eq!(*squares[3].get_value().get()?, fq_squares[3]);
    // 2^{2^4} = 2^16 = 65536
    assert_eq!(*squares[4].get_value().get()?, fq_squares[4]);
    // 2^{2^5} = 2^32 = 4294967296
    assert_eq!(*squares[5].get_value().get()?, fq_squares[5]);

    assert!(cs.is_satisfied());
    Ok(())
  }

  #[test]
  fn test_powers() -> Result<(), SynthesisError> {
    // Make sure gadget mirrors the actual computation
    let mut cs = TestConstraintSystem::<Fq>::new();
    let s = AllocatedNum::alloc(cs.namespace(|| "s"), || Ok(Fq::from(2))).unwrap();

    let allocated_powers = alloc_powers(cs.namespace(|| "alloc_powers"), &s, 6).unwrap();
    let fq_powers = powers(&Fq::from(2), 6);

    assert_eq!(allocated_powers.len(), 6);

    for (i, allocated_power) in allocated_powers.iter().enumerate() {
      assert_eq!(*allocated_power.get_value().get()?, fq_powers[i]);
    }

    assert!(cs.is_satisfied());
    Ok(())
  }
}
