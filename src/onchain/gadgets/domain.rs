use crate::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;

#[derive(Clone, Debug)]
/// Defines an evaluation domain over a prime field. The domain is a coset of
/// size `1<<dim`.
pub struct AllocatedRadix2Domain<F: PrimeField> {
  /// generator of subgroup g
  pub gen: F,
  /// dimension of evaluation domain, which is log2(size of coset)
  pub dim: u64,
}
impl<F: PrimeField> AllocatedRadix2Domain<F> {
  /// Construct an evaluation domain with the given offset.
  pub fn new(
    gen: F,
    dimension: u64,
  ) -> Self {
    Self {
      gen,
      dim: dimension,
    }
  }
}

#[derive(Clone)]
/// Stores a UV polynomial in evaluation form.
pub struct AllocatedEvaluations<F: PrimeField> {
  /// Evaluations of univariate polynomial over domain
  pub evals: Vec<AllocatedNum<F>>,
  /// Optional Lagrange Interpolator. Useful for lagrange interpolation.
  pub lagrange_interpolator: Option<LagrangeInterpolator<F>>,
  domain: AllocatedRadix2Domain<F>,
}

impl<F: PrimeField> AllocatedEvaluations<F> {
  /// Construct `Self` from evaluations and a domain.
  /// `interpolate` indicates if user wants to interpolate this polynomial
  /// using lagrange interpolation.
  pub fn from_vec_and_domain(
    evaluations: Vec<AllocatedNum<F>>,
    domain: AllocatedRadix2Domain<F>,
    interpolate: bool,
  ) -> Self {
    assert_eq!(
      evaluations.len(),
      1 << domain.dim,
      "evaluations and domain has different dimensions"
    );

    let mut ev = Self {
      evals: evaluations.clone(),
      lagrange_interpolator: None,
      domain,
    };
    if interpolate {
      ev.generate_interpolation_cache();
    }
    ev
  }

  /// Precompute necessary data for Lagrange interpolation, storing either
  /// a precomputed LagrangeInterpolator if the offset is known, or a
  /// cache of subgroup elements otherwise.
  pub fn generate_interpolation_cache(&mut self) {
    // Check if the domain offset has a known value at synthesis-time.
    //
    // If so, we'll build a LagrangeInterpolator right away given the
    // known offset and the evaluation values.
    // Otherwise, we'll precompute the subgroup elements so that they
    // can be used later for a dynamic offset.
      // Gather the known values of the evaluations
      let poly_evaluations_val: Vec<_> = self
        .evals
        .iter()
        .map(|eval| eval.get_value().unwrap_or(F::ZERO))
        .collect();

      // Build the LagrangeInterpolator using the known offset, generator, dimension,
      // and evaluation values.
      let lagrange_interpolator = LagrangeInterpolator::new(
        F::ONE,
        self.domain.gen,
        self.domain.dim,
        poly_evaluations_val,
      );

      self.lagrange_interpolator = Some(lagrange_interpolator);
  }
  /// Compute Lagrange coefficients for each evaluation, given `interpolation_point`.
  /// Only valid if the domain offset is constant (i.e., offset has a known value).
  ///
  /// This adapts the logic seen in the provided code snippet, replacing
  /// FpVar-like operations with bellpepper constraints for AllocatedNum.
  pub fn compute_lagrange_coefficients<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    interpolation_point: &AllocatedNum<F>,
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let lagrange_interpolator = self
            .lagrange_interpolator
            .as_ref()
            .expect("lagrange interpolator has not been initialized. \
            Call `self.generate_interpolation_cache` first or set `interpolate` to true in constructor. ");

    // Evaluate the vanishing polynomial constraints at interpolation_point:
    let vp_t = lagrange_interpolator
      .domain_vp
      .evaluate_constraints(cs.namespace(|| "vanishing_poly_eval"), interpolation_point)?;

    let t_val = interpolation_point.get_value();

    // For each domain element, we create a "part" (A_element) in the circuit, then enforce:
    // A_element * lag_coeff = vp_t,
    // where A_element = v_inv_elems[i] * t - v_inv_elems[i] * all_domain_elems[i].
    // In the circuit, we handle these as allocated variables and linear constraints.
    let mut lagrange_coeffs = Vec::with_capacity(lagrange_interpolator.domain_order);

    for (i, (&dom_elem, &v_inv)) in lagrange_interpolator
      .all_domain_elems
      .iter()
      .zip(lagrange_interpolator.v_inv_elems.iter())
      .enumerate()
    {
      // Precompute the expected numeric value of a_element if available
      let a_element_val = match (t_val, Some(dom_elem), Some(v_inv)) {
        (Some(tv), Some(de), Some(vi)) => Some(tv * vi - de * vi),
        _ => None,
      };

      // Allocate a_element in the circuit
      let a_element = AllocatedNum::alloc(cs.namespace(|| format!("a_element_{i}")), || {
        a_element_val.ok_or(SynthesisError::AssignmentMissing)
      })?;

      // We next require a_element * lag_coeff = vp_t.
      // The value of lag_coeff is also allocated, and we enforce the product constraint.
      let lag_coeff_val = match (a_element_val, vp_t.get_value()) {
        (Some(ae), Some(vp_t_val)) => {
          // If a_element is non-zero in the actual assignment, we can invert.
          // The code snippet asserts a_element != 0 if the point is not in the coset.
          let inv_ae = ae.invert().expect("a_element must be invertible");
          Some(vp_t_val * inv_ae)
        }
        _ => None,
      };

      let lag_coeff = AllocatedNum::alloc(cs.namespace(|| format!("lag_coeff_{i}")), || {
        lag_coeff_val.ok_or(SynthesisError::AssignmentMissing)
      })?;

      // Now enforce: a_element * lag_coeff = vp_t
      cs.enforce(
        || format!("a_element_{i} * lag_coeff_{i} = vp_t"),
        |lc| lc + a_element.get_variable(),
        |lc| lc + lag_coeff.get_variable(),
        |lc| lc + vp_t.get_variable(),
      );

      lagrange_coeffs.push(lag_coeff);
    }

    Ok(lagrange_coeffs)
  }

  /// Returns a gadget that interpolates and then evaluates at `interpolation_point`.
  /// If the domain offset is constant (known at synthesis time), this will use
  /// fewer constraints via compute_lagrange_coefficients. Otherwise, it will do
  /// a more general approach.
  pub fn interpolate_and_evaluate<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    interpolation_point: &AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
      self.lagrange_interpolate_with_constant_offset(&mut cs, interpolation_point)
  }

  /// Interpolate with constant offset. Uses fewer constraints by creating
  /// a Lagrange coefficient for each domain element in the circuit, then
  /// summing up the product of that coefficient and the stored evaluation.
  fn lagrange_interpolate_with_constant_offset<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    interpolation_point: &AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    let lagrange_interpolator = self
            .lagrange_interpolator
            .as_ref()
            .expect("lagrange interpolator has not been initialized. Call `generate_interpolation_cache` first or set `interpolate` to true.");

    // Compute the Lagrange coefficients:
    let lagrange_coeffs = self.compute_lagrange_coefficients(
      cs.namespace(|| "compute_lagrange_coeffs"),
      interpolation_point,
    )?;

    // Sum up each coefficient * eval, taking only as many values
    // as the domain_order indicates.
    let domain_order = lagrange_interpolator.domain_order;
    let mut accum = AllocatedNum::alloc(cs.namespace(|| "accum_init"), || Ok(F::ZERO))?;
    for (i, (coeff, eval)) in lagrange_coeffs
      .into_iter()
      .zip(&self.evals)
      .take(domain_order)
      .enumerate()
    {
      let product = eval.mul(cs.namespace(|| format!("eval_{i} * coeff_{i}")), &coeff)?;
      accum = accum.add(cs.namespace(|| format!("accum_add_{i}")), &product)?;
    }

    Ok(accum)
  }
}

/// A simple vanishing polynomial.
/// Z_H(x) = x^m - offset^m, where m = 1 << dim
#[derive(Clone, Debug)]
pub struct VanishingPolynomial<F: PrimeField> {
  /// h^|H|
  pub constant_term: F,
  /// log_2(|H|)
  pub dim_h: u64,
  /// |H|
  pub order_h: u64,
}

impl<F: PrimeField> VanishingPolynomial<F> {
  /// returns a VanishingPolynomial of coset `H = h<g>`.
  pub fn new(offset: F, dim_h: u64) -> Self {
    let order_h = 1 << dim_h;
    VanishingPolynomial {
      constant_term: offset.pow([order_h]),
      dim_h,
      order_h,
    }
  }

  /// Evaluates the vanishing polynomial without generating the constraints.
  pub fn evaluate(&self, x: &F) -> F {
    let mut result = x.pow([self.order_h]);
    result -= &self.constant_term;
    result
  }

  /// Evaluates the constraints and just gives you the gadget for the result.
  /// Caution for use in holographic lincheck: The output has 2 entries in one
  /// matrix
  pub fn evaluate_constraints<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    x: &AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    if self.dim_h == 1 {
      // Allocate the result
      let res = AllocatedNum::alloc(cs.namespace(|| "trivial case"), || {
        Ok(F::ZERO)
      })?;
      return Ok(res);
    }

    // Compute x^(2^dim_h) - offset^(2^dim_h).
    // We do repeated squaring to get x^(2^dim_h).
    let mut cur = x.square(cs.namespace(|| "square x"))?;
    for i in 1..self.dim_h {
      cur = cur.square(cs.namespace(|| format!("square iteration {}", i)))?;
    }

    // Allocate the constant term offset^(2^dim_h) in the circuit
    let offset_term = AllocatedNum::alloc(cs.namespace(|| "allocate constant_term"), || {
      Ok(self.constant_term)
    })?;

    // Subtract the constant term from the result
    let res = AllocatedNum::alloc(cs.namespace(|| "subtract constant_term"), || {
      cur.get_value().and_then(|a| offset_term.get_value().map(|b| a - b)).ok_or(SynthesisError::AssignmentMissing)
    })?;

    // Enforce: a - b = res
    cs.enforce(
      || "subtract constant_term",
      |lc| lc + cur.get_variable() - offset_term.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + res.get_variable(),
    );

    Ok(res)
  }
}

/// Performs an in-place batch inversion of the elements of `vals`,
/// then multiplies each element by the scalar `c`.
fn batch_inversion_and_mul<F: PrimeField>(vals: &mut [F], c: &F) {
  // We use an algorithm that first accumulates the product of all inputs,
  // then invert that product once, and finally walk back through the inputs
  // to compute the individual inverses.
  let mut acc = F::ONE;
  let mut scratch = Vec::with_capacity(vals.len());
  scratch.resize(vals.len(), F::ONE);

  // Forward pass: compute partial products
  for (i, v) in vals.iter().enumerate() {
    scratch[i] = acc;
    acc.mul_assign(v);
  }

  // acc now holds the product of all vals
  // Compute the inverse of the product
  acc = acc.invert().unwrap();

  // Backward pass: compute each inverse using the product inverse
  // and partial products from the forward pass.
  for (i, v) in vals.iter_mut().enumerate().rev() {
    let tmp = acc * scratch[i]; // inverse of the current element
    acc.mul_assign(*v); // update acc for the next iteration
                        // multiply the element's inverse by c
    *v = tmp * *c;
  }
}

/// Struct describing Lagrange interpolation for a multiplicative coset I,
/// with |I| a power of 2.
#[derive(Clone)]
pub struct LagrangeInterpolator<F: PrimeField> {
  pub(crate) domain_order: usize,
  pub(crate) all_domain_elems: Vec<F>,
  pub(crate) v_inv_elems: Vec<F>,
  pub(crate) domain_vp: VanishingPolynomial<F>,
  pub(crate) poly_evaluations: Vec<F>,
}

impl<F: PrimeField> LagrangeInterpolator<F> {
  /// Returns a lagrange interpolator, given the domain specification.
  ///
  /// domain_offset = h,
  /// domain_generator = g,
  /// domain_dim = log2 of the domain size,
  /// poly_evaluations = f(h), f(h*g), f(h*g^2), ...
  pub fn new(
    domain_offset: F,
    domain_generator: F,
    domain_dim: u64,
    poly_evaluations: Vec<F>,
  ) -> Self {
    let domain_order = 1 << domain_dim;
    assert_eq!(poly_evaluations.len(), domain_order);

    // Collect all elements of the domain: h, h*g, h*g^2, ...
    let mut cur_elem = domain_offset;
    let mut all_domain_elems = vec![domain_offset];
    for _ in 1..domain_order {
      cur_elem.mul_assign(&domain_generator);
      all_domain_elems.push(cur_elem);
    }

    // We will compute v_inv[i] = 1 / ∏_{j != i} [h*g^i - h*g^j].
    // A known relation allows us to do this with a short loop:
    //   v_inv[0] = m * h^(m-1)
    //   v_inv[i+1] = v_inv[i] * g_inv
    let g_inv = domain_generator
      .invert()
      .expect("domain_generator must be invertible");
    let m = F::from(domain_order as u64);
    // v_inv[0] = m * h^(domain_order-1)
    let mut v_inv_i = m * domain_offset.pow_vartime([((domain_order - 1) as u64)]);
    let mut v_inv_elems: Vec<F> = Vec::with_capacity(domain_order);
    for _ in 0..domain_order {
      v_inv_elems.push(v_inv_i);
      v_inv_i.mul_assign(g_inv);
    }

    // Build the vanishing polynomial
    let vp = VanishingPolynomial::new(domain_offset, domain_dim);

    Self {
      domain_order,
      all_domain_elems,
      v_inv_elems,
      domain_vp: vp,
      poly_evaluations,
    }
  }

  /// Computes the individual Lagrange coefficients at `interpolation_point`.
  /// That is, compute L_{i}(t) for each i in [0..domain_order], where:
  /// L_{i,H}(t) = Z_H(t) * v_inv_elems[i] / (t - h*g^i).
  pub(crate) fn compute_lagrange_coefficients(&self, interpolation_point: F) -> Vec<F> {
    let mut inverted_lagrange_coeffs = Vec::with_capacity(self.domain_order);
    // For each element in the domain, we multiply v_inv_elems[i] by (t - domain_elem)
    // so that we can invert them all at once, then multiply by Z_H(t).
    for i in 0..self.domain_order {
      let l = self.v_inv_elems[i]; // v_inv
      let r = self.all_domain_elems[i];
      // We'll invert l*(t - r) in a batch
      inverted_lagrange_coeffs.push(l * (interpolation_point - r));
    }
    // Evaluate Z_H(t)
    let vp_t = self.domain_vp.evaluate(&interpolation_point);

    // Perform the batch inversion, then multiply by Z_H(t)
    batch_inversion_and_mul(&mut inverted_lagrange_coeffs, &vp_t);

    inverted_lagrange_coeffs
  }

  /// Interpolates the polynomial at `interpolation_point`.
  /// That is, compute ∑ f(h*g^i)*L_{i,H}(t).
  pub fn interpolate(&self, interpolation_point: F) -> F {
    let lagrange_coeffs = self.compute_lagrange_coefficients(interpolation_point);
    let mut interpolation = F::ZERO;
    for (i, coeff) in lagrange_coeffs.iter().enumerate().take(self.domain_order) {
      interpolation.add_assign(&(*coeff * self.poly_evaluations[i]));
    }
    interpolation
  }
}