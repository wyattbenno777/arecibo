use crate::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;

#[derive(Clone, Debug)]
/// Defines an evaluation domain over a prime field. The domain is a coset of
/// size `1<<dim`.
pub struct AllocatedRadix2Domain<F: PrimeField> {
  /// generator of subgroup g
  pub gen: F,
  /// index of the quotient group (i.e. the `offset`)
  offset: AllocatedNum<F>,
  /// dimension of evaluation domain, which is log2(size of coset)
  pub dim: u64,
}
impl<F: PrimeField> AllocatedRadix2Domain<F> {
  /// Construct an evaluation domain with the given offset.
  pub fn new<CS>(
    mut cs: CS,
    gen: F,
    dimension: u64,
    offset: AllocatedNum<F>,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<F>,
  {
    // Enforce that the offset is not zero

    // Allocate 1/(offset - 0) to prove offset ≠ 0
    let inv = cs.alloc(
      || "inverse",
      || {
        let offset_val = offset.get_value().unwrap();
        Ok(offset_val.invert().unwrap())
      },
    )?;

    // Enforce that offset * inv = 1, which is only possible if offset ≠ 0
    // println!("offset * inv = 1: {:?}", 
    //   offset.get_value().and_then(|a| inv.().map(|b| a * b)) == Some(F::ONE));
    // cs.enforce(
    //   || "inverse exists",
    //   |lc| lc + offset.get_variable(),
    //   |lc| lc + inv,
    //   |lc| lc + CS::one(),
    // );
    cs.enforce(
      || "offset * inv = 1",
      |lc| lc + offset.get_variable() - inv,
      |lc| lc,
      |lc| lc,
    );

    Ok(Self {
      gen,
      offset,
      dim: dimension,
    })
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
  /// Contains all domain elements of `domain.base_domain`.
  ///
  /// This is a cache for lagrange interpolation when offset is non-constant.
  /// Will be `None` if offset is constant or `interpolate` is set to
  /// `false`.
  subgroup_points: Option<Vec<F>>,
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
      subgroup_points: None,
    };
    if interpolate {
      ev.generate_interpolation_cache();
    }
    println!("Evaluations == ev.evals: {:?}", evaluations == ev.evals);
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
    if let Some(offset_val) = self.domain.offset.get_value() {
      // Gather the known values of the evaluations
      let poly_evaluations_val: Vec<_> = self
        .evals
        .iter()
        .map(|eval| eval.get_value().unwrap_or(F::ZERO))
        .collect();

      // Build the LagrangeInterpolator using the known offset, generator, dimension,
      // and evaluation values.
      let lagrange_interpolator = LagrangeInterpolator::new(
        offset_val,
        self.domain.gen,
        self.domain.dim,
        poly_evaluations_val,
      );

      self.lagrange_interpolator = Some(lagrange_interpolator);
    } else {
      // The offset is not a known constant. We will compute and store
      // all subgroup elements h*g^k for k in [0..(1 << dim)], so that
      // interpolation can be performed once the offset is determined.
      let size = 1 << self.domain.dim;
      let mut subgroup_points = Vec::with_capacity(size);

      // The underlying code treats the offset as h, but since we
      // can't finalize interpolation now, we at least store the generator powers.
      // Typically, we would want to multiply each by the unknown offset,
      // but that can't be done if offset is not known yet.
      let mut cur_elem = F::ONE;
      subgroup_points.push(cur_elem);
      for _ in 1..size {
        cur_elem *= self.domain.gen;
        subgroup_points.push(cur_elem);
      }

      self.subgroup_points = Some(subgroup_points);
    }
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
      if a_element.get_value().and_then(|a| lag_coeff.get_value().map(|b| a * b)) != vp_t.get_value() {
      println!("a_element * lag_coeff = vp_t: {:?}", 
        a_element.get_value().and_then(|a| lag_coeff.get_value().map(|b| a * b)) == vp_t.get_value());
      }
      // cs.enforce(
      //   || format!("a_element_{i} * lag_coeff_{i} = vp_t"),
      //   |lc| lc + a_element.get_variable(),
      //   |lc| lc + lag_coeff.get_variable(),
      //   |lc| lc + vp_t.get_variable(),
      // );
      cs.enforce(
        || format!("a_element_{i} * lag_coeff_{i} = vp_t"),
        |lc| lc + a_element.get_variable() - lag_coeff.get_variable() + vp_t.get_variable(),
        |lc| lc,
        |lc| lc,
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
    cs: CS,
    interpolation_point: &AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    // If offset is known at synthesis time, use optimized approach.
    if self.domain.offset.get_value().is_some() {
      println!("offset is known at synthesis time");
      self.lagrange_interpolate_with_constant_offset(cs, interpolation_point)
    } else {
      println!("offset is not known at synthesis time");
      self.lagrange_interpolate_with_non_constant_offset(cs, interpolation_point)
    }
  }

  /// Interpolate with constant offset. Uses fewer constraints by creating
  /// a Lagrange coefficient for each domain element in the circuit, then
  /// summing up the product of that coefficient and the stored evaluation.
  fn lagrange_interpolate_with_constant_offset<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
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

  /// Interpolate with non-constant offset. We do not know the offset value
  /// at synthesis time, so we rely on subgroup_points for the base coset powers.
  /// The idea is similar to the snippet provided, but adapted to bellpepper's
  /// AllocatedNum/ConstraintSystem interface.
  fn lagrange_interpolate_with_non_constant_offset<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    interpolation_point: &AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    // We need the subgroup_points precomputed, which store the generator powers.
    // (None indicates this was never cached.)
    let subgroup_points = self
      .subgroup_points
      .as_ref()
      .expect("Interpolation cache missing: call generate_interpolation_cache() first");

    // The domain dimension (log2), so the size is 2^dim.
    let domain_size = 1_usize << self.domain.dim;

    // We now compute offset^domain_size and alpha^domain_size.
    // We'll do repeated squaring for both. This mirrors the snippet's code:
    //   alpha^size - offset^size
    // Then we'll enforce that they're not equal (vanishing polynomial != 0).
    let offset_to_size = self.exp_power_of_two(
      cs.namespace(|| "offset^size"),
      &self.domain.offset,
      domain_size,
    )?;
    let alpha_to_size = self.exp_power_of_two(
      cs.namespace(|| "alpha^size"),
      interpolation_point,
      domain_size,
    )?;

    // lhs_numerator = alpha^size - offset^size
    let lhs_numerator_val = match (alpha_to_size.get_value(), offset_to_size.get_value()) {
      (Some(a), Some(b)) => Some(a - b),
      _ => None,
    };
    let lhs_numerator = AllocatedNum::alloc(cs.namespace(|| "lhs_numerator"), || {
      lhs_numerator_val.ok_or(SynthesisError::AssignmentMissing)
    })?;
    // Enforce alpha_to_size - offset_to_size = lhs_numerator
    println!("alpha_to_size - offset_to_size = lhs_numerator: {:?}", 
      alpha_to_size.get_value().and_then(|a| offset_to_size.get_value().map(|b| a - b)) == lhs_numerator.get_value());
    // cs.enforce(
    //   || "lhs_numerator = alpha^size - offset^size",
    //   |lc| lc + alpha_to_size.get_variable() - offset_to_size.get_variable(),
    //   |lc| lc + CS::one(),
    //   |lc| lc + lhs_numerator.get_variable(),
    // );
    cs.enforce(
      || "lhs_numerator = alpha^size - offset^size",
      |lc| lc + alpha_to_size.get_variable() - offset_to_size.get_variable() + lhs_numerator.get_variable(),
      |lc| lc,
      |lc| lc,
    );

    // Make sure lhs_numerator != 0, so alpha isn't in the multiplicative coset.
    // We'll do a standard "inverse exists" trick: allocate an inverse, multiply,
    // and enforce result  = 1.
    let inverse_numerator = AllocatedNum::alloc(cs.namespace(|| "inverse lhs_numerator"), || {
      let val = lhs_numerator_val.ok_or(SynthesisError::AssignmentMissing)?;
      // If val = 0 in the real assignment, it would fail the proof,
      // so we assume it's invertible.
      Ok(val.invert().expect("lhs_numerator must be invertible"))
    })?;
    println!("lhs_numerator * inv != 0: {:?}", 
      lhs_numerator.get_value().and_then(|a| inverse_numerator.get_value().map(|b| a * b)) == Some(F::ONE));
    // cs.enforce(
    //   || "lhs_numerator * inv != 0",
    //   |lc| lc + lhs_numerator.get_variable(),
    //   |lc| lc + inverse_numerator.get_variable(),
    //   |lc| lc + CS::one(),
    // );
    cs.enforce(
      || "lhs_numerator * inv != 0",
      |lc| lc + lhs_numerator.get_variable() - inverse_numerator.get_variable(),
      |lc| lc,
      |lc| lc,
    );

    // Now compute lhs_denominator = offset^size * domain_size
    // (The snippet uses: size * offset^size.)
    let domain_size_as_fe = F::from(domain_size as u64);
    let domain_size_alloc = AllocatedNum::alloc(cs.namespace(|| "domain_size_alloc"), || {
      Ok(domain_size_as_fe)
    })?;
    // No constraints needed if it's a known constant, but for clarity we do:
    // Enforce domain_size_alloc is indeed domain_size_as_fe if you like, but often left as a constant.

    let lhs_denominator = offset_to_size.mul(
      cs.namespace(|| "lhs_denominator = offset^size * domain_size"),
      &domain_size_alloc,
    )?;

    // Now invert lhs_denominator to define lhs = lhs_numerator / lhs_denominator
    // We'll do the standard trick again.
    let lhs_denominator_val = match (offset_to_size.get_value(), domain_size_alloc.get_value()) {
      (Some(o), Some(ds)) => Some(o * ds),
      _ => None,
    };
    let inv_lhs_denom = AllocatedNum::alloc(cs.namespace(|| "inverse lhs_denominator"), || {
      let val = lhs_denominator_val.ok_or(SynthesisError::AssignmentMissing)?;
      Ok(val.invert().expect("lhs_denominator must be invertible"))
    })?;
    // Enforce that lhs_denominator * inv_lhs_denom = 1
    println!("lhs_denominator * inv_lhs_denom = 1: {:?}", 
      lhs_denominator.get_value().and_then(|a| inv_lhs_denom.get_value().map(|b| a * b)) == Some(F::ONE));
    // cs.enforce(
    //   || "check inv_lhs_denom",
    //   |lc| lc + lhs_denominator.get_variable(),
    //   |lc| lc + inv_lhs_denom.get_variable(),
    //   |lc| lc + CS::one(),
    // );
    cs.enforce(
      || "check inv_lhs_denom",
      |lc| lc + lhs_denominator.get_variable() - inv_lhs_denom.get_variable(),
      |lc| lc,
      |lc| lc,
    );

    // so lhs = lhs_numerator * inv_lhs_denom
    let lhs = lhs_numerator.mul(cs.namespace(|| "lhs"), &inv_lhs_denom)?;

    // Next we define alpha_coset_offset_inv = alpha / offset.
    // We'll do the same approach: offset is known nonzero, so we can enforce alpha * offset_inv.
    // We already enforced offset != 0 in AllocatedRadix2Domain::new.
    let offset_inv = AllocatedNum::alloc(cs.namespace(|| "offset_inv"), || {
      let off_val = self
        .domain
        .offset
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(off_val.invert().expect("offset must be invertible"))
    })?;
    // offset * offset_inv = 1
    println!("offset * offset_inv = 1: {:?}", 
      self.domain.offset.get_value().and_then(|a| offset_inv.get_value().map(|b| a * b)) == Some(F::ONE));
    // cs.enforce(
    //   || "offset * offset_inv = 1",
    //   |lc| lc + self.domain.offset.get_variable(),
    //   |lc| lc + offset_inv.get_variable(),
    //   |lc| lc + CS::one(),
    // );
    cs.enforce(
      || "check inv_lhs_denom",
      |lc| lc + self.domain.offset.get_variable() - offset_inv.get_variable(),
      |lc| lc,
      |lc| lc,
    );
    let alpha_coset_offset_inv = interpolation_point.mul(
      cs.namespace(|| "alpha_coset_offset_inv_unnorm"),
      &offset_inv,
    )?;

    // We'll accumulate the final interpolation result in accum.
    // accum = sum_{i=0..domain_size-1} (self.evals[i] * L_i(alpha)),
    // where L_i(alpha) = lhs / (alpha_coset_offset_inv * subgroup_points[i]^-1 - 1).
    let mut accum = AllocatedNum::alloc(cs.namespace(|| "accum_init"), || Ok(F::ZERO))?;

    for (i, eval) in self.evals.iter().enumerate() {
      // In the snippet, "subgroup_point_inv" is (1 / subgroup_points[i]) out-of-circuit.
      // Here we only store the direct powers, not the inverses. If the code that built
      // subgroup_points also built inverses (like the snippet implies),
      // you can fetch them. Otherwise, you'd need to invert them here similarly.
      let subgroup_element = subgroup_points[i];

      // We'll allocate the subgroup_point_inv for clarity.
      let sp_inv = AllocatedNum::alloc(cs.namespace(|| format!("sp_inv_{i}")), || {
        Ok(
          subgroup_element
            .invert()
            .expect("subgroup_element must be invertible"),
        )
      })?;
      // Enforce subgroup_points[i] * sp_inv = 1
      let sp_const = AllocatedNum::alloc(cs.namespace(|| format!("sp_const_{i}")), || {
        Ok(subgroup_element)
      })?;
      // println!("subgroup_points[i] * sp_inv = 1: {:?}", 
      //   subgroup_element.get_value().and_then(|a| sp_inv.get_value().map(|b| a * b)) == Some(F::ONE));
      // cs.enforce(
      //   || format!("check sp_inv_{i}"),
      //   |lc| lc + sp_const.get_variable(),
      //   |lc| lc + sp_inv.get_variable(),
      //   |lc| lc + CS::one(),
      // );
      cs.enforce(
        || format!("check sp_inv_{i}"),
        |lc| lc + sp_const.get_variable() - sp_inv.get_variable(),
        |lc| lc,
        |lc| lc,
      );

      // Now compute lag_denom = (alpha_coset_offset_inv * sp_inv) - 1
      // We'll allocate alpha_coset_offset_inv * sp_inv first:
      let alpha_sp_inv =
        alpha_coset_offset_inv.mul(cs.namespace(|| format!("alpha_sp_inv_{i}")), &sp_inv)?;
      let lag_denom_val = match alpha_sp_inv.get_value() {
        Some(val) => Some(val - F::ONE),
        None => None,
      };
      let lag_denom = AllocatedNum::alloc(cs.namespace(|| format!("lag_denom_{i}")), || {
        lag_denom_val.ok_or(SynthesisError::AssignmentMissing)
      })?;
      // println!("alpha_sp_inv - 1 = lag_denom: {:?}", 
      //   alpha_sp_inv.get_value().and_then(|a| lag_denom.get_value().map(|b| a - b - F::ONE)) == Some(F::ZERO));
      // cs.enforce(
      //   || format!("lag_denom_{i} = alpha_sp_inv_{i} - 1"),
      //   |lc| lc + alpha_sp_inv.get_variable() - CS::one(),
      //   |lc| lc + CS::one(),
      //   |lc| lc + lag_denom.get_variable(),
      // );
      cs.enforce(
        || format!("lag_denom_{i} = alpha_sp_inv_{i} - 1"),
        |lc| lc + alpha_sp_inv.get_variable() - lag_denom.get_variable(),
        |lc| lc,
        |lc| lc,
      );

      // inverse of lag_denom
      let inv_lag_denom =
        AllocatedNum::alloc(cs.namespace(|| format!("inv_lag_denom_{i}")), || {
          let val = lag_denom_val.ok_or(SynthesisError::AssignmentMissing)?;
          Ok(val.invert().expect("lag_denom must be invertible"))
        })?;
      // println!("lag_denom * inv_lag_denom = 1: {:?}", 
      //   lag_denom.get_value().and_then(|a| inv_lag_denom.get_value().map(|b| a * b)) == Some(F::ONE));
      // cs.enforce(
      //   || format!("lag_denom * inv = 1_{i}"),
      //   |lc| lc + lag_denom.get_variable(),
      //   |lc| lc + inv_lag_denom.get_variable(),
      //   |lc| lc + CS::one(),
      // );
      cs.enforce(
        || format!("lag_denom_{i} = alpha_sp_inv_{i} - 1"),
        |lc| lc + lag_denom.get_variable() - inv_lag_denom.get_variable(),
        |lc| lc,
        |lc| lc,
      );


      // L_i(alpha) = lhs * inv_lag_denom
      let lag_coeff = lhs.mul(cs.namespace(|| format!("lag_coeff_{i}")), &inv_lag_denom)?;

      // Multiply the evaluation by the lagrange coefficient
      let product = eval.mul(cs.namespace(|| format!("eval * lag_coeff_{i}")), &lag_coeff)?;
      // Add to accum
      accum = accum.add(cs.namespace(|| format!("accum_add_{i}")), &product)?;
    }

    Ok(accum)
  }

  /// A helper for exponentiating an AllocatedNum<F> by 2^power in-circuit via repeated squaring.
  fn exp_power_of_two<CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    base: &AllocatedNum<F>,
    power: usize,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    if power == 0 {
      // x^0 = 1
      let one = AllocatedNum::alloc(cs.namespace(|| "constant one"), || Ok(F::ONE))?;
      // Usually you'd skip constraints for a known constant, but you can store it as a variable if needed.
      return Ok(one);
    }
    let mut result = base.clone();
    for i in 1..power {
      result = result.square(cs.namespace(|| format!("square iteration {i}")))?;
    }
    Ok(result)
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
    let vp = VanishingPolynomial {
      constant_term: offset.pow([order_h]),
      dim_h,
      order_h,
    };
    vp
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
      // TODO: This looks like an overkill.
      // In the trivial domain size case, this polynomial simplifies out
      // (x - x = 0)
      let val = match x.get_value() {
        Some(x_v) => Ok(x_v - x_v),
        _ => Err(SynthesisError::AssignmentMissing),
      }?;

      // Allocate the result
      let res = AllocatedNum::alloc(cs.namespace(|| format!("trivial case")), || Ok(val))?;

      // Enforce: a - b = res
      println!("x - x = res: {:?}", 
        x.get_value().and_then(|a| x.get_value().map(|b| a - b)) == res.get_value());
      // cs.enforce(
      //   || "trivial case",
      //   |lc| lc + x.get_variable() - x.get_variable(),
      //   |lc| lc + CS::one(),
      //   |lc| lc + res.get_variable(),
      // );
      cs.enforce(
        || "trivial case",
        |lc| lc + x.get_variable() - x.get_variable() + res.get_variable(),
        |lc| lc,
        |lc| lc,
      );
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
    let val = match (cur.get_value(), offset_term.get_value()) {
      (Some(cur_v), Some(offset_term_v)) => Ok(cur_v - offset_term_v),
      _ => Err(SynthesisError::AssignmentMissing),
    }?;

    let res = AllocatedNum::alloc(cs.namespace(|| "subtract constant_term"), || Ok(val))?;

    // Enforce: a - b = res
    println!("x^(2^dim_h) - offset^(2^dim_h) = res: {:?}", 
      cur.get_value().and_then(|a| offset_term.get_value().map(|b| a - b)) == res.get_value());
    // cs.enforce(
    //   || "subtract constant_term",
    //   |lc| lc + cur.get_variable() - offset_term.get_variable(),
    //   |lc| lc + CS::one(),
    //   |lc| lc + res.get_variable(),
    // );
    cs.enforce(
      || "subtract constant_term",
      |lc| lc + cur.get_variable() - offset_term.get_variable() + res.get_variable(),
      |lc| lc,
      |lc| lc,
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
    for i in 0..self.domain_order {
      interpolation.add_assign(&(lagrange_coeffs[i] * self.poly_evaluations[i]));
    }
    interpolation
  }
}