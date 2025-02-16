//! This module implements circuits that incrementally compute the grand products
//! derived from the RoK from memory checks to grand product checks

use ff::PrimeField;
use itertools::Itertools;

use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{
    alloc_one, alloc_zero,
    int::{add, enforce_equal, enforce_lt_32, mul},
    nebula::{alloc_avt_tuple, countable_hash, randomized_hash_func},
  },
  hypernova::rs::StepCircuit,
};

/// Maximum number of memory ops allowed per step of the zkVM
pub const MEMORY_OPS_PER_STEP: usize = 8;

/// Circuit to compute multiset hashes of (RS, WS)
#[derive(Clone, Debug)]
pub struct OpsCircuit {
  RS: Vec<(usize, u64, u64)>, // Vec<(a, v, t)>
  WS: Vec<(usize, u64, u64)>, // Vec<(a, v, t)>
}

impl<F> StepCircuit<F> for OpsCircuit
where
  F: PrimeField + PartialOrd,
{
  fn arity(&self) -> usize {
    6
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let (gamma, alpha, mut gts, mut h_rs, mut h_ws, size) = {
      (
        z[0].clone(),
        z[1].clone(),
        z[2].clone(),
        z[3].clone(),
        z[4].clone(),
        z[5].clone(),
      )
    };
    let one = alloc_one(cs.namespace(|| "one"));
    // Used to assert |RS| = |WS|
    let mut RS_size_count = alloc_zero(cs.namespace(|| "RS_size_count"));
    let mut WS_size_count = alloc_zero(cs.namespace(|| "WS_size_count"));

    // 2. for i in 0..|RS|
    for (i, (rs, ws)) in self.RS.iter().zip_eq(self.WS.iter()).enumerate() {
      // (a) (a,v,rt) ← RS[i]
      let (r_addr, r_val, r_ts) = alloc_avt_tuple(cs.namespace(|| format!("rs{i}")), *rs)?;

      // (b) (a′,v′,wt) ← WS[i]
      let (w_addr, w_val, w_ts) = alloc_avt_tuple(cs.namespace(|| format!("ws{i}")), *ws)?;

      // (c) gts ← gts + 1
      gts = add(cs.namespace(|| format!("{i},  gts ← gts + 1")), &gts, &one)?;

      // (d) assert rt < ts
      enforce_lt_32(cs.namespace(|| "enforce_lt_32"), &r_ts, &gts)?;

      // (e) assert wt = ts
      enforce_equal(cs, || format!("{i} assert wt = ts"), &w_ts, &gts);

      // Get Hash(gamma, alpha, a, v, rt)
      let hash_rs = randomized_hash_func(
        cs.namespace(|| format!("{i}, Hash(gamma, alpha, a, v, rt)")),
        &r_addr,
        &r_val,
        &r_ts,
        &gamma,
        &alpha,
      )?;

      // Add to RS size count
      let RS_counting_el =
        countable_hash(cs.namespace(|| format!("RS_counting_el_{i}")), &hash_rs)?;
      RS_size_count = add(
        cs.namespace(|| format!("{i}, RS_size_count ← RS_size_count + RS_counting_el")),
        &RS_size_count,
        &RS_counting_el,
      )?;

      // (f) h_RS ← h_RS · Hash(gamma, alpha, a, v, rt)
      h_rs = mul(
        cs.namespace(|| format!("{i}, update h_rs")),
        &h_rs,
        &hash_rs,
      )?;

      // (g) h_WS ← h_WS · Hash(gamma, alpha, a′, v′, wt)
      let hash_ws = randomized_hash_func(
        cs.namespace(|| format!("{i}, Hash(gamma, alpha, wa, wv, wt)")),
        &w_addr,
        &w_val,
        &w_ts,
        &gamma,
        &alpha,
      )?;

      // Add to count for WS
      let WS_counting_el =
        countable_hash(cs.namespace(|| format!("WS_counting_el_{i}")), &hash_ws)?;
      WS_size_count = add(
        cs.namespace(|| format!("{i}, WS_size_count ← WS_size_count + WS_counting_el")),
        &WS_size_count,
        &WS_counting_el,
      )?;
      h_ws = mul(
        cs.namespace(|| format!("{i}, update h_ws")),
        &h_ws,
        &hash_ws,
      )?;
    }

    // assert |RS| = |WS|
    enforce_equal(cs, || "assert |RS| = |WS|", &RS_size_count, &WS_size_count);
    enforce_equal(cs, || "|RS| == size", &RS_size_count, &size);

    Ok(vec![gamma, alpha, gts, h_rs, h_ws, size])
  }
}

impl OpsCircuit {
  /// Create a new instance of OpsCircuit for computing multiset hashes of
  /// (RS, WS)
  pub fn new(RS: Vec<(usize, u64, u64)>, WS: Vec<(usize, u64, u64)>) -> Self {
    OpsCircuit { RS, WS }
  }
}

impl Default for OpsCircuit {
  fn default() -> Self {
    OpsCircuit {
      RS: vec![(0, 0, 0); MEMORY_OPS_PER_STEP / 2],
      WS: vec![(0, 0, 0); MEMORY_OPS_PER_STEP / 2],
    }
  }
}
