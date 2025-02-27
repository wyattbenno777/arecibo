use crate::{
  frontend::{
    num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem, Split, SynthesisError,
  },
  gadgets::{conditionally_select2, nebula::allocated_avt, Num},
  hypernova::rs::StepCircuit,
  provider::{ipa_pc, Bn256EngineIPA},
  spartan::{lin_snark::LinearizedR1CSSNARK, snark::RelaxedR1CSSNARK},
  traits::{CurveCycleEquipped, Dual, Engine},
  NovaError,
};
use ff::PrimeField;
use itertools::Itertools;

use super::{
  api::{NebulaPublicParams, NebulaSNARK, RecursiveSNARKEngine, StepSize},
  product_circuits::convert_advice,
};

/// Maximum number of memory ops allowed per step of the heapify zkvm
const MEMORY_OPS_PER_STEP: usize = 7;

/// Used in the lt circuit to determine how many bits the range check should
/// check
const MAX_BITS: usize = 32;

// Basic type alias's for specifying proving curve-cycle
type E1 = Bn256EngineIPA;
type E2 = Dual<E1>;
type EE1 = ipa_pc::EvaluationEngine<E1>;
type EE2 = ipa_pc::EvaluationEngine<E2>;
type S1 = LinearizedR1CSSNARK<E1, EE1>;
type S2 = RelaxedR1CSSNARK<E2, EE2>;

#[test]
fn test_heapify() {
  let step_size = StepSize::new(1).set_memory_step_size(2);
  let pp: NebulaPublicParams<E1, S1, S2, MEMORY_OPS_PER_STEP> =
    NebulaSNARK::setup(&HeapifyCircuit::empty(), step_size);

  // Calculate testing memory (heap) size. We keep the test simple and ensure
  // memory size is a power of two
  let size_log = 5;

  // Get IS, FS, RS, & WS
  let (init_memory, mut final_memory) = heap_memory(size_log);
  let (read_ops, write_ops) = memory_ops_trace(&mut final_memory);

  // Custom zkVM engine for heapify
  let heapify_engine = HeapifyEngine {
    read_ops: read_ops.clone(),
    write_ops: write_ops.clone(),
    start_addr: ((init_memory.len() - 4) / 2),
  };

  let ms_err = "Multisets should be valid and input circuit should be sat";

  // Prove vm execution and memory consistency
  let (nebula_snark, U) = NebulaSNARK::prove(
    &pp,
    step_size,
    (init_memory, final_memory, read_ops, write_ops),
    heapify_engine,
  )
  .expect(ms_err);

  // Verify vm execution and memory consistency
  nebula_snark.verify(&pp, &U).expect(ms_err);

  // setup and compress
  let r1cs_err = "R1CS instance, witness pairs should be sat";
  let spartan = nebula_snark.compress(&pp).expect(r1cs_err);
  spartan.verify(&pp, &U).expect(r1cs_err);

  if false {
    let spartan_str = serde_json::to_string(&spartan).unwrap();
    println!("SNARK size {} KB", spartan_str.len() / 1024);
  }
}

struct HeapifyEngine {
  read_ops: Vec<Vec<(usize, u64, u64)>>,
  write_ops: Vec<Vec<(usize, u64, u64)>>,
  start_addr: usize,
}

impl<E> RecursiveSNARKEngine<E> for HeapifyEngine
where
  E: CurveCycleEquipped,
  <E as Engine>::Scalar: PartialOrd,
{
  type Circuit = HeapifyCircuit;
  fn circuits(&mut self) -> Result<Vec<Self::Circuit>, NovaError> {
    let mut circuits = Vec::new();
    for (RS, WS) in self.read_ops.iter().zip_eq(self.write_ops.iter()) {
      circuits.push(HeapifyCircuit {
        RS: RS.clone(),
        WS: WS.clone(),
      });
    }
    Ok(circuits)
  }

  fn z0(&self) -> Vec<<E>::Scalar> {
    vec![E::Scalar::from(self.start_addr as u64)]
  }
}

// returns initial_memory (IS) & final_memory (FS)
fn heap_memory(size_log: u32) -> (Vec<(usize, u64, u64)>, Vec<(usize, u64, u64)>) {
  let memory_size = 2usize.pow(size_log);
  let mut init_memory = (0..memory_size - 1)
    .map(|i| (i, (memory_size - 2 - i) as u64, 0_u64))
    .collect_vec();
  init_memory.push((memory_size - 1, 0, 0)); // attach 1 dummy element to assure table size is power of 2
  (init_memory.clone(), init_memory)
}

#[derive(Default, Clone, Debug)]
struct HeapifyCircuit {
  pub RS: Vec<(usize, u64, u64)>,
  pub WS: Vec<(usize, u64, u64)>,
}

impl<F> StepCircuit<F> for HeapifyCircuit
where
  F: PrimeField + PartialOrd,
{
  fn synthesize<CS>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
  where
    CS: ConstraintSystem<F>,
  {
    let parent_node_addr = z[0].clone();
    let left_child_addr = AllocatedNum::alloc(cs.namespace(|| "left_child_addr"), || {
      parent_node_addr
        .get_value()
        .map(|i| i.mul(F::from(2)) + F::ONE)
        .ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
      || "(2*addr + 1) * 1 = left_child_addr",
      |lc| lc + (F::from(2), parent_node_addr.get_variable()) + CS::one(),
      |lc| lc + CS::one(),
      |lc| lc + left_child_addr.get_variable(),
    );
    let right_child_addr = AllocatedNum::alloc(cs.namespace(|| "right_child_addr"), || {
      left_child_addr
        .get_value()
        .map(|i| i + F::ONE)
        .ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
      || "(left_child_addr + 1) * 1 = right_child_addr",
      |lc| lc + left_child_addr.get_variable() + CS::one(),
      |lc| lc + CS::one(),
      |lc| lc + right_child_addr.get_variable(),
    );
    let parent = self.read(cs.namespace(|| "get parent value"), &parent_node_addr, 0)?;
    let left_child = self.read(cs.namespace(|| "get left child value"), &left_child_addr, 1)?;
    let right_child = self.read(
      cs.namespace(|| "get right child value"),
      &right_child_addr,
      2,
    )?;
    let is_left_child_smaller = less_than(
      cs.namespace(|| "left_child < parent"),
      &left_child,
      &parent,
      MAX_BITS,
    )?;
    let new_parent_left = conditionally_select2(
      cs.namespace(|| "new_left_pair_parent"),
      &left_child,
      &parent,
      &is_left_child_smaller,
    )?;
    let new_left_child = conditionally_select2(
      cs.namespace(|| "new_left_pair_child"),
      &parent,
      &left_child,
      &is_left_child_smaller,
    )?;
    self.write(
      cs.namespace(|| "write new left parent"),
      &parent_node_addr,
      &new_parent_left,
      3,
    )?;
    self.write(
      cs.namespace(|| "write new left child"),
      &left_child_addr,
      &new_left_child,
      4,
    )?;
    let is_right_child_smaller = less_than(
      cs.namespace(|| "right_child < parent"),
      &right_child,
      &new_parent_left,
      MAX_BITS,
    )?;
    let new_parent_right = conditionally_select2(
      cs.namespace(|| "new_right_pair_parent"),
      &right_child,
      &new_parent_left,
      &is_right_child_smaller,
    )?;
    let new_right_child = conditionally_select2(
      cs.namespace(|| "new_right_pair_child"),
      &new_parent_left,
      &right_child,
      &is_right_child_smaller,
    )?;
    self.write(
      cs.namespace(|| "write new right parent"),
      &parent_node_addr,
      &new_parent_right,
      5,
    )?;
    self.write(
      cs.namespace(|| "write new right child"),
      &right_child_addr,
      &new_right_child,
      6,
    )?;
    let next_addr = AllocatedNum::alloc(cs.namespace(|| "next_addr"), || {
      parent_node_addr
        .get_value()
        .map(|addr| addr - F::ONE)
        .ok_or(SynthesisError::AssignmentMissing)
    })?;
    cs.enforce(
      || "(next_addr + 1) * 1 = addr",
      |lc| lc + next_addr.get_variable() + CS::one(),
      |lc| lc + CS::one(),
      |lc| lc + parent_node_addr.get_variable(),
    );
    Ok(vec![next_addr])
  }

  fn arity(&self) -> usize {
    1
  }

  fn advice(&self) -> (Vec<F>, Vec<F>) {
    (convert_advice(&self.RS, &self.WS), vec![])
  }
}

impl HeapifyCircuit {
  pub fn empty() -> Self {
    HeapifyCircuit {
      RS: vec![(0, 0, 0); MEMORY_OPS_PER_STEP],
      WS: vec![(0, 0, 0); MEMORY_OPS_PER_STEP],
    }
  }

  /// Pefrom a read to zkVM read-write memory.  for a read operation, the
  /// advice is (a, v, rt) and (a, v, wt); F checks that the address a in
  /// the advice matches the address it requested and then uses the
  /// provided value v (e.g., in the rest of its computation).
  fn read<CS, F>(
    &self,
    mut cs: CS,
    addr: &AllocatedNum<F>,
    advice_idx: usize,
  ) -> Result<AllocatedNum<F>, SynthesisError>
  where
    F: PrimeField,
    CS: ConstraintSystem<F>,
  {
    let (advice_addr, advice_val, _) = allocated_avt(
      cs.namespace(|| "allocate advice"),
      self.RS[advice_idx],
      Split::ZERO,
    )?;

    // allocate the WS aswell, so it can be incrementaly committed
    let _ = allocated_avt(
      cs.namespace(|| "allocate WS advice"),
      self.WS[advice_idx],
      Split::ZERO,
    )?;

    // F checks that the address a in the advice matches the address it
    // requested
    cs.enforce(
      || "addr == advice_addr",
      |lc| lc + addr.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + advice_addr.get_variable(),
    );
    Ok(advice_val)
  }

  /// Perform a write to zkVM read-write memory.  For a write operation, the
  /// advice is (a, v, rt) and (a, v′, wt); F checks that the address a
  /// and the value v′ match the address and value it wishes to write.
  /// Otherwise, F ignores the remaining components in the provided advice.
  fn write<CS, F>(
    &self,
    mut cs: CS,
    addr: &AllocatedNum<F>,
    val: &AllocatedNum<F>,
    advice_idx: usize,
  ) -> Result<(), SynthesisError>
  where
    F: PrimeField,
    CS: ConstraintSystem<F>,
  {
    // allocate the RS aswell, so it can be incrementaly committed
    let _ = allocated_avt(
      cs.namespace(|| "allocate WS advice"),
      self.RS[advice_idx],
      Split::ZERO,
    )?;

    // Allocate the advice
    let (advice_addr, advice_val, _) = allocated_avt(
      cs.namespace(|| "allocate advice"),
      self.WS[advice_idx],
      Split::ZERO,
    )?;

    // F checks that the address a  match the address it wishes to write to.
    cs.enforce(
      || "addr == advice_addr",
      |lc| lc + addr.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + advice_addr.get_variable(),
    );

    // F checks that the value v′ match value it wishes to write.
    cs.enforce(
      || "val == advice_val",
      |lc| lc + val.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + advice_val.get_variable(),
    );
    Ok(())
  }
}

fn memory_ops_trace(
  memory: &mut [(usize, u64, u64)],
) -> (Vec<Vec<(usize, u64, u64)>>, Vec<Vec<(usize, u64, u64)>>) {
  let mut read_ops = Vec::new();
  let mut write_ops = Vec::new();
  let initial_index = (memory.len() - 4) / 2;
  let num_steps = initial_index + 1;
  let mut global_ts = 0;
  for i in 0..num_steps {
    let mut RS = Vec::new();
    let mut WS = Vec::new();
    let parent_addr = initial_index - i;
    read_op(parent_addr, &mut global_ts, memory, &mut RS, &mut WS);
    let left_child_addr = 2 * parent_addr + 1;
    read_op(left_child_addr, &mut global_ts, memory, &mut RS, &mut WS);
    let right_child_addr = 2 * parent_addr + 2;
    read_op(right_child_addr, &mut global_ts, memory, &mut RS, &mut WS);

    // Swap parent with left
    let (new_parent_left, new_left_child) = if memory[left_child_addr].1 < memory[parent_addr].1 {
      (memory[left_child_addr].1, memory[parent_addr].1)
    } else {
      (memory[parent_addr].1, memory[left_child_addr].1)
    };
    write_op(
      parent_addr,
      new_parent_left,
      &mut global_ts,
      memory,
      &mut RS,
      &mut WS,
    );
    write_op(
      left_child_addr,
      new_left_child,
      &mut global_ts,
      memory,
      &mut RS,
      &mut WS,
    );

    // Swap parent with right
    let (new_parent_right, new_right_child) = if memory[right_child_addr].1 < new_parent_left {
      (memory[right_child_addr].1, new_parent_left)
    } else {
      (new_parent_left, memory[right_child_addr].1)
    };
    write_op(
      parent_addr,
      new_parent_right,
      &mut global_ts,
      memory,
      &mut RS,
      &mut WS,
    );
    write_op(
      right_child_addr,
      new_right_child,
      &mut global_ts,
      memory,
      &mut RS,
      &mut WS,
    );
    read_ops.push(RS);
    write_ops.push(WS);
  }
  (read_ops, write_ops)
}

/// a < b ? 1 : 0
pub fn less_than<F: PrimeField + PartialOrd, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  n_bits: usize,
) -> Result<AllocatedNum<F>, SynthesisError> {
  assert!(n_bits < 64, "not support n_bits {n_bits} >= 64");
  let range = F::from(1u64 << n_bits);
  // diff = (lhs - rhs) + (if lt { range } else { 0 });
  let diff = Num::alloc(cs.namespace(|| "diff"), || {
    a.get_value()
      .zip(b.get_value())
      .map(|(a, b)| {
        let lt = a < b;
        (a - b) + (if lt { range } else { F::ZERO })
      })
      .ok_or(SynthesisError::AssignmentMissing)
  })?;
  diff.fits_in_bits(cs.namespace(|| "diff fit in bits"), n_bits)?;
  let diff = diff.as_allocated_num(cs.namespace(|| "diff_alloc_num"))?;
  let lt = AllocatedNum::alloc(cs.namespace(|| "lt"), || {
    a.get_value()
      .zip(b.get_value())
      .map(|(a, b)| F::from(u64::from(a < b)))
      .ok_or(SynthesisError::AssignmentMissing)
  })?;
  cs.enforce(
    || "lt is bit",
    |lc| lc + lt.get_variable(),
    |lc| lc + CS::one() - lt.get_variable(),
    |lc| lc,
  );
  cs.enforce(
    || "lt ⋅ range == diff - lhs + rhs",
    |lc| lc + (range, lt.get_variable()),
    |lc| lc + CS::one(),
    |lc| lc + diff.get_variable() - a.get_variable() + b.get_variable(),
  );
  Ok(lt)
}

/// Read operation between an untrusted memory and a checker
fn read_op(
  addr: usize,
  global_ts: &mut u64,
  FS: &mut [(usize, u64, u64)],
  RS: &mut Vec<(usize, u64, u64)>,
  WS: &mut Vec<(usize, u64, u64)>,
) {
  // 1. ts ← ts + 1
  *global_ts += 1;

  // untrusted memory responds with a value-timestamp pair (v, t)
  let (_, r_val, r_ts) = FS[addr];

  // 2. assert t < ts
  debug_assert!(r_ts < *global_ts);

  // 3. RS ← RS ∪ {(a,v,t)};
  RS.push((addr, r_val, r_ts));

  // 4. store (v, ts) at address a in the untrusted memory; and
  FS[addr] = (addr, r_val, *global_ts);

  // 5. WS ← WS ∪ {(a,v,ts)}.
  WS.push((addr, r_val, *global_ts));
}

/// Write operation between an untrusted memory and a checker
fn write_op(
  addr: usize,
  val: u64,
  global_ts: &mut u64,
  FS: &mut [(usize, u64, u64)],
  RS: &mut Vec<(usize, u64, u64)>,
  WS: &mut Vec<(usize, u64, u64)>,
) {
  // 1. ts ← ts + 1
  *global_ts += 1;

  // untrusted memory responds with a value-timestamp pair (v, t)
  let (_, r_val, r_ts) = FS[addr];

  // 2. assert t < ts
  debug_assert!(r_ts < *global_ts);

  // 3. RS ← RS ∪ {(a,v,t)};
  RS.push((addr, r_val, r_ts));

  // 4. store (v', ts) at address a in the untrusted memory; and
  FS[addr] = (addr, val, *global_ts);

  // 5. WS ← WS ∪ {(a,v',ts)}.
  WS.push((addr, val, *global_ts));
}

#[allow(dead_code)]
fn debug_step<E>(circuit: &impl StepCircuit<E::Scalar>, z_i: &[E::Scalar]) -> Result<(), NovaError>
where
  E: CurveCycleEquipped,
{
  let mut cs = TestConstraintSystem::<E::Scalar>::new();
  let z_i: Vec<AllocatedNum<E::Scalar>> = z_i
    .iter()
    .enumerate()
    .map(|(i, scalar)| AllocatedNum::alloc(cs.namespace(|| format!("z_{}", i)), || Ok(*scalar)))
    .collect::<Result<Vec<_>, _>>()?;
  circuit
    .synthesize(&mut cs, &z_i)
    .map_err(|_| NovaError::from(SynthesisError::AssignmentMissing))?;
  let is_sat = cs.is_satisfied();
  if !is_sat {
    assert!(is_sat);
  }
  Ok(())
}
