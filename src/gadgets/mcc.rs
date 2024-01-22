//! This module implements lookup gadget for applications built with Nova.
use std::collections::btree_map::Iter;
use std::collections::btree_map::Values;
use std::collections::BTreeMap;

use crate::Engine;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use std::cmp::Ord;

use crate::constants::NUM_CHALLENGE_BITS;
use crate::traits::commitment::CommitmentEngineTrait;
use crate::traits::AbsorbInROTrait;
use crate::traits::ROCircuitTrait;
use crate::traits::ROConstants;
use crate::traits::ROConstantsCircuit;
use crate::traits::ROTrait;
use ff::{Field, PrimeField};

use super::utils::scalar_as_base;
use super::utils::{alloc_one, conditionally_select2, le_bits_to_num, less_than, add_allocated_num};
use crate::spartan::math::Math;

/*
  This is a modification of the lookup gadget discussed here.
  https://github.com/lurk-lab/arecibo/pull/48 by @hero78119
  I instead focus on only using it as a memory consistancy check for a Nova based zkVM.
*/

/* 
  Memory trace vec:
  A multiplicity is how many times a specific item has been accessed.
  They are deterministic: i.e. it is always multiplicity++
  All reads are followed by a write that increases the multiplicity value for that cell.
  At the end of the procedure one final Read will occur, but this time not followed by a write.
  This makes Read.len == Write.len and we are ready for permutation check.
*/

#[derive(Clone, Debug)]
pub enum MemoryTraceEnum<T> {
  Read(T, T, T), // addr, read_value, multiplicity
  Write(T, T, T, T), // addr, read_value, new_value, multiplicity
}

/// R1CS for reads and writes into the table.
#[derive(Clone)]
pub struct MemoryR1CS<E: Engine> {
  expected_memory_trace: Vec<MemoryTraceEnum<E::Scalar>>,
  memory_trace_allocated_num: Vec<MemoryTraceEnum<AllocatedNum<E::Scalar>>>,
  memory_size_log2: usize,
  cursor: usize,
}

impl<E: Engine> MemoryR1CS<E> {
  /// read value from table
  pub fn read<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &mut self,
    mut cs: CS,
    addr: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    <E as Engine>::Scalar: Ord + PartialEq + Eq,
  {
    assert!(
      self.cursor < self.expected_memory_trace.len(),
      "cursor {} out of range with expected length {}",
      self.cursor,
      self.expected_memory_trace.len()
    );
    let MemoryTraceEnum::Read(expected_addr, expected_read_value, expected_read_counter) =
      self.expected_memory_trace[self.cursor]
    else {
      Err(SynthesisError::AssignmentMissing)?
    };

    if let Some(key) = addr.get_value() {
      assert!(
        key == expected_addr,
        "read address {:?} mismatch with expected {:?}",
        key,
        expected_addr
      );
    }
    let read_value =
      AllocatedNum::alloc(cs.namespace(|| "read_value"), || Ok(expected_read_value))?;
    let read_counter = AllocatedNum::alloc(cs.namespace(|| "read_counter"), || {
      Ok(expected_read_counter)
    })?;
    self
      .memory_trace_allocated_num
      .push(MemoryTraceEnum::Read::<AllocatedNum<E::Scalar>>(
        addr.clone(),
        read_value.clone(),
        read_counter,
      )); // append read trace

    self.cursor += 1;
    Ok(read_value)
  }

  /// write value to lookup table
  pub fn write<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &mut self,
    mut cs: CS,
    addr: &AllocatedNum<E::Scalar>,
    value: &AllocatedNum<E::Scalar>,
  ) -> Result<(), SynthesisError>
  where
    <E as Engine>::Scalar: Ord,
  {
    assert!(
      self.cursor < self.expected_memory_trace.len(),
      "cursor {} out of range with expected length {}",
      self.cursor,
      self.expected_memory_trace.len()
    );
    let MemoryTraceEnum::Write(
      expected_addr,
      expected_read_value,
      expected_read_counter,
      expected_write_value,
    ) = self.expected_memory_trace[self.cursor]
    else {
      Err(SynthesisError::AssignmentMissing)?
    };

    if let Some((addr, value)) = addr.get_value().zip(value.get_value()) {
      assert!(
        addr == expected_addr,
        "write address {:?} mismatch with expected {:?}",
        addr,
        expected_addr
      );
      assert!(
        value == expected_write_value,
        "write value {:?} mismatch with expected {:?}",
        value,
        expected_write_value
      );
    }

    let expected_read_value =
      AllocatedNum::alloc(cs.namespace(|| "read_value"), || Ok(expected_read_value))?;
    let expected_read_counter = AllocatedNum::alloc(cs.namespace(|| "read_counter"), || {
      Ok(expected_read_counter)
    })?;
    self.memory_trace_allocated_num.push(MemoryTraceEnum::Write(
      addr.clone(),
      expected_read_value,
      value.clone(),
      expected_read_counter,
    )); // append write trace
    self.cursor += 1;
    Ok(())
  }

  /// commit memory_trace to lookup
  #[allow(clippy::too_many_arguments)]
  pub fn commit<E2: Engine, CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &mut self,
    mut cs: CS,
    ro_const: ROConstantsCircuit<E2>,
    prev_intermediate_gamma: &AllocatedNum<E::Scalar>,
    challenges: &(AllocatedNum<E::Scalar>, AllocatedNum<E::Scalar>),
    prev_R: &AllocatedNum<E::Scalar>,
    prev_W: &AllocatedNum<E::Scalar>,
    prev_multiplicity: &AllocatedNum<E::Scalar>,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,
      AllocatedNum<E::Scalar>,
      AllocatedNum<E::Scalar>,
      AllocatedNum<E::Scalar>,
    ),
    SynthesisError,
  >
  where
    <E as Engine>::Scalar: Ord,
    E: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E as Engine>::Scalar>,
  {
    let mut ro = E2::ROCircuit::new(
      ro_const,
      1 + 3 * self.expected_memory_trace.len(), // prev_challenge + [(address, value, counter)]
    );
    ro.absorb(prev_intermediate_gamma);
    let memory_trace_allocated_num = &self.memory_trace_allocated_num;
    let (next_R, next_W, next_multiplicity) = memory_trace_allocated_num.iter().enumerate().try_fold(
      (prev_R.clone(), prev_W.clone(), prev_multiplicity.clone()),
      |(prev_R, prev_W, prev_multiplicity), (i, rwtrace)| match rwtrace {
        MemoryTraceEnum::Read(addr, read_value, expected_read_counter) => {
          let (next_R, next_W, next_multiplicity) = self.rw_operation_circuit(
            cs.namespace(|| format!("{}th read ", i)),
            addr,
            challenges,
            read_value,
            read_value,
            &prev_R,
            &prev_W,
            expected_read_counter,
            &prev_multiplicity,
          )?;
          ro.absorb(addr);
          ro.absorb(read_value);
          ro.absorb(expected_read_counter);
          Ok::<
            (
              AllocatedNum<E::Scalar>,
              AllocatedNum<E::Scalar>,
              AllocatedNum<E::Scalar>,
            ),
            SynthesisError,
          >((next_R, next_W, next_multiplicity))
        }
        MemoryTraceEnum::Write(addr, read_value, write_value, read_counter) => {
          let (next_R, next_W, next_multiplicity) = self.rw_operation_circuit(
            cs.namespace(|| format!("{}th write ", i)),
            addr,
            challenges,
            read_value,
            write_value,
            &prev_R,
            &prev_W,
            read_counter,
            &prev_multiplicity,
          )?;
          ro.absorb(addr);
          ro.absorb(read_value);
          ro.absorb(read_counter);
          Ok::<
            (
              AllocatedNum<E::Scalar>,
              AllocatedNum<E::Scalar>,
              AllocatedNum<E::Scalar>,
            ),
            SynthesisError,
          >((next_R, next_W, next_multiplicity))
        }
      },
    )?;
    let hash_bits = ro.squeeze(cs.namespace(|| "challenge"), NUM_CHALLENGE_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), &hash_bits)?;
    Ok((next_R, next_W, next_multiplicity, hash))
  }

  #[allow(clippy::too_many_arguments)]
  fn rw_operation_circuit<F: PrimeField, CS: ConstraintSystem<F>>(
    &self,
    mut cs: CS,
    addr: &AllocatedNum<F>,
    challenges: &(AllocatedNum<F>, AllocatedNum<F>),
    read_value: &AllocatedNum<F>,
    write_value: &AllocatedNum<F>,
    prev_R: &AllocatedNum<F>,
    prev_W: &AllocatedNum<F>,
    read_counter: &AllocatedNum<F>,
    prev_multiplicity: &AllocatedNum<F>,
  ) -> Result<(AllocatedNum<F>, AllocatedNum<F>, AllocatedNum<F>), SynthesisError>
  where
    F: Ord,
  {
    let (alpha, gamma) = challenges;
    // update R
    let gamma_square = gamma.mul(cs.namespace(|| "gamme^2"), gamma)?;
    // read_value_term = gamma * value
    let read_value_term = gamma.mul(cs.namespace(|| "read_value_term"), read_value)?;
    // counter_term = gamma^2 * counter
    let read_counter_term = gamma_square.mul(cs.namespace(|| "read_counter_term"), read_counter)?;
    // new_R = R + 1 / (alpha + (addr + gamma * value + gamma^2 * counter))
    let new_R = AllocatedNum::alloc(cs.namespace(|| "new_R"), || {
      prev_R
        .get_value()
        .zip(alpha.get_value())
        .zip(addr.get_value())
        .zip(read_value_term.get_value())
        .zip(read_counter_term.get_value())
        .map(|((((R, alpha), addr), value_term), counter_term)| {
          R + (alpha + (addr + value_term + counter_term))
            .invert()
            .expect("invert failed due to read term is 0") // negilible probability for invert failed
        })
        .ok_or(SynthesisError::AssignmentMissing)
    })?;
    let mut r_blc = LinearCombination::<F>::zero();
    r_blc = r_blc
      + alpha.get_variable()
      + addr.get_variable()
      + read_value_term.get_variable()
      + read_counter_term.get_variable();
    cs.enforce(
      || "R update",
      |lc| lc + new_R.get_variable() - prev_R.get_variable(),
      |_| r_blc,
      |lc| lc + CS::one(),
    );

    let alloc_num_one = alloc_one(cs.namespace(|| "one"));
    // max{read_counter, multiplicity} logic on read-write lookup
    // read_counter on read-only
    // - max{read_counter, multiplicity} if read-write table
    // - read_counter if read-only table
    // +1 will be hadle later
    let (write_counter, write_counter_term) = {
      // write_counter = read_counter < prev_multiplicity ? prev_multiplicity: read_counter
      // TODO optimise with `max` table lookup to save more constraints
      let lt = less_than(
        cs.namespace(|| "read_counter < a"),
        read_counter,
        prev_multiplicity,
        self.memory_size_log2,
      )?;
      let write_counter = conditionally_select2(
        cs.namespace(|| {
          "write_counter = read_counter < prev_multiplicity ? prev_multiplicity: read_counter"
        }),
        prev_multiplicity,
        read_counter,
        &lt,
      )?;
      let write_counter_term =
        gamma_square.mul(cs.namespace(|| "write_counter_term"), &write_counter)?;
      (write_counter, write_counter_term)
    };

    // update W
    // write_value_term = gamma * value
    let write_value_term = gamma.mul(cs.namespace(|| "write_value_term"), write_value)?;
    let new_W = AllocatedNum::alloc(cs.namespace(|| "new_W"), || {
      prev_W
        .get_value()
        .zip(alpha.get_value())
        .zip(addr.get_value())
        .zip(write_value_term.get_value())
        .zip(write_counter_term.get_value())
        .zip(gamma_square.get_value())
        .map(
          |(((((W, alpha), addr), value_term), write_counter_term), gamma_square)| {
            W + (alpha + (addr + value_term + write_counter_term + gamma_square))
              .invert()
              .expect("invert failed due to write term is 0") // negilible probability for invert failed
          },
        )
        .ok_or(SynthesisError::AssignmentMissing)
    })?;
    // new_W = W + 1 / (alpha - (addr + gamma * value + gamma^2 * counter))
    let mut w_blc = LinearCombination::<F>::zero();
    w_blc = w_blc
      + alpha.get_variable()
      + addr.get_variable()
      + write_value_term.get_variable()
      + write_counter_term.get_variable()
      + gamma_square.get_variable();
    cs.enforce(
      || "W update",
      |lc| lc + new_W.get_variable() - prev_W.get_variable(),
      |_| w_blc,
      |lc| lc + CS::one(),
    );

    let new_multiplicity = add_allocated_num(
      cs.namespace(|| "new_multiplicity"),
      &write_counter,
      &alloc_num_one,
    )?;

    // update accu
    Ok((new_R, new_W, new_multiplicity))
  }
}

/* 
  Memory Consistency Object:
  memory_trace: Vec[Enum(Read or Write)]
  Table should be init with a write (addr, val, 0).
*/
#[derive(Clone, Debug)]
pub struct MemoryConsistencyObject<E: Engine> {
  memory_trace: Vec<MemoryTraceEnum<E::Scalar>>,
  table_aux: BTreeMap<E::Scalar, (E::Scalar, E::Scalar)>, // addr, (value, counter)
  multiplicity: E::Scalar,
  pub(crate) memory_size_log2: usize, // max cap for multiplicity operation in bits
}

impl<E: Engine> MemoryConsistencyObject<E> {
  /// new lookup table
  pub fn new(
    memory_size: usize,
  ) -> MemoryConsistencyObject<E>
  where
    E::Scalar: Ord,
  {
    let memory_size_log2 = memory_size.log_2();
    let mut table_aux = BTreeMap::new();
    let mut memory_trace = vec![];

    for i in 0..memory_size {
      let address = E::Scalar::from(i as u64);
      table_aux.insert(
        address, 
        (E::Scalar::ZERO, E::Scalar::ZERO)
      );

      memory_trace.push(MemoryTraceEnum::Write(
        address,
        E::Scalar::ZERO, //value
        E::Scalar::ZERO, //multiplicity
        E::Scalar::ZERO, //new value
      )); 
    }

    Self {
      memory_trace,
      table_aux,
      multiplicity: E::Scalar::ZERO,
      memory_size_log2,
    }
  }

  pub fn table_size(&self) -> usize {
    self.table_aux.len()
  }

  pub fn values(&self) -> Values<'_, E::Scalar, (E::Scalar, E::Scalar)> {
    self.table_aux.values()
  }

  /*
   All reads are followed by a write that increments the multiplicity.
  */
  fn rw_operation(&mut self, addr: E::Scalar, external_value: Option<E::Scalar>) -> (E::Scalar, E::Scalar)
  where
    E::Scalar: Ord,
  {

    let (read_value, multiplicity) = self
      .table_aux
      .get(&addr)
      .cloned()
      .unwrap();

    self
    .memory_trace
    .push(MemoryTraceEnum::Read(addr, read_value, multiplicity));

    let (write_value, write_counter) = (
        external_value.unwrap_or(read_value), // Write the new value or keep the read value.
        multiplicity + E::Scalar::ONE,
    );

    // Follow all reads by a write until last step.
    self
    .memory_trace
    .push(MemoryTraceEnum::Write(addr, read_value, write_value, write_counter));
    
    self.table_aux.insert(addr, (write_value, write_counter)); // note. This updates when addr exists.
    self.multiplicity = write_counter;
    (write_value, multiplicity)
  }

  /// commit memory_trace to lookup
  pub fn snapshot<E2: Engine>(
    &mut self,
    ro_consts: ROConstants<E2>,
    prev_intermediate_gamma: E::Scalar,
  ) -> (E::Scalar, MemoryR1CS<E>)
  where
    <E as Engine>::Scalar: Ord,
    E: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E as Engine>::Scalar>,
  {
    let mut hasher: <E2 as Engine>::RO =
      <E2 as Engine>::RO::new(ro_consts, 1 + self.memory_trace.len() * 3);
    hasher.absorb(prev_intermediate_gamma);

    let memory_trace = self.memory_trace.drain(..).collect::<Vec<_>>();

    // This absorbs addr, read_value, multiplicity.
    let rw_processed = memory_trace
      .into_iter()
      .map(|rwtrace| {
        let (memory_trace_with_counter, addr, end_value, multiplicity) = match rwtrace {
          MemoryTraceEnum::Read(addr, expected_read_value, multiplicity) => {
            (
              MemoryTraceEnum::Read(addr, expected_read_value, multiplicity),
              addr,
              expected_read_value,
              multiplicity,
            )
          }
          MemoryTraceEnum::Write(addr, read_value, write_value, multiplicity) => {
            (
              MemoryTraceEnum::Write(addr, read_value, write_value, multiplicity),
              addr,
              write_value,
              multiplicity,
            )
          }
        };
        hasher.absorb(addr);
        hasher.absorb(end_value);
        hasher.absorb(multiplicity);

        memory_trace_with_counter
      })
      .collect();
    let hash_bits = hasher.squeeze(NUM_CHALLENGE_BITS);
    let next_intermediate_gamma = scalar_as_base::<E2>(hash_bits);
    (
      next_intermediate_gamma,
      MemoryR1CS {
        expected_memory_trace: rw_processed,
        memory_trace_allocated_num: vec![],
        cursor: 0,
        memory_size_log2: self.memory_size_log2,
      },
    )
  }

  /// Get permutation fingerprint alpha and gamma
  pub fn get_challenge<E2: Engine>(
    &self,
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
    intermediate_gamma: E::Scalar,
  ) -> (E::Scalar, E::Scalar)
  where
    E: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E as Engine>::Scalar>,
  {
    let ro_consts =
      <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants::default();

    let (final_values, final_counters): (Vec<_>, Vec<_>) =
      self.table_aux.values().copied().unzip();

    // final_value and final_counter
    let (comm_final_value, comm_final_counter) = rayon::join(
      || E::CE::commit(ck, &final_values),
      || E::CE::commit(ck, &final_counters),
    );

    // gamma
    let mut hasher = <E as Engine>::RO::new(ro_consts.clone(), 7);
    let intermediate_gamma: E2::Scalar = scalar_as_base::<E>(intermediate_gamma);
    hasher.absorb(intermediate_gamma);
    comm_final_value.absorb_in_ro(&mut hasher);
    comm_final_counter.absorb_in_ro(&mut hasher);
    let gamma = hasher.squeeze(NUM_CHALLENGE_BITS);

    // alpha
    let mut hasher = <E as Engine>::RO::new(ro_consts, 1);
    hasher.absorb(scalar_as_base::<E>(gamma));
    let alpha = hasher.squeeze(NUM_CHALLENGE_BITS);
    (alpha, gamma)
  }
}

impl<'a, E: Engine> IntoIterator for &'a MemoryConsistencyObject<E> {
  type Item = (&'a E::Scalar, &'a (E::Scalar, E::Scalar));
  type IntoIter = Iter<'a, E::Scalar, (E::Scalar, E::Scalar)>;

  fn into_iter(self) -> Self::IntoIter {
    self.table_aux.iter()
  }
}

#[cfg(test)]
mod test {
  use crate::{
    // bellpepper::test_shape_cs::TestShapeCS,
    constants::NUM_CHALLENGE_BITS,
    gadgets::{
      utils::{alloc_one, alloc_zero, scalar_as_base},
    },
    provider::{poseidon::PoseidonConstantsCircuit, PallasEngine, VestaEngine},
    traits::{Engine, ROConstantsCircuit},
  };
  use ff::Field;

  use super::MemoryConsistencyObject;
  use crate::traits::ROTrait;
  use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};

  #[test]
  fn test_lookup_simulation() {
    type E1 = PallasEngine;
    type E2 = VestaEngine;

    // let mut cs: TestShapeCS<E1> = TestShapeCS::new();
    let mut mcc =
      MemoryConsistencyObject::<E1>::new(3);
      
    let (read_value, _multiplicity) = mcc.rw_operation(<E1 as Engine>::Scalar::ZERO, None);
    assert_eq!(read_value, <E1 as Engine>::Scalar::ZERO);

    let (read_value, _multiplicity) = mcc.rw_operation(<E1 as Engine>::Scalar::ONE, None);
    assert_eq!(read_value, <E1 as Engine>::Scalar::ZERO);

    mcc.rw_operation(
      <E1 as Engine>::Scalar::ZERO,
      Some(<E1 as Engine>::Scalar::from(111)),
    );

    let (read_value, _multiplicity) = mcc.rw_operation(<E1 as Engine>::Scalar::ZERO, None);
    assert_eq!(read_value, <E1 as Engine>::Scalar::from(111));

    let ro_consts: ROConstantsCircuit<E2> = PoseidonConstantsCircuit::default();
    let prev_intermediate_gamma = <E1 as Engine>::Scalar::ONE;

    let (next_intermediate_gamma, _) =
      mcc.snapshot::<E2>(ro_consts.clone(), prev_intermediate_gamma);

    let mut hasher = <E2 as Engine>::RO::new(ro_consts, 1 + 11 * 3);
    hasher.absorb(prev_intermediate_gamma);

    // Init table.
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // addr
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // value
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // multiplicity

    hasher.absorb(<E1 as Engine>::Scalar::ONE); // addr
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // value
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // multiplicity
    
    hasher.absorb(<E1 as Engine>::Scalar::from(2)); // addr
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // value
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // multiplicity

    // first rw in 0.
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // addr
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // value
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // multiplicity
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // addr
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // value
    hasher.absorb(<E1 as Engine>::Scalar::ONE); // multiplicity

    // first rw in 1
    hasher.absorb(<E1 as Engine>::Scalar::ONE); // addr
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // value
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // multiplicity
    hasher.absorb(<E1 as Engine>::Scalar::ONE); // addr
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // value
    hasher.absorb(<E1 as Engine>::Scalar::ONE); // multiplicity

    // second rw in 0
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // addr
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // value
    hasher.absorb(<E1 as Engine>::Scalar::ONE); // multiplicity
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // addr
    hasher.absorb(<E1 as Engine>::Scalar::from(111)); // value
    hasher.absorb(<E1 as Engine>::Scalar::from(2)); // multiplicity

    // third rw in 0
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // addr
    hasher.absorb(<E1 as Engine>::Scalar::from(111)); // value
    hasher.absorb(<E1 as Engine>::Scalar::from(2)); // multiplicity
    hasher.absorb(<E1 as Engine>::Scalar::ZERO); // addr
    hasher.absorb(<E1 as Engine>::Scalar::from(111)); // value
    hasher.absorb(<E1 as Engine>::Scalar::from(3)); // multiplicity
    
    let res = hasher.squeeze(NUM_CHALLENGE_BITS);
    assert_eq!(scalar_as_base::<E2>(res), next_intermediate_gamma);
  }


  // this test will not work yet.
  #[test]
  fn test_write_read_on_rwlookup() {
    type E1 = PallasEngine;
    type E2 = VestaEngine;

    let ro_consts: ROConstantsCircuit<E2> = PoseidonConstantsCircuit::default();

    let mut cs = TestConstraintSystem::<<E1 as Engine>::Scalar>::new();
    let mut mcc = MemoryConsistencyObject::<E1>::new(124);

    let challenges = (
      AllocatedNum::alloc(cs.namespace(|| "alpha"), || {
        Ok(<E1 as Engine>::Scalar::from(5))
      })
      .unwrap(),
      AllocatedNum::alloc(cs.namespace(|| "gamma"), || {
        Ok(<E1 as Engine>::Scalar::from(7))
      })
      .unwrap(),
    );
    let (alpha, gamma) = &challenges;
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let one = alloc_one(cs.namespace(|| "one"));
    let prev_intermediate_gamma = &one;
    let prev_multiplicity = &zero;
    let addr = zero.clone();
    let write_value_1 = AllocatedNum::alloc(cs.namespace(|| "write value 1"), || {
      Ok(<E1 as Engine>::Scalar::from(101))
    })
    .unwrap();

    mcc.rw_operation(
      addr.get_value().unwrap(),
      Some(write_value_1.get_value().unwrap()),
    );

    let (read_value, _multiplicity) = mcc.rw_operation(addr.get_value().unwrap(), None);
    assert_eq!(read_value, <E1 as Engine>::Scalar::from(101));
    let (_, mut lookup_trace) = mcc.snapshot::<E2>(
      ro_consts.clone(),
      prev_intermediate_gamma.get_value().unwrap(),
    );
    lookup_trace
      .write(cs.namespace(|| "write_value 1"), &addr, &write_value_1)
      .unwrap();
    let read_value = lookup_trace
      .read(cs.namespace(|| "read_value 1"), &addr)
      .unwrap();
    assert_eq!(
      read_value.get_value(),
      Some(<E1 as Engine>::Scalar::from(101))
    );

    let (prev_W, prev_R) = (&one, &one);
    let (next_R, next_W, next_multiplicity, next_intermediate_gamma) = lookup_trace
      .commit::<E2, _>(
        cs.namespace(|| "commit"),
        ro_consts.clone(),
        prev_intermediate_gamma,
        &challenges,
        prev_W,
        prev_R,
        prev_multiplicity,
      )
      .unwrap();
    assert_eq!(
      next_multiplicity.get_value(),
      Some(<E1 as Engine>::Scalar::from(2))
    );
    // next_R check
    assert_eq!(
      next_R.get_value(),
      prev_R
        .get_value()
        .zip(alpha.get_value())
        .zip(gamma.get_value())
        .zip(addr.get_value())
        .zip(read_value.get_value())
        .map(|((((prev_R, alpha), gamma), addr), read_value)| prev_R
          + (alpha
            + (addr
              + gamma * <E1 as Engine>::Scalar::ZERO
              + gamma * gamma * <E1 as Engine>::Scalar::ZERO))
            .invert()
            .unwrap()
          + (alpha + (addr + gamma * read_value + gamma * gamma * <E1 as Engine>::Scalar::ONE))
            .invert()
            .unwrap())
    );
    // next_W check
    assert_eq!(
      next_W.get_value(),
      prev_W
        .get_value()
        .zip(alpha.get_value())
        .zip(gamma.get_value())
        .zip(addr.get_value())
        .zip(read_value.get_value())
        .map(|((((prev_W, alpha), gamma), addr), read_value)| {
          prev_W
            + (alpha + (addr + gamma * read_value + gamma * gamma * (<E1 as Engine>::Scalar::ONE)))
              .invert()
              .unwrap()
            + (alpha
              + (addr + gamma * read_value + gamma * gamma * (<E1 as Engine>::Scalar::from(2))))
            .invert()
            .unwrap()
        }),
    );

    let mut hasher = <E2 as Engine>::RO::new(ro_consts, 7);
    hasher.absorb(prev_intermediate_gamma.get_value().unwrap());
    hasher.absorb(addr.get_value().unwrap());
    hasher.absorb(<E1 as Engine>::Scalar::ZERO);
    hasher.absorb(<E1 as Engine>::Scalar::ZERO);
    hasher.absorb(addr.get_value().unwrap());
    hasher.absorb(read_value.get_value().unwrap());
    hasher.absorb(<E1 as Engine>::Scalar::ONE);
    let res = hasher.squeeze(NUM_CHALLENGE_BITS);
    assert_eq!(
      scalar_as_base::<E2>(res),
      next_intermediate_gamma.get_value().unwrap()
    );
    // TODO check rics is_sat
    // let (_, _) = cs.r1cs_shape_with_commitmentkey();
    // let (U1, W1) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // // Make sure that the first instance is satisfiable
    // assert!(shape.is_sat(&ck, &U1, &W1).is_ok());
  }
}
