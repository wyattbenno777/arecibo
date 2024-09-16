use crate::{errors::NovaError, traits::Engine};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, Namespace, SynthesisError};
use core::marker::PhantomData;

use generic_array::typenum::U24;
use poseidon_sponge::{
  circuit2::Elt,
  poseidon::PoseidonConstants,
  sponge::{
    api::{IOPattern, SpongeAPI},
    circuit::SpongeCircuit,
    vanilla::{Mode::Simplex, Sponge, SpongeTrait},
  },
};
use std::collections::{BTreeMap, HashSet};

/// A Poseidon-based Transcript to use outside circuits
pub struct PoseidonTranscript<'a, E>
where
  E: Engine,
{
  challenges: BTreeMap<String, E::Scalar>,
  sponge: Sponge<'a, E::Scalar, U24>,
}

impl<'a, E: Engine> PoseidonTranscript<'a, E> {
  pub fn new(constants: &'a PoseidonConstants<E::Scalar, U24>) -> Self {
    let mut sponge = Sponge::<E::Scalar, U24>::new_with_constants(constants, Simplex);
    sponge.start(IOPattern(vec![]), None, &mut ());
    Self {
      challenges: BTreeMap::new(),
      sponge,
    }
  }

  /// Compute a challenge by hashing the current state
  pub fn squeeze(&mut self, label: String) -> Result<E::Scalar, NovaError> {
    let acc = &mut ();

    let hash = SpongeAPI::squeeze(&mut self.sponge, 1, acc)[0];

    if label != *"" {
      // if self.challenges.contains_key(&label) {
      //   panic!("Challenge label {} already exists", label);
      // }
      self.challenges.insert(label, hash);
    }

    Ok(hash)
  }

  pub fn absorb(&mut self, element: E::Scalar) {
    let acc = &mut ();
    SpongeAPI::absorb(&mut self.sponge, 1, &[element], acc);
  }

  fn _dom_sep(&mut self, _bytes: &'static [u8]) {
    todo!()
  }
}

/// A Poseidon-based Transcript to use inside circuits
pub struct PoseidonTranscriptCircuit<E>
where
  E: Engine,
{
  challenges: HashSet<String>,
  _p: PhantomData<E>,
}

impl<E: Engine> PoseidonTranscriptCircuit<E> {
  pub fn new<'a, CS>(
    sponge: &mut SpongeCircuit<'a, E::Scalar, U24, CS>,
    acc: &mut Namespace<'a, E::Scalar, CS>,
  ) -> Self
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    sponge.start(IOPattern(vec![]), None, acc);
    Self {
      challenges: HashSet::new(),
      _p: PhantomData,
    }
  }

  /// Compute a challenge by hashing the current state
  pub fn squeeze<'a, CS>(
    &mut self,
    label: String,
    sponge: &mut SpongeCircuit<'a, E::Scalar, U24, CS>,
    acc: &mut Namespace<'a, E::Scalar, CS>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let hash = SpongeAPI::squeeze(sponge, 1, acc);
    let hash = Elt::ensure_allocated(
      &hash[0],
      &mut acc.namespace(|| format!("ensure allocated {label}")),
      true,
    )?;

    if label != *"" {
      // if self.challenges.contains(&label) {
      //   panic!("Challenge label {} already exists", label);
      // }
      self.challenges.insert(label);
    }

    Ok(hash)
  }

  pub fn absorb<'a, CS>(
    &mut self,
    element: AllocatedNum<E::Scalar>,
    sponge: &mut SpongeCircuit<'a, E::Scalar, U24, CS>,
    acc: &mut Namespace<'a, E::Scalar, CS>,
  ) where
    CS: ConstraintSystem<E::Scalar>,
  {
    let elt_el = Elt::Allocated(element);
    SpongeAPI::absorb(sponge, 1, &[elt_el], acc);
  }

  fn _dom_sep(&mut self, _bytes: &'static [u8]) {
    todo!()
  }
}

#[cfg(test)]
mod tests {
  use crate::provider::PallasEngine;
  use crate::spartan::verify_circuit::gadgets::poseidon_transcript::PoseidonTranscriptCircuit;
  use bellpepper_core::num::AllocatedNum;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use bellpepper_core::ConstraintSystem;
  use poseidon_sponge::sponge::circuit::SpongeCircuit;
  use poseidon_sponge::sponge::vanilla::Mode::Simplex;
  use poseidon_sponge::sponge::vanilla::SpongeTrait;

  use generic_array::typenum::U24;
  use pasta_curves::pallas::Scalar;

  use poseidon_sponge::sponge::vanilla::Sponge;
  use poseidon_sponge::Strength;

  use super::PoseidonTranscript;
  #[test]
  fn test_poseidon_transcript() {
    let constants = Sponge::<Scalar, U24>::api_constants(Strength::Standard);
    let mut transcript = PoseidonTranscript::<PallasEngine>::new(&constants);

    let num = Scalar::from(42u64);
    transcript.absorb(num);
    let num = Scalar::from(100u64);
    transcript.absorb(num);

    let _hash = transcript.squeeze("test".to_string()).unwrap();

    let num = Scalar::from(42u64);
    transcript.absorb(num);

    let _hash2 = transcript.squeeze("test2".to_string()).unwrap();
  }

  #[test]
  fn test_poseidon_transcript_circuit() {
    let constants = Sponge::<Scalar, U24>::api_constants(Strength::Standard);
    let mut sponge = SpongeCircuit::<Scalar, U24, _>::new_with_constants(&constants, Simplex);

    let mut cs = TestConstraintSystem::<Scalar>::new();
    let mut ns = cs.namespace(|| "ns");

    let mut transcript = PoseidonTranscriptCircuit::<PallasEngine>::new(&mut sponge, &mut ns);

    let num = AllocatedNum::alloc(ns.namespace(|| "num"), || Ok(Scalar::from(42u64))).unwrap();
    transcript.absorb(num, &mut sponge, &mut ns);

    let num = AllocatedNum::alloc(ns.namespace(|| "num2"), || Ok(Scalar::from(100u64))).unwrap();
    transcript.absorb(num, &mut sponge, &mut ns);

    let _hash = transcript
      .squeeze("tau".to_string(), &mut sponge, &mut ns)
      .unwrap();

    let num = AllocatedNum::alloc(ns.namespace(|| "num3"), || Ok(Scalar::from(42u64))).unwrap();
    transcript.absorb(num, &mut sponge, &mut ns);

    let _hash2 = transcript
      .squeeze("tau2".to_string(), &mut sponge, &mut ns)
      .unwrap();

    let root_cs = ns.get_root();

    assert!(root_cs.is_satisfied());
  }

  #[test]
  fn test_default_and_circuit() {
    let constants = Sponge::<Scalar, U24>::api_constants(Strength::Standard);
    let mut sponge_circuit =
      SpongeCircuit::<Scalar, U24, _>::new_with_constants(&constants, Simplex);
    let mut cs = TestConstraintSystem::<Scalar>::new();
    let mut ns = cs.namespace(|| "ns");

    let constants = Sponge::<Scalar, U24>::api_constants(Strength::Standard);
    let mut transcript = PoseidonTranscript::<PallasEngine>::new(&constants);
    let mut transcript_circuit =
      PoseidonTranscriptCircuit::<PallasEngine>::new(&mut sponge_circuit, &mut ns);

    let num = Scalar::from(42u64);
    let num1 = Scalar::from(100u64);

    let alloc_num = AllocatedNum::alloc(ns.namespace(|| "num"), || Ok(num)).unwrap();
    let alloc_num1 = AllocatedNum::alloc(ns.namespace(|| "num1"), || Ok(num1)).unwrap();

    transcript.absorb(num);
    transcript.absorb(num1);

    transcript_circuit.absorb(alloc_num.clone(), &mut sponge_circuit, &mut ns);
    transcript_circuit.absorb(alloc_num1.clone(), &mut sponge_circuit, &mut ns);

    let hash = transcript.squeeze("test".to_string()).unwrap();
    let alloc_hash = transcript_circuit
      .squeeze("test".to_string(), &mut sponge_circuit, &mut ns)
      .unwrap();

    assert_eq!(hash, alloc_hash.get_value().unwrap());

    let num2 = Scalar::from(42u64);
    let alloc_num2 = AllocatedNum::alloc(ns.namespace(|| "num2"), || Ok(num2)).unwrap();

    transcript.absorb(num2);
    transcript_circuit.absorb(alloc_num2.clone(), &mut sponge_circuit, &mut ns);

    let hash2 = transcript.squeeze("test2".to_string()).unwrap();
    let alloc_hash2 = transcript_circuit
      .squeeze("test2".to_string(), &mut sponge_circuit, &mut ns)
      .unwrap();

    assert_eq!(hash2, alloc_hash2.get_value().unwrap());
  }
}
