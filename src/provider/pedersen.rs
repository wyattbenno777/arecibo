//! This module provides an implementation of a commitment engine
use crate::{
  errors::NovaError,
  frontend::{AllocatedBit, ConstraintSystem, SynthesisError},
  gadgets::AllocatedPoint,
  provider::traits::DlogGroup,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    AbsorbInROTrait, Engine, ROTrait, TranscriptReprTrait,
  },
  zip_with,
};
use core::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, Mul, MulAssign},
};
use ff::{Field, PrimeFieldBits};
use group::{
  prime::{PrimeCurve, PrimeCurveAffine},
  Curve, Group, GroupEncoding,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::iter::zip;

/// A type that holds commitment generators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentKey<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  // this is a hack; we just assume the size of the element.
  // Look for the static assertions in provider macros for a justification
  pub(crate) ck: Vec<<E::GE as PrimeCurve>::Affine>,
  pub(crate) h: Option<<E::GE as PrimeCurve>::Affine>,
}

impl<E> Len for CommitmentKey<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  fn length(&self) -> usize {
    self.ck.len()
  }
}

/// A type that holds a commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<E: Engine> {
  // this is a hack; we just assume the size of the element.
  // Look for the static assertions in provider macros for a justification
  pub(crate) comm: E::GE,
}

/// A type that holds a compressed commitment
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedCommitment<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  pub(crate) comm: <E::GE as DlogGroup>::Compressed,
}

impl<E> CommitmentTrait<E> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  type CompressedCommitment = CompressedCommitment<E>;

  fn compress(&self) -> Self::CompressedCommitment {
    CompressedCommitment {
      comm: <E::GE as GroupEncoding>::to_bytes(&self.comm).into(),
    }
  }

  fn to_coordinates(&self) -> (E::Base, E::Base, bool) {
    self.comm.to_coordinates()
  }

  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, NovaError> {
    let opt_comm = <<E as Engine>::GE as GroupEncoding>::from_bytes(&c.comm.clone().into());
    let Some(comm) = Option::from(opt_comm) else {
      return Err(NovaError::DecompressionError);
    };
    Ok(Self { comm })
  }

  fn reinterpret_as_generator(&self) -> <<E as Engine>::GE as PrimeCurve>::Affine {
    self.comm.to_affine()
  }
}

impl<E> Default for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn default() -> Self {
    Self {
      comm: E::GE::identity(),
    }
  }
}

impl<E> TranscriptReprTrait<E::GE> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    let is_infinity_byte = (!is_infinity).into();
    [
      x.to_transcript_bytes(),
      y.to_transcript_bytes(),
      [is_infinity_byte].to_vec(),
    ]
    .concat()
  }
}

impl<E> AbsorbInROTrait<E> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    ro.absorb(x);
    ro.absorb(y);
    ro.absorb(if is_infinity {
      E::Base::ONE
    } else {
      E::Base::ZERO
    });
  }
}

impl<E> TranscriptReprTrait<E::GE> for CompressedCommitment<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.comm.to_transcript_bytes()
  }
}

impl<E> MulAssign<E::Scalar> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  fn mul_assign(&mut self, scalar: E::Scalar) {
    *self = Self {
      comm: self.comm * scalar,
    };
  }
}

impl<'b, E> Mul<&'b E::Scalar> for &'_ Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  type Output = Commitment<E>;
  fn mul(self, scalar: &'b E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E> Mul<E::Scalar> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  type Output = Self;

  fn mul(self, scalar: E::Scalar) -> Self {
    Self {
      comm: self.comm * scalar,
    }
  }
}

impl<E> Add for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  type Output = Self;

  fn add(self, other: Self) -> Self {
    Self {
      comm: self.comm + other.comm,
    }
  }
}

/// A type that holds blinding generator
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerandKey<E: Engine>
where
  E::GE: DlogGroup,
{
  h: <E::GE as PrimeCurve>::Affine,
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentEngine<E> {
  _p: PhantomData<E>,
}

impl<E> CommitmentEngineTrait<E> for CommitmentEngine<E>
where
  E: Engine,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  type CommitmentKey = CommitmentKey<E>;
  type Commitment = Commitment<E>;
  type DerandKey = DerandKey<E>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    let gens = E::GE::from_label(label, n.next_power_of_two() + 1);
    let (h, ck) = gens.split_first().unwrap();

    Self::CommitmentKey {
      ck: ck.to_vec(),
      h: Some(*h),
    }
  }

  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey {
    assert!(ck.h.is_some());
    Self::DerandKey {
      h: *ck.h.as_ref().unwrap(),
    }
  }

  fn commit_at(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &E::Scalar,
    idx: usize,
  ) -> Self::Commitment {
    assert!(ck.ck.len() > idx);
    assert!(ck.ck.len() - idx >= v.len());
    if ck.h.is_some() {
      let mut scalars: Vec<E::Scalar> = v.to_vec();
      scalars.push(*r);
      let mut bases = ck.ck[idx..idx + v.len()].to_vec();
      bases.push(*ck.h.as_ref().unwrap());
      Commitment {
        comm: E::GE::vartime_multiscalar_mul(&scalars, &bases),
      }
    } else {
      assert_eq!(*r, E::Scalar::ZERO);
      Commitment {
        comm: E::GE::vartime_multiscalar_mul(v, &ck.ck[idx..idx + v.len()]),
      }
    }
  }

  fn commit_gadget<CS: ConstraintSystem<E::Base>>(
    cs: &mut CS,
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    r: &E::Scalar,
  ) -> Result<AllocatedPoint<E::GE>, SynthesisError> {
    let mut scalars = v.to_vec();
    scalars.push(*r);

    let mut bases = ck.ck[..v.len()].to_vec();
    bases.push(*ck.h.as_ref().unwrap());

    let mut acc = AllocatedPoint::<E::GE>::default(cs.namespace(|| "allocate zero"));
    for (i, (s, b)) in zip(scalars, bases).enumerate() {
      let mut s_bits = Vec::new();
      // TODO: Optimize. This looks very inefficient
      for (j, bit) in s.to_le_bits().iter().enumerate() {
        let allocated_bit = AllocatedBit::alloc(
          cs.namespace(|| format!("allocate bit {}-{}", i, j)),
          Some(*bit),
        )?;
        s_bits.push(allocated_bit);
      }
      let coordinates = b.to_curve().to_coordinates();
      let alloc_b =
        AllocatedPoint::<E::GE>::alloc(cs.namespace(|| "allocate bases"), Some(coordinates))?;
      let res = alloc_b.scalar_mul(cs.namespace(|| "scalar mul"), &s_bits)?;
      acc = acc.add(cs.namespace(|| format!("add {}", i)), &res)?;
    }
    Ok(acc)
  }

  fn derandomize(
    dk: &Self::DerandKey,
    commit: &Self::Commitment,
    r: &E::Scalar,
  ) -> Self::Commitment {
    Commitment {
      comm: commit.comm - <E::GE as DlogGroup>::group(&dk.h) * r,
    }
  }
}

/// A trait listing properties of a commitment key that can be managed in a divide-and-conquer fashion
pub trait CommitmentKeyExtTrait<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  /// Splits the commitment key into two pieces at a specified point
  fn split_at(self, n: usize) -> (Self, Self)
  where
    Self: Sized;

  /// Combines two commitment keys into one
  fn combine(&self, other: &Self) -> Self;

  /// Folds the two commitment keys into one using the provided weights
  fn fold(L: &Self, R: &Self, w1: &E::Scalar, w2: &E::Scalar) -> Self;

  /// Scales the commitment key using the provided scalar
  fn scale(&mut self, r: &E::Scalar);

  /// Reinterprets commitments as commitment keys
  fn reinterpret_commitments_as_ck(
    c: &[<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as CommitmentTrait<E>>::CompressedCommitment],
  ) -> Result<Self, NovaError>
  where
    Self: Sized;
}

impl<E> CommitmentKeyExtTrait<E> for CommitmentKey<E>
where
  E: Engine<CE = CommitmentEngine<E>>,
  E::GE: DlogGroup<ScalarExt = E::Scalar>,
{
  fn split_at(self, n: usize) -> (Self, Self) {
    (
      CommitmentKey {
        ck: self.ck[0..n].to_vec(),
        h: self.h,
      },
      CommitmentKey {
        ck: self.ck[n..].to_vec(),
        h: self.h,
      },
    )
  }

  fn combine(&self, other: &Self) -> Self {
    let ck = {
      let mut c = self.ck.clone();
      c.extend(other.ck.clone());
      c
    };
    CommitmentKey { ck, h: self.h }
  }

  // combines the left and right halves of `self` using `w1` and `w2` as the weights
  fn fold(L: &Self, R: &Self, w1: &E::Scalar, w2: &E::Scalar) -> Self {
    debug_assert!(L.ck.len() == R.ck.len());
    let ck_curve: Vec<E::GE> = zip_with!(par_iter, (L.ck, R.ck), |l, r| {
      E::GE::vartime_multiscalar_mul(&[*w1, *w2], &[*l, *r])
    })
    .collect();
    let mut ck_affine = vec![<E::GE as PrimeCurve>::Affine::identity(); L.ck.len()];
    E::GE::batch_normalize(&ck_curve, &mut ck_affine);

    Self {
      ck: ck_affine,
      h: L.h,
    }
  }

  /// Scales each element in `self` by `r`
  fn scale(&mut self, r: &E::Scalar) {
    let ck_scaled: Vec<E::GE> = self.ck.par_iter().map(|g| *g * r).collect();
    E::GE::batch_normalize(&ck_scaled, &mut self.ck);
  }

  /// reinterprets a vector of commitments as a set of generators
  fn reinterpret_commitments_as_ck(c: &[CompressedCommitment<E>]) -> Result<Self, NovaError> {
    let d = c
      .par_iter()
      .map(|c| Commitment::<E>::decompress(c).map(|c| c.comm))
      .collect::<Result<Vec<E::GE>, NovaError>>()?;
    let mut ck = vec![<E::GE as PrimeCurve>::Affine::identity(); d.len()];
    E::GE::batch_normalize(&d, &mut ck);
    Ok(Self { ck, h: None })
  }
}
