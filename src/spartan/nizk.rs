//! This module defines nizk proofs
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
use crate::errors::NovaError;
use crate::traits::commitment::CommitmentEngineTrait;
use crate::traits::{
  commitment::{CommitmentTrait, ZKCommitmentEngineTrait, Len},
  TranscriptEngineTrait,
};
use crate::{Commitment, CommitmentKey, CompressedCommitment};
use ff::Field;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use crate::Engine;
use core::ops::{Add, Sub, Mul};

/// KnowledgeProof
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KnowledgeProof<E: Engine> {
  alpha: CompressedCommitment<E>,
  z1: <E as Engine>::Scalar,
  z2: <E as Engine>::Scalar,
}

/// EqualityProof
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EqualityProof<E: Engine>  where E::CE: ZKCommitmentEngineTrait<E> {
  /// alpha
  pub alpha: CompressedCommitment<E>,
  /// z
  pub z: <E as Engine>::Scalar,
}

/// ProductProof
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProductProof<E: Engine> {
  alpha: CompressedCommitment<E>,
  beta: CompressedCommitment<E>,
  delta: CompressedCommitment<E>,
  z: [<E as Engine>::Scalar; 5],
}

/// DocProductProof
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DotProductProof<E: Engine> {
  delta: CompressedCommitment<E>,
  beta: CompressedCommitment<E>,
  z: Vec<<E as Engine>::Scalar>,
  z_delta: <E as Engine>::Scalar,
  z_beta: <E as Engine>::Scalar,
}

/// KnowledgeProof
impl<E: Engine> KnowledgeProof<E> where E::CE: ZKCommitmentEngineTrait<E> {

  /// protocol name
  pub fn protocol_name() -> &'static [u8] {
    b"knowledge proof"
  }

  /// prove
  pub fn prove(
    ck_n: &CommitmentKey<E>,
    transcript: &mut <E as Engine>::TE,
    x: &<E as Engine>::Scalar,
    r: &<E as Engine>::Scalar,
  ) -> Result<(KnowledgeProof<E>, CompressedCommitment<E>), NovaError> {
    transcript.dom_sep(Self::protocol_name());

    // produce two random scalars
    let t1 = <E as Engine>::Scalar::random(&mut OsRng);
    let t2 = <E as Engine>::Scalar::random(&mut OsRng);

    let C = <E as Engine>::CE::zkcommit(ck_n, &[*x], r).compress();
    transcript.absorb(b"C", &C);

    let alpha = <E as Engine>::CE::zkcommit(ck_n, &[t1], &t2).compress();
    transcript.absorb(b"alpha", &alpha);

    let c = transcript.squeeze(b"c")?;

    let z1 = *x * c + t1;
    let z2 = *r * c + t2;

    Ok((Self { alpha, z1, z2 }, C))
  }

  /// verify
  pub fn verify(
    &self,
    ck_n: &CommitmentKey<E>,
    transcript: &mut <E as Engine>::TE,
    C: &CompressedCommitment<E>,
  ) -> Result<(), NovaError> {
    transcript.dom_sep(Self::protocol_name());
    transcript.absorb(b"C", C);
    transcript.absorb(b"alpha", &self.alpha);

    let c = transcript.squeeze(b"c")?;

    let lhs = <E as Engine>::CE::zkcommit(ck_n, &[self.z1], &self.z2).compress();
    let rhs = (Commitment::<E>::decompress(C)? * c
      + Commitment::<E>::decompress(&self.alpha)?)
    .compress();

    if lhs == rhs {
      Ok(())
    } else {
      Err(NovaError::InvalidZkKnowledgeProof)
    }
  }
}

/// EqualityProof
impl<E: Engine> EqualityProof<E> 
where 
E::CE: ZKCommitmentEngineTrait<E>, 
E::CE: CommitmentEngineTrait<E>,
<E::CE as CommitmentEngineTrait<E>>::Commitment: Sub<Output = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment>, 
<<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as Sub>::Output as Mul<<E as Engine>::Scalar>>::Output: Add<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment> {
  /// protocol name
  pub fn protocol_name() -> &'static [u8] {
    b"equality proof"
  }

  /// prove
  pub fn prove(
    ck_n: &CommitmentKey<E>,
    transcript: &mut <E as Engine>::TE,
    v1: &<E as Engine>::Scalar,
    s1: &<E as Engine>::Scalar,
    v2: &<E as Engine>::Scalar,
    s2: &<E as Engine>::Scalar,
  ) -> Result<(EqualityProof<E>, CompressedCommitment<E>, CompressedCommitment<E>), NovaError> {
    transcript.dom_sep(Self::protocol_name());

    // produce a random scalar
    let r = <E as Engine>::Scalar::random(&mut OsRng);

    let C1 = <E as Engine>::CE::zkcommit(ck_n, &[*v1], s1).compress();
    transcript.absorb(b"C1", &C1);

    let C2 = <E as Engine>::CE::zkcommit(ck_n, &[*v2], s2).compress();
    transcript.absorb(b"C2", &C2);

    let alpha = <E as Engine>::CE::zkcommit(
      ck_n,
      &[<E as Engine>::Scalar::ZERO],
      &r,
    )
    .compress(); // h^r
    transcript.absorb(b"alpha", &alpha);

    let c = transcript.squeeze(b"c")?;

    let z = c * (*s1 - *s2) + r;

    Ok((Self { alpha, z }, C1, C2))
  }

  /// verify
  pub fn verify(
    &self,
    gens_n: &CommitmentKey<E>,
    transcript: &mut <E as Engine>::TE,
    C1: &CompressedCommitment<E>,
    C2: &CompressedCommitment<E>,
  ) -> Result<(), NovaError> {
    transcript.dom_sep(Self::protocol_name());
    transcript.absorb(b"C1", C1);
    transcript.absorb(b"C2", C2);
    transcript.absorb(b"alpha", &self.alpha);

    let c = transcript.squeeze(b"c")?;

    let rhs = {
      let C: <E::CE as CommitmentEngineTrait<E>>::Commitment = Commitment::<E>::decompress(C1)?
        - Commitment::<E>::decompress(C2)?;
      (C * c + Commitment::<E>::decompress(&self.alpha)?).compress()
    };

    let lhs = <E as Engine>::CE::zkcommit(
      gens_n,
      &[<E as Engine>::Scalar::ZERO],
      &self.z,
    )
    .compress(); // h^z

    if lhs == rhs {
      Ok(())
    } else {
      Err(NovaError::InvalidZkEqualityProof)
    }
  }
}

/// product proof
impl<E: Engine> ProductProof<E> where E::CE: ZKCommitmentEngineTrait<E> {
  /// protocol name
  pub fn protocol_name() -> &'static [u8] {
    b"product proof"
  }

  /// prove
  pub fn prove(
    ck_n: &CommitmentKey<E>,
    transcript: &mut <E as Engine>::TE,
    x: &<E as Engine>::Scalar,
    rX: &<E as Engine>::Scalar,
    y: &<E as Engine>::Scalar,
    rY: &<E as Engine>::Scalar,
    z: &<E as Engine>::Scalar,
    rZ: &<E as Engine>::Scalar,
  ) -> Result<
    (
      ProductProof<E>,
      CompressedCommitment<E>,
      CompressedCommitment<E>,
      CompressedCommitment<E>,
    ),
    NovaError,
  > {
    transcript.dom_sep(Self::protocol_name());

    // produce 5 random scalars
    let b1 = <E as Engine>::Scalar::random(&mut OsRng);
    let b2 = <E as Engine>::Scalar::random(&mut OsRng);
    let b3 = <E as Engine>::Scalar::random(&mut OsRng);
    let b4 = <E as Engine>::Scalar::random(&mut OsRng);
    let b5 = <E as Engine>::Scalar::random(&mut OsRng);

    let X = <E as Engine>::CE::zkcommit(ck_n, &[*x], rX).compress();
    transcript.absorb(b"X", &X);

    let Y = <E as Engine>::CE::zkcommit(ck_n, &[*y], rY).compress();
    transcript.absorb(b"Y", &Y);

    let Z = <E as Engine>::CE::zkcommit(ck_n, &[*z], rZ).compress();
    transcript.absorb(b"Z", &Z);

    let alpha = <E as Engine>::CE::zkcommit(ck_n, &[b1], &b2).compress();
    transcript.absorb(b"alpha", &alpha);

    let beta = <E as Engine>::CE::zkcommit(ck_n, &[b3], &b4).compress();
    transcript.absorb(b"beta", &beta);

    let delta = {
      let h_to_b5 = <E as Engine>::CE::zkcommit(ck_n, &[<E as Engine>::Scalar::ZERO], &b5); // h^b5
      (Commitment::<E>::decompress(&X)? * b3 + h_to_b5).compress() // X^b3*h^b5
    };

    transcript.absorb(b"delta", &delta);

    let c = transcript.squeeze(b"c")?;

    let z1 = b1 + c * *x;
    let z2 = b2 + c * *rX;
    let z3 = b3 + c * *y;
    let z4 = b4 + c * *rY;
    let z5 = b5 + c * (*rZ - *rX * *y);
    let z = [z1, z2, z3, z4, z5];

    Ok((
      Self {
        alpha,
        beta,
        delta,
        z,
      },
      X,
      Y,
      Z,
    ))
  }

  /// check_equality
  fn check_equality(
    P: &CompressedCommitment<E>,
    X: &CompressedCommitment<E>,
    c: &<E as Engine>::Scalar,
    ck_n: &CommitmentKey<E>,
    z1: &<E as Engine>::Scalar,
    z2: &<E as Engine>::Scalar,
  ) -> Result<bool, NovaError> {
    let lhs = (Commitment::<E>::decompress(P)? + Commitment::<E>::decompress(X)? * *c).compress();
    let rhs = <E as Engine>::CE::zkcommit(ck_n, &[*z1], z2).compress();

    Ok(lhs == rhs)
  }

  /// verify
  pub fn verify(
    &self,
    ck_n: &CommitmentKey<E>,
    transcript: &mut <E as Engine>::TE,
    X: &CompressedCommitment<E>,
    Y: &CompressedCommitment<E>,
    Z: &CompressedCommitment<E>,
  ) -> Result<(), NovaError> {
    transcript.dom_sep(Self::protocol_name());

    transcript.absorb(b"X", X);
    transcript.absorb(b"Y", Y);
    transcript.absorb(b"Z", Z);
    transcript.absorb(b"alpha", &self.alpha);
    transcript.absorb(b"beta", &self.beta);
    transcript.absorb(b"delta", &self.delta);

    let z1 = self.z[0];
    let z2 = self.z[1];
    let z3 = self.z[2];
    let z4 = self.z[3];
    let z5 = self.z[4];

    let c = transcript.squeeze(b"c")?;

    let res = ProductProof::<E>::check_equality(&self.alpha, X, &c, ck_n, &z1, &z2)?
      && ProductProof::<E>::check_equality(&self.beta, Y, &c, ck_n, &z3, &z4)?;

    let res2 = {
      let lhs = (Commitment::<E>::decompress(&self.delta)? + Commitment::<E>::decompress(Z)? * c)
        .compress();

      let h_to_z5 = <E as Engine>::CE::zkcommit(ck_n, &[<E as Engine>::Scalar::ZERO], &z5); // h^z5
      let rhs = (Commitment::<E>::decompress(X)? * z3 + h_to_z5).compress(); // X^z3*h^z5
      lhs == rhs
    };

    if res && res2 {
      Ok(())
    } else {
      Err(NovaError::InvalidZkProductProof)
    }
  }
}

/// DotProductProof
impl<E: Engine> DotProductProof<E> where E::CE: ZKCommitmentEngineTrait<E> {
  /// protocol name
  pub fn protocol_name() -> &'static [u8] {
    b"dot product proof"
  }

  /// compute dot product
  pub fn compute_dotproduct(a: &[<E as Engine>::Scalar], b: &[<E as Engine>::Scalar]) -> <E as Engine>::Scalar {
    assert_eq!(a.len(), b.len());
    let mut result = <E as Engine>::Scalar::ZERO;

    for i in 0..a.len() {
      result += a[i] * b[i];
    }

    result
  }

  /// prove
  pub fn prove(
    ck_1: &CommitmentKey<E>, // generator of size 1
    ck_n: &CommitmentKey<E>, // generators of size n
    transcript: &mut <E as Engine>::TE,
    x_vec: &[<E as Engine>::Scalar],
    blind_x: &<E as Engine>::Scalar,
    a_vec: &[<E as Engine>::Scalar],
    y: &<E as Engine>::Scalar,
    blind_y: &<E as Engine>::Scalar,
  ) -> Result<(Self, CompressedCommitment<E>, CompressedCommitment<E>), NovaError> {
    transcript.dom_sep(Self::protocol_name());

    let n = x_vec.len();
    assert_eq!(x_vec.len(), a_vec.len());
    assert_eq!(ck_n.length(), a_vec.len());
    assert_eq!(ck_1.length(), 1);

    // produce randomness for the proofs
    let d_vec = (0..n)
      .map(|_i| <E as Engine>::Scalar::random(&mut OsRng))
      .collect::<Vec<<E as Engine>::Scalar>>();

    let r_delta = <E as Engine>::Scalar::random(&mut OsRng);
    let r_beta = <E as Engine>::Scalar::random(&mut OsRng);

    let Cx = <E as Engine>::CE::zkcommit(ck_n, x_vec, blind_x).compress();
    transcript.absorb(b"Cx", &Cx);

    let Cy = <E as Engine>::CE::zkcommit(ck_1, &[*y], blind_y).compress();
    transcript.absorb(b"Cy", &Cy);

    transcript.absorb(b"a", &a_vec);

    let delta = <E as Engine>::CE::zkcommit(ck_n, &d_vec, &r_delta).compress();
    transcript.absorb(b"delta", &delta);

    let dotproduct_a_d = DotProductProof::<E>::compute_dotproduct(a_vec, &d_vec);

    let beta = <E as Engine>::CE::zkcommit(ck_1, &[dotproduct_a_d], &r_beta).compress();
    transcript.absorb(b"beta", &beta);

    let c = transcript.squeeze(b"c")?;

    let z = (0..d_vec.len())
      .map(|i| c * x_vec[i] + d_vec[i])
      .collect::<Vec<<E as Engine>::Scalar>>();

    let z_delta = c * blind_x + r_delta;
    let z_beta = c * blind_y + r_beta;

    Ok((
      DotProductProof {
        delta,
        beta,
        z,
        z_delta,
        z_beta,
      },
      Cx,
      Cy,
    ))
  }

  /// verify
  pub fn verify(
    &self,
    ck_1: &CommitmentKey<E>, // generator of size 1
    ck_n: &CommitmentKey<E>, // generator of size n
    transcript: &mut <E as Engine>::TE,
    a_vec: &[<E as Engine>::Scalar],
    Cx: &CompressedCommitment<E>,
    Cy: &CompressedCommitment<E>,
  ) -> Result<(), NovaError> {
    assert_eq!(ck_n.length(), a_vec.len());
    assert_eq!(ck_1.length(), 1);

    transcript.dom_sep(Self::protocol_name());

    transcript.absorb(b"Cx", Cx);
    transcript.absorb(b"Cy", Cy);
    transcript.absorb(b"a", &a_vec);
    transcript.absorb(b"delta", &self.delta);
    transcript.absorb(b"beta", &self.beta);

    let c = transcript.squeeze(b"c")?;

    let mut result = Commitment::<E>::decompress(Cx)? * c
      + Commitment::<E>::decompress(&self.delta)?
      == <E as Engine>::CE::zkcommit(ck_n, &self.z, &self.z_delta);

    let dotproduct_z_a = DotProductProof::<E>::compute_dotproduct(&self.z, a_vec);
    result &= Commitment::<E>::decompress(Cy)? * c + Commitment::<E>::decompress(&self.beta)?
      == <E as Engine>::CE::zkcommit(ck_1, &[dotproduct_z_a], &self.z_beta);

    if result {
      Ok(())
    } else {
      Err(NovaError::InvalidZkDotProductProof)
    }
  }
}
