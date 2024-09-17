//! batched zk pp snark
//!
//!

use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::{
    zkppsnark::{ZKR1CSShapeSparkCommitment, R1CSShapeSparkRepr},
  },
  traits::{
    commitment::{CommitmentEngineTrait, Len, ZKCommitmentEngineTrait},
    zkevaluation::EvaluationEngineTrait,
    snark::{BatchedRelaxedR1CSSNARKTrait, DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine,
  },
  CommitmentKey,
};
//use crate::spartan::nizk::ProductProof;
use core::ops::{Add, Sub, Mul};
use crate::spartan::zksnark::SumcheckGens;
use crate::provider::zk_pedersen;
use crate::provider::zk_ipa_pc;
use crate::provider::traits::DlogGroup;
use core::slice;
//use ff::Field;
//use itertools::{chain, Itertools as _};
use once_cell::sync::*;
//use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
//use rand_core::OsRng;

//use super::zksumcheck::ZKSumcheckProof;
use crate::provider::zk_pedersen::CommitmentKeyExtTrait;
use unzip_n::unzip_n;

unzip_n!(pub 3);
unzip_n!(pub 5);

/// A type that represents the prover's key
#[derive(Debug)]
#[allow(unused)]
pub struct ProverKey<E: Engine> {
  pk_ee: zk_ipa_pc::ProverKey<E>,
  sumcheck_gens: SumcheckGens<E>,
  S_repr: Vec<R1CSShapeSparkRepr<E>>,
  S_comm: Vec<ZKR1CSShapeSparkCommitment<E>>,
  vk_digest: E::Scalar, // digest of verifier's key
}


/// A type that represents the verifier's key
#[derive(Debug, Serialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine> {
  vk_ee: zk_ipa_pc::VerifierKey<E>,
  S_comm: Vec<ZKR1CSShapeSparkCommitment<E>>,
  sumcheck_gens: SumcheckGens<E>,
  num_vars: Vec<usize>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E> VerifierKey<E>
where 
    E: Engine + Serialize + for<'de> Deserialize<'de>,
    <E as Engine>::CE: CommitmentEngineTrait<E>,
    <E as Engine>::GE: DlogGroup<ScalarExt = E::Scalar>,
    <E as Engine>::CE: ZKCommitmentEngineTrait<E>,
    <<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey: CommitmentKeyExtTrait<E>,
{
    fn new(
        num_vars: Vec<usize>,
        S_comm: Vec<ZKR1CSShapeSparkCommitment<E>>,
        vk_ee: zk_ipa_pc::VerifierKey<E>,
    ) -> Self
    where
      E::CE: CommitmentEngineTrait<E>,
    {
        let scalar_gen = zk_ipa_pc::EvaluationEngine::get_scalar_gen_vk(vk_ee.clone());

        Self {
            num_vars,
            S_comm,
            vk_ee,
            sumcheck_gens: SumcheckGens::<E>::new(b"gens_s", &scalar_gen),
            digest: Default::default(),
        }
    }
}


impl<E: Engine> SimpleDigestible for VerifierKey<E> {}

impl<E: Engine> DigestHelperTrait<E> for VerifierKey<E> {
  /// Returns the digest of the verifier's key
  fn digest(&self) -> E::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::new(self);
        dc.digest()
      })
      .cloned()
      .expect("Failure to retrieve digest!")
  }
}


/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  deserialize = "E: Deserialize<'de>"
))]
pub struct BatchedRelaxedR1CSSNARK<E: Engine + Serialize> {
  rand_sc: Vec<E::Scalar>,
}


impl<E: Engine + Serialize + for<'de> Deserialize<'de>> BatchedRelaxedR1CSSNARKTrait<E>
  for BatchedRelaxedR1CSSNARK<E> 
where 
  E: Engine<CE = zk_pedersen::CommitmentEngine<E>>,
  <E as Engine>::CE: ZKCommitmentEngineTrait<E>,
  <E as Engine>::GE: DlogGroup<ScalarExt = E::Scalar>,
  E::CE: CommitmentEngineTrait<E>,
  E::CE: CommitmentEngineTrait<E, Commitment = zk_pedersen::Commitment<E>, CommitmentKey = zk_pedersen::CommitmentKey<E>>,
  <E::CE as CommitmentEngineTrait<E>>::Commitment: Add<Output = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment>, 
  <E::CE as CommitmentEngineTrait<E>>::Commitment: Sub<Output = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment>, 
  <<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as Sub>::Output as Mul<<E as Engine>::Scalar>>::Output: Add<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment> 
{
  type ProverKey = ProverKey<E>;
  type VerifierKey = VerifierKey<E>;

  fn ck_floor() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
    Box::new(|shape: &R1CSShape<E>| -> usize {
      // the commitment key should be large enough to commit to the R1CS matrices
      std::cmp::max(
        shape.A.len() + shape.B.len() + shape.C.len(),
        std::cmp::max(shape.num_cons, 2 * shape.num_vars),
      )
    })
  }

  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: Vec<&R1CSShape<E>>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    for s in S.iter() {
      // check the provided commitment key meets minimal requirements
      if ck.length() < <Self as BatchedRelaxedR1CSSNARKTrait<_>>::ck_floor()(s) {
        // return Err(NovaError::InvalidCommitmentKeyLength);
        return Err(NovaError::InternalError);
      }
    }
    let (pk_ee, vk_ee) = zk_ipa_pc::EvaluationEngine::setup(ck.clone());

    let S = S.iter().map(|s| s.pad()).collect::<Vec<_>>();
    let S_repr = S.iter().map(R1CSShapeSparkRepr::new).collect::<Vec<_>>();
    let S_comm = S_repr
      .iter()
      .map(|s_repr| s_repr.commit(&*ck))
      .collect::<Vec<_>>();
    let num_vars = S.iter().map(|s| s.num_vars).collect::<Vec<_>>();
    let vk = VerifierKey::new(num_vars, S_comm.clone(), vk_ee);

    let scalar_gen: crate::provider::zk_pedersen::CommitmentKey<E> = zk_ipa_pc::EvaluationEngine::get_scalar_gen_pk(pk_ee.clone());
    let pk = ProverKey {
      pk_ee,
      sumcheck_gens: SumcheckGens::<E>::new(b"gens_s", &scalar_gen),
      S_repr,
      S_comm,
      vk_digest: vk.digest(),
    };
    Ok((pk, vk))
  }

  fn prove(
    _ck: &CommitmentKey<E>,
    _pk: &Self::ProverKey,
    _S: Vec<&R1CSShape<E>>,
    _U: &[RelaxedR1CSInstance<E>],
    _W: &[RelaxedR1CSWitness<E>],
  ) -> Result<Self, NovaError> {
    
    let rand_sc = vec![E::Scalar::from(12)];

    Ok(Self {
      rand_sc
    })
  }

  fn verify(&self, _vk: &Self::VerifierKey, _U: &[RelaxedR1CSInstance<E>]) -> Result<(), NovaError> {
    Ok(())
  }
}


impl<E: Engine + Serialize + for<'de> Deserialize<'de>> RelaxedR1CSSNARKTrait<E>
  for BatchedRelaxedR1CSSNARK<E> 
where 
  E: Engine<CE = zk_pedersen::CommitmentEngine<E>>,
  E::CE: ZKCommitmentEngineTrait<E>, 
  <E as Engine>::GE: DlogGroup<ScalarExt = <E as Engine>::Scalar>,
  E::CE: CommitmentEngineTrait<E, Commitment = zk_pedersen::Commitment<E>, CommitmentKey = zk_pedersen::CommitmentKey<E>>,
{
  type ProverKey = ProverKey<E>;

  type VerifierKey = VerifierKey<E>;

  fn ck_floor() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
    <Self as BatchedRelaxedR1CSSNARKTrait<E>>::ck_floor()
  }

  fn setup(
    ck: Arc<CommitmentKey<E>>,
    S: &R1CSShape<E>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    <Self as BatchedRelaxedR1CSSNARKTrait<E>>::setup(ck, vec![S])
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
  ) -> Result<Self, NovaError> {
    let slice_U = slice::from_ref(U);
    let slice_W = slice::from_ref(W);

    <Self as BatchedRelaxedR1CSSNARKTrait<E>>::prove(ck, pk, vec![S], slice_U, slice_W)
  }

  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<E>) -> Result<(), NovaError> {
    let slice = slice::from_ref(U);
    <Self as BatchedRelaxedR1CSSNARKTrait<E>>::verify(self, vk, slice)
  }
}
