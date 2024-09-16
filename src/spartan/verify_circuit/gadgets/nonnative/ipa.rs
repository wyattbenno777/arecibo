//! This module implements `EvaluationEngine` using an IPA-based polynomial commitment scheme

use crate::gadgets::AllocatedPoint;

use crate::spartan::verify_circuit::gadgets::poseidon_transcript::PoseidonTranscript;
use crate::traits::commitment::CommitmentTrait;

use crate::{
  errors::{NovaError, PCSError},
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup, util::field::batch_invert},
  spartan::polys::eq::EqPolynomial,
  traits::{commitment::CommitmentEngineTrait, Engine},
  Commitment, CommitmentKey, CE,
};

use bellpepper_core::ConstraintSystem;
use core::iter;
use ff::Field;

use generic_array::typenum::U24;

use poseidon_sponge::sponge::api::SpongeOp;
use poseidon_sponge::sponge::vanilla::Sponge;
use poseidon_sponge::sponge::vanilla::SpongeTrait;
use poseidon_sponge::Strength;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::spartan::verify_circuit::ipa_prover_poseidon::ipa_pc::{
  inner_product, InnerProductArgument, InnerProductInstance, VerifierKey,
};
/// Provides an implementation of a polynomial evaluation engine using IPA
#[derive(Clone, Debug)]
pub struct EvaluationEngineGadget<E> {
  _p: PhantomData<E>,
}

impl<E> EvaluationEngineGadget<E>
where
  E: Engine,
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  /// A method to verify purported evaluations of a batch of polynomials
  pub fn verify<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    vk: &VerifierKey<E>,
    comm: &Commitment<E>,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &InnerProductArgument<E>,
  ) -> Result<(), NovaError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::evals_from_points(point), eval);

    let mut io_pattern = vec![SpongeOp::Absorb(1), SpongeOp::Squeeze(1)];
    for _i in 0..usize::try_from(u.b_vec.len().ilog2()).unwrap() {
      io_pattern.push(SpongeOp::Squeeze(1));
    }
    let constants = Sponge::<E::Scalar, U24>::api_constants(Strength::Standard);
    let mut transcript = PoseidonTranscript::<E>::new(&constants);

    let arg_gadget = IPAGadget { ipa: arg.clone() };

    arg_gadget.verify(
      cs.namespace(|| "verify"),
      &vk.ck_v,
      vk.ck_s.clone(),
      1 << point.len(),
      &u,
      &mut transcript,
    )?;

    Ok(())
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct IPAGadget<E: Engine> {
  pub ipa: InnerProductArgument<E>,
}

impl<E> IPAGadget<E>
where
  E: Engine,
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  pub fn verify<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    ck: &CommitmentKey<E>,
    mut ck_c: CommitmentKey<E>,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut PoseidonTranscript<'_, E>,
  ) -> Result<(), NovaError> {
    let (ck, _) = ck.clone().split_at(U.b_vec.len());

    // transcript.dom_sep(Self::protocol_name());
    if U.b_vec.len() != n
      || n != (1 << self.ipa.L_vec.len())
      || self.ipa.L_vec.len() != self.ipa.R_vec.len()
      || self.ipa.L_vec.len() >= 32
    {
      return Err(NovaError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    // transcript.absorb(b"U", U);
    transcript.absorb(U.c);

    // sample a random base for committing to the inner product
    let r = transcript.squeeze(String::from("ipa r"))?;
    ck_c.scale(&r);

    let P = U.comm_a_vec + CE::<E>::commit(&ck_c, &[U.c]);

    // compute a vector of public coins using self.ipa.L_vec and self.ipa.R_vec
    let r = (0..self.ipa.L_vec.len())
      .map(|i| {
        // transcript.absorb(b"L", &self.ipa.L_vec[i]);
        // transcript.absorb(b"R", &self.ipa.R_vec[i]);
        transcript.squeeze(format!("prove inner r_{i}"))
      })
      .collect::<Result<Vec<E::Scalar>, NovaError>>()?;

    // precompute scalars necessary for verification
    let r_square: Vec<E::Scalar> = (0..self.ipa.L_vec.len())
      .into_par_iter()
      .map(|i| r[i] * r[i])
      .collect();
    let r_inverse = batch_invert(r.clone())?;
    let r_inverse_square: Vec<E::Scalar> = (0..self.ipa.L_vec.len())
      .into_par_iter()
      .map(|i| r_inverse[i] * r_inverse[i])
      .collect();

    // compute the vector with the tensor structure
    let s = {
      let mut s = vec![E::Scalar::ZERO; n];
      s[0] = {
        let mut v = E::Scalar::ONE;
        for r_inverse_i in r_inverse {
          v *= r_inverse_i;
        }
        v
      };
      for i in 1..n {
        let pos_in_r = (31 - (i as u32).leading_zeros()) as usize;
        s[i] = s[i - (1 << pos_in_r)] * r_square[(self.ipa.L_vec.len() - 1) - pos_in_r];
      }
      s
    };

    let ck_hat = {
      let c = CE::<E>::commit(&ck, &s).compress();
      CommitmentKey::<E>::reinterpret_commitments_as_ck(&[c])?
    };

    let b_hat = inner_product(&U.b_vec, &s);

    let P_hat = {
      let ck_folded = {
        let ck_L = CommitmentKey::<E>::reinterpret_commitments_as_ck(&self.ipa.L_vec)?;
        let ck_R = CommitmentKey::<E>::reinterpret_commitments_as_ck(&self.ipa.R_vec)?;
        let ck_P = CommitmentKey::<E>::reinterpret_commitments_as_ck(&[P.compress()])?;
        ck_L.combine(&ck_R).combine(&ck_P)
      };

      CE::<E>::commit(
        &ck_folded,
        &r_square
          .iter()
          .chain(r_inverse_square.iter())
          .chain(iter::once(&E::Scalar::ONE))
          .copied()
          .collect::<Vec<E::Scalar>>(),
      )
    };

    let check = CE::<E>::commit(
      &ck_hat.combine(&ck_c),
      &[self.ipa.a_hat, self.ipa.a_hat * b_hat],
    );
    if P_hat != check {
      return Err(NovaError::PCSError(PCSError::InvalidPCS));
    }

    let alloc_check = AllocatedPoint::<E::GE>::alloc(
      cs.namespace(|| "allocate C_2"),
      Some(check.to_coordinates()),
    )?;

    let alloc_P_hat = AllocatedPoint::<E::GE>::alloc(
      cs.namespace(|| "allocate P_hat"),
      Some(P_hat.to_coordinates()),
    )?;

    alloc_check.enforce_equal(cs.namespace(|| "check equality"), &alloc_P_hat)?;

    Ok(())
  }
}
