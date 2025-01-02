//! This module defines the needed wrong-field NIFS prover

use super::utils::absorb_U;
use crate::cyclefold::util::absorb_primary_commitment;
use crate::{
  constants::{NUM_CHALLENGE_BITS, NUM_FE_IN_EMULATED_POINT},
  errors::NovaError,
  gadgets::scalar_as_base,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROConstants, ROTrait},
  Commitment, CommitmentKey, CompressedCommitment,
};

/// A SNARK that holds the proof of a step of an incremental computation of the primary circuit
/// in the CycleFold folding scheme.
/// The difference of this folding scheme from the Nova NIFS in `src/nifs.rs` is that this
#[derive(Debug)]
pub struct RelaxedNIFS<E1>
where
  E1: CurveCycleEquipped,
{
  pub(crate) comm_T: CompressedCommitment<E1>,
}

impl<E1> RelaxedNIFS<E1>
where
  E1: CurveCycleEquipped,
{
  /// Takes a relaxed R1CS instance-witness pair (U1, W1) and an R1CS instance-witness pair (U2, W2)
  /// and folds them into a new relaxed R1CS instance-witness pair (U, W) and a commitment to the
  /// cross term T. It also provides the challenge r used to fold the instances.
  pub fn prove(
    ck: &CommitmentKey<E1>,
    ro_consts: &ROConstants<Dual<E1>>,
    pp_digest: &E1::Scalar,
    S: &R1CSShape<E1>,
    U1: &RelaxedR1CSInstance<E1>,
    W1: &RelaxedR1CSWitness<E1>,
    U2: &RelaxedR1CSInstance<E1>,
    W2: &RelaxedR1CSWitness<E1>,
  ) -> Result<
    (
      Self,
      (RelaxedR1CSInstance<E1>, RelaxedR1CSWitness<E1>),
      E1::Scalar,
    ),
    NovaError,
  > {
    let arity = U1.X.len();

    if arity != U2.X.len() {
      return Err(NovaError::InvalidInputLength);
    }

    let mut ro = <Dual<E1> as Engine>::RO::new(
      ro_consts.clone(),
      1 + 2 * NUM_FE_IN_EMULATED_POINT + arity + 1 + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + U.comm_E + U.X + U.u + comm_T
    );

    ro.absorb(*pp_digest);

    absorb_U::<E1>(U2, &mut ro);

    let (T, comm_T) = S.commit_T_relaxed(ck, U1, W1, U2, W2)?;

    absorb_primary_commitment::<E1, Dual<E1>>(&comm_T, &mut ro);

    let r = scalar_as_base::<Dual<E1>>(ro.squeeze(NUM_CHALLENGE_BITS));

    let U = U1.fold_relaxed(U2, &comm_T, &r);

    let W = W1.fold_relaxed(W2, &T, &r)?;

    Ok((
      Self {
        comm_T: comm_T.compress(),
      },
      (U, W),
      r,
    ))
  }

  #[allow(dead_code)]
  /// Takes as input a relaxed R1CS instance `U1` and R1CS instance `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  pub fn verify(
    &self,
    ro_consts: &ROConstants<Dual<E1>>,
    pp_digest: &E1::Scalar,
    U1: &RelaxedR1CSInstance<E1>,
    U2: &RelaxedR1CSInstance<E1>,
  ) -> Result<RelaxedR1CSInstance<E1>, NovaError> {
    let arity = U1.X.len();

    if arity != U2.X.len() {
      return Err(NovaError::InvalidInputLength);
    }

    // initialize a new RO
    let mut ro = <Dual<E1> as Engine>::RO::new(
      ro_consts.clone(),
      1 + 2 * NUM_FE_IN_EMULATED_POINT + arity + 1 + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + U.comm_E + U.X + U.u + comm_T
    );

    // append the digest of pp to the transcript
    ro.absorb(*pp_digest);

    // append U2 to transcript, U1 does not need to absorbed since U2.X[0] = Hash(params, U1, i, z0, zi)
    absorb_U::<E1>(U2, &mut ro);

    // append `comm_T` to the transcript and obtain a challenge
    let comm_T = Commitment::<E1>::decompress(&self.comm_T)?;
    absorb_primary_commitment::<E1, Dual<E1>>(&comm_T, &mut ro);

    // compute a challenge from the RO
    let r = scalar_as_base::<Dual<E1>>(ro.squeeze(NUM_CHALLENGE_BITS));

    // fold the instance using `r` and `comm_T`
    let U = U1.fold_relaxed(U2, &comm_T, &r);

    // return the folded instance
    Ok(U)
  }
}
