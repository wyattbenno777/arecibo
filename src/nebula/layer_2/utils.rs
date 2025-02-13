use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_CHALLENGE_BITS},
  cyclefold::util::absorb_primary_commitment,
  errors::NovaError,
  gadgets::{f_to_nat, nat_to_limbs, scalar_as_base},
  nebula::nifs::PrimaryRelaxedNIFS,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    commitment::CommitmentEngineTrait, AbsorbInROTrait, CurveCycleEquipped, Dual, Engine,
    ROConstants, ROTrait,
  },
  Commitment, CommitmentKey,
};
use ff::PrimeFieldBits;
use serde::{Deserialize, Serialize};

use super::nifs::NIFS;

pub(crate) fn absorb_U<E1>(U: &RelaxedR1CSInstance<E1>, ro: &mut impl ROTrait<E1::Scalar, E1::Base>)
where
  E1: CurveCycleEquipped,
{
  absorb_primary_commitment::<E1, Dual<E1>>(&U.comm_W, ro);
  absorb_primary_commitment::<E1, Dual<E1>>(&U.comm_E, ro);
  ro.absorb(U.u);
  for x in &U.X {
    ro.absorb(*x);
  }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub(crate) struct RelaxedFoldingData<E: Engine> {
  pub U1: RelaxedR1CSInstance<E>,
  pub U2: RelaxedR1CSInstance<E>,
  pub T: Commitment<E>,
}

pub fn scalar_to_bools<E>(scalar: E::Scalar) -> Option<[bool; NUM_CHALLENGE_BITS]>
where
  E: Engine,
{
  scalar
    .to_le_bits()
    .iter()
    .map(|b| Some(*b))
    .take(NUM_CHALLENGE_BITS)
    .collect::<Option<Vec<_>>>()
    .map(|v| v.try_into().unwrap())
}

pub(crate) fn absorb_U_bn<E1>(U: &RelaxedR1CSInstance<E1>, ro: &mut E1::RO)
where
  E1: Engine,
{
  U.comm_W.absorb_in_ro(ro);
  U.comm_E.absorb_in_ro(ro);
  let u_limbs: Vec<E1::Scalar> = nat_to_limbs(&f_to_nat(&U.u), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
  for limb in u_limbs {
    ro.absorb(scalar_as_base::<E1>(limb));
  }

  // absorb each element of self.X in bignum format
  for x in &U.X {
    let limbs: Vec<E1::Scalar> = nat_to_limbs(&f_to_nat(x), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
    for limb in limbs {
      ro.absorb(scalar_as_base::<E1>(limb));
    }
  }
}

#[derive(Debug, Clone)]
pub struct Layer2FoldingData<E>
where
  E: CurveCycleEquipped,
{
  pub(crate) pp_digest: Option<E::Scalar>,
  pub(crate) nifs: Option<NIFS<E>>,
  pub(crate) U1: Option<RelaxedR1CSInstance<E>>,
  pub(crate) U2: Option<RelaxedR1CSInstance<E>>,
  pub(crate) E_new: Option<Commitment<E>>,
  pub(crate) W_new: Option<Commitment<E>>,
  pub(crate) U2_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
}

impl<E> Layer2FoldingData<E>
where
  E: CurveCycleEquipped,
{
  pub fn new(
    pp_digest: Option<E::Scalar>,
    nifs: Option<NIFS<E>>,
    U1: Option<RelaxedR1CSInstance<E>>,
    U2: Option<RelaxedR1CSInstance<E>>,
    E_new: Option<Commitment<E>>,
    W_new: Option<Commitment<E>>,
    U2_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
  ) -> Self {
    Self {
      pp_digest,
      nifs,
      U1,
      U2,
      E_new,
      W_new,
      U2_secondary,
    }
  }
}

pub fn random_fold_and_derandom<E>(
  S: &R1CSShape<E>,
  ck: &CommitmentKey<E>,
  ro_const: &ROConstants<Dual<E>>,
  digest: E::Scalar,
  U: &RelaxedR1CSInstance<E>,
  W: &RelaxedR1CSWitness<E>,
) -> Result<
  (
    RelaxedR1CSInstance<E>,
    RelaxedR1CSWitness<E>,
    PrimaryRelaxedNIFS<E>,
    E::Scalar,
    E::Scalar,
    RelaxedR1CSInstance<E>,
  ),
  NovaError,
>
where
  E: CurveCycleEquipped,
{
  // Fold random instance and witness
  let (random_U, random_W) = S.sample_random_instance_witness(ck)?;
  let (nifs_r, (U, W), _) =
    PrimaryRelaxedNIFS::prove(ck, ro_const, &digest, S, (U, W), (&random_U, &random_W))?;
  let (derandom_W, wit_blind, err_blind) = W.derandomize();
  let derandom_U = U.derandomize(&E::CE::derand_key(ck), &wit_blind, &err_blind);
  Ok((
    derandom_U, derandom_W, nifs_r, wit_blind, err_blind, random_U,
  ))
}
