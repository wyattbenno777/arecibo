//! batched zk pp snark
//!
//!

use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::{
    math::Math,
    polys::{
      eq::EqPolynomial,
      identity::IdentityPolynomial,
      masked_eq::MaskedEqPolynomial,
      multilinear::{MultilinearPolynomial, SparsePolynomial},
      power::PowPolynomial,
      univariate::{CompressedUniPoly, UniPoly},
    },
    powers,
    zkppsnark::{ZKR1CSShapeSparkCommitment, R1CSShapeSparkRepr},
    zksumcheck::{engine::{
      InnerSumcheckInstance, MemorySumcheckInstance, OuterSumcheckInstance, ZKSumcheckEngine,
      WitnessBoundSumcheck,
    }},
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len, ZKCommitmentEngineTrait},
    zkevaluation::EvaluationEngineTrait,
    snark::{BatchedRelaxedR1CSSNARKTrait, DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine, TranscriptEngineTrait,
  },
  zip_with, zip_with_for_each, Commitment, CommitmentKey, CompressedCommitment,
};
use crate::spartan::nizk::ProductProof;
use core::ops::{Add, Sub, Mul};
use crate::spartan::zksnark::SumcheckGens;
use crate::provider::zk_pedersen;
use crate::provider::traits::DlogGroup;
use core::slice;
use ff::Field;
use itertools::{chain, Itertools as _};
use once_cell::sync::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rand_core::OsRng;

use super::zksumcheck::ZKSumcheckProof;

use unzip_n::unzip_n;

unzip_n!(pub 3);
unzip_n!(pub 5);

/// A type that represents the prover's key
#[derive(Debug)]
pub struct ProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  sumcheck_gens: SumcheckGens<E>,
  S_repr: Vec<R1CSShapeSparkRepr<E>>,
  S_comm: Vec<ZKR1CSShapeSparkCommitment<E>>,
  vk_digest: E::Scalar, // digest of verifier's key
}

/// A type that represents the verifier's key
#[derive(Debug, Serialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine, EE: EvaluationEngineTrait<E>>  {
  vk_ee: EE::VerifierKey,
  S_comm: Vec<ZKR1CSShapeSparkCommitment<E>>,
  sumcheck_gens: SumcheckGens<E>,
  num_vars: Vec<usize>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}
impl<E: Engine, EE: EvaluationEngineTrait<E>> VerifierKey<E, EE>
where 
  <E as Engine>::CE: CommitmentEngineTrait<E>,
  <E as Engine>::GE: DlogGroup,
  <E as Engine>::CE: ZKCommitmentEngineTrait<E>,
{
  fn new(
    num_vars: Vec<usize>,
    S_comm: Vec<ZKR1CSShapeSparkCommitment<E>>,
    vk_ee: EE::VerifierKey,
  ) -> Self {
    let scalar_gen = EE::get_scalar_gen_vk(vk_ee.clone());

    Self {
      num_vars,
      S_comm,
      vk_ee,
      sumcheck_gens: SumcheckGens::<E>::new(b"gens_s", &scalar_gen),
      digest: Default::default(),
    }
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for VerifierKey<E, EE> {}

impl<E: Engine, EE: EvaluationEngineTrait<E>> DigestHelperTrait<E> for VerifierKey<E, EE> {
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
pub struct BatchedRelaxedR1CSSNARK<E: Engine + Serialize, EE: EvaluationEngineTrait<E>> {
  // commitment to oracles: the first three are for Az, Bz, Cz,
  // and the last two are for memory reads
  comms_Az_Bz_Cz: Vec<[CompressedCommitment<E>; 3]>,
  comms_L_row_col: Vec<[CompressedCommitment<E>; 2]>,
  // commitments to aid the memory checks
  // [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col]
  comms_mem_oracles: Vec<[CompressedCommitment<E>; 4]>,

  // claims about Az, Bz, and Cz polynomials
  comm_evals_Az_Bz_Cz_at_tau: Vec<[CompressedCommitment<E>; 3]>,

  // sum-check
  sc: ZKSumcheckProof<E>,

  // claims from the end of sum-check
  comm_evals_Az_Bz_Cz_W_E: Vec<[CompressedCommitment<E>; 5]>,

  prod_comm_Az_Bz: Vec<(ProductProof<E>, CompressedCommitment<E>)>,

  comm_evals_L_row_col: Vec<[CompressedCommitment<E>; 2]>,

  prod_comm_L_row_L_col: Vec<(ProductProof<E>, CompressedCommitment<E>)>,
  // [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col]
  comm_evals_mem_oracle: Vec<[CompressedCommitment<E>; 4]>,

  prod_comm_L_row_L_col_val_A_B_C: Vec<[(ProductProof<E>, CompressedCommitment<E>); 3]>,
  // [val_A, val_B, val_C, row, col, ts_row, ts_col]
  comm_evals_mem_preprocessed: Vec<[CompressedCommitment<E>; 7]>,

  comm_claims: Vec<CompressedCommitment<E>>,

  comm_z_vec: Vec<CompressedCommitment<E>>,

  prods_comms_w_plus_r_inv_row_blind_row: Vec<(ProductProof<E>, CompressedCommitment<E>)>,
  prods_comms_w_plus_r_inv_row_blind_col: Vec<(ProductProof<E>, CompressedCommitment<E>)>,
  prods_comms_w_plus_r_inv_row_L_row: Vec<(ProductProof<E>, CompressedCommitment<E>)>,
  prods_comms_w_plus_r_inv_row_L_col: Vec<(ProductProof<E>, CompressedCommitment<E>)>,
  prods_comms_t_plus_r_inv_col_Z: Vec<(ProductProof<E>, CompressedCommitment<E>)>,


  // a PCS evaluation argument
  eval_arg: EE::EvaluationArgument,
}

impl<E: Engine + Serialize + for<'de> Deserialize<'de>, EE: EvaluationEngineTrait<E>> BatchedRelaxedR1CSSNARKTrait<E>
  for BatchedRelaxedR1CSSNARK<E, EE> 
  where 
  EE: EvaluationEngineTrait<E>,
  <E as Engine>::CE: ZKCommitmentEngineTrait<E>,
  <E as Engine>::GE: DlogGroup<ScalarExt = E::Scalar>,
  E::CE: CommitmentEngineTrait<E>,
  E::CE: CommitmentEngineTrait<E, Commitment = zk_pedersen::Commitment<E>, CommitmentKey = zk_pedersen::CommitmentKey<E>>,
  <E::CE as CommitmentEngineTrait<E>>::Commitment: Add<Output = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment>, 
  <E::CE as CommitmentEngineTrait<E>>::Commitment: Sub<Output = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment>, 
  // <E::CE as CommitmentEngineTrait<E>>::Commitment: Mul<<E as Engine>::Scalar, Output = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment>, 
  <<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as Sub>::Output as Mul<<E as Engine>::Scalar>>::Output: Add<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment> 
{
  type ProverKey = ProverKey<E, EE>;
  type VerifierKey = VerifierKey<E, EE>;

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
    let (pk_ee, vk_ee) = EE::setup(ck.clone());

    let S = S.iter().map(|s| s.pad()).collect::<Vec<_>>();
    let S_repr = S.iter().map(R1CSShapeSparkRepr::new).collect::<Vec<_>>();
    let S_comm = S_repr
      .iter()
      .map(|s_repr| s_repr.commit(&*ck))
      .collect::<Vec<_>>();
    let num_vars = S.iter().map(|s| s.num_vars).collect::<Vec<_>>();
    let vk = VerifierKey::new(num_vars, S_comm.clone(), vk_ee);

    let scalar_gen: crate::provider::zk_pedersen::CommitmentKey<E> = EE::get_scalar_gen_pk(pk_ee.clone());
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
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: Vec<&R1CSShape<E>>,
    U: &[RelaxedR1CSInstance<E>],
    W: &[RelaxedR1CSWitness<E>],
  ) -> Result<Self, NovaError> {
    // Pad shapes so that num_vars = num_cons = Nᵢ and check the sizes are correct
    let S = S.par_iter().map(|s| s.pad()).collect::<Vec<_>>();

    // N[i] = max(|Aᵢ|+|Bᵢ|+|Cᵢ|, 2*num_varsᵢ, num_consᵢ)
    let Nis = pk.S_repr.iter().map(|s| s.N).collect::<Vec<_>>();
    assert!(Nis.iter().all(|&Ni| Ni.is_power_of_two()));
    let N_max = *Nis.iter().max().unwrap();

    let num_instances = U.len();

    // Pad [(Wᵢ,Eᵢ)] to the next power of 2 (not to Ni)
    let W = zip_with!(par_iter, (W, S), |w, s| w.pad(s)).collect::<Vec<RelaxedR1CSWitness<E>>>();

    // number of rounds of sum-check
    let num_rounds_sc = N_max.log_2();

    // Initialize transcript with vk || [Uᵢ]
    let mut transcript = E::TE::new(b"BatchedRelaxedR1CSSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);
    if num_instances > 1 {
      let num_instances_field = E::Scalar::from(num_instances as u64);
      transcript.absorb(b"n", &num_instances_field);
    }
    transcript.absorb(b"U", &U);

    // Append public inputs to Wᵢ: Zᵢ = [Wᵢ, uᵢ, Xᵢ]
    let polys_Z = zip_with!(par_iter, (W, U, Nis), |W, U, Ni| {
      // poly_Z will be resized later, so we preallocate the correct capacity
      let mut poly_Z = Vec::with_capacity(*Ni);
      poly_Z.extend(W.W.iter().chain([&U.u]).chain(U.X.iter()));
      poly_Z
    })
    .collect::<Vec<Vec<E::Scalar>>>();

    // Move polys_W and polys_E, as well as U.u out of U
    let (comms_W_E, us): (Vec<_>, Vec<_>) = U.iter().map(|U| ([U.comm_W, U.comm_E], U.u)).unzip();
    let (polys_W, polys_E): (Vec<_>, Vec<_>) = W.into_iter().map(|w| (w.W, w.E)).unzip();

    // Compute [Az, Bz, Cz]
    let mut polys_Az_Bz_Cz = zip_with!(par_iter, (polys_Z, S), |z, s| {
      let (Az, Bz, Cz) = s.multiply_vec(z)?;
      Ok([Az, Bz, Cz])
    })
    .collect::<Result<Vec<_>, NovaError>>()?;

    let blind_Az = <E as Engine>::Scalar::random(&mut OsRng);

    let blind_Bz = <E as Engine>::Scalar::random(&mut OsRng);

    let blind_Cz = <E as Engine>::Scalar::random(&mut OsRng);

    // Commit to [Az, Bz, Cz] and add to transcript
    let comms_Az_Bz_Cz = polys_Az_Bz_Cz
      .par_iter()
      .map(|[Az, Bz, Cz]| {
        let (comm_Az, (comm_Bz, comm_Cz)) = rayon::join(
          || E::CE::zkcommit(ck, Az, &blind_Az),
          || rayon::join(|| E::CE::zkcommit(ck, Bz, &blind_Bz), || E::CE::zkcommit(ck, Cz, &blind_Cz)),
        );
        [comm_Az, comm_Bz, comm_Cz]
      })
      .collect::<Vec<_>>();
    comms_Az_Bz_Cz
      .iter()
      .for_each(|comms| transcript.absorb(b"c", &comms.as_slice()));

    // Compute eq(tau) for each instance in log2(Ni) variables
    let tau = transcript.squeeze(b"t")?;
    let all_taus = PowPolynomial::squares(&tau, N_max.log_2());

    let (polys_tau, coords_tau): (Vec<_>, Vec<_>) = Nis
      .par_iter()
      .map(|&N_i| {
        let log_Ni = N_i.log_2();
        let eqp: EqPolynomial<_> = all_taus[..log_Ni].iter().cloned().collect();
        let evals = eqp.evals();
        let coords = eqp.r;
        (evals, coords)
      })
      .unzip();

    // Pad [Az, Bz, Cz] to Ni
    polys_Az_Bz_Cz
      .par_iter_mut()
      .zip_eq(Nis.par_iter())
      .for_each(|(az_bz_cz, &Ni)| {
        az_bz_cz
          .par_iter_mut()
          .for_each(|mz| mz.resize(Ni, E::Scalar::ZERO))
      });

    // Evaluate and commit to [Az(tau), Bz(tau), Cz(tau)]
    let evals_Az_Bz_Cz_at_tau = zip_with!(
      par_iter,
      (polys_Az_Bz_Cz, coords_tau),
      |ABCs, tau_coords| {
        let [Az, Bz, Cz] = ABCs;
        let (eval_Az, (eval_Bz, eval_Cz)) = rayon::join(
          || MultilinearPolynomial::evaluate_with(Az, tau_coords),
          || {
            rayon::join(
              || MultilinearPolynomial::evaluate_with(Bz, tau_coords),
              || MultilinearPolynomial::evaluate_with(Cz, tau_coords),
            )
          },
        );
        [eval_Az, eval_Bz, eval_Cz]
      }
    )
    .collect::<Vec<_>>();

    let comm_evals_Az_Bz_Cz_at_tau =  evals_Az_Bz_Cz_at_tau.iter().map(|e| {
      e.iter().map(|eval| {
        let blind_eval = <E as Engine>::Scalar::random(&mut OsRng);
        E::CE::zkcommit(ck, &[*eval], &blind_eval).compress()
      }).collect::<Vec<_>>()
      .try_into()
      .unwrap()
    }).collect::<Vec<_>>();

    // absorb the claimed evaluations into the transcript
    for evals in evals_Az_Bz_Cz_at_tau.iter() {
      transcript.absorb(b"e", &evals.as_slice());
    }

    // Pad Zᵢ, E to Nᵢ
    let polys_Z = polys_Z
      .into_par_iter()
      .zip_eq(Nis.par_iter())
      .map(|(mut poly_Z, &Ni)| {
        poly_Z.resize(Ni, E::Scalar::ZERO);
        poly_Z
      })
      .collect::<Vec<_>>();

    // Pad both W,E to have the same size. This is inefficient for W since the second half is empty,
    // but it makes it easier to handle the batching at the end.
    let polys_E = polys_E
      .into_par_iter()
      .zip_eq(Nis.par_iter())
      .map(|(mut poly_E, &Ni)| {
        poly_E.resize(Ni, E::Scalar::ZERO);
        poly_E
      })
      .collect::<Vec<_>>();

    let polys_W = polys_W
      .into_par_iter()
      .zip_eq(Nis.par_iter())
      .map(|(mut poly_W, &Ni)| {
        poly_W.resize(Ni, E::Scalar::ZERO);
        poly_W
      })
      .collect::<Vec<_>>();

    // (2) send commitments to the following two oracles
    // L_row(i) = eq(tau, row(i)) for all i in [0..Nᵢ]
    // L_col(i) = z(col(i)) for all i in [0..Nᵢ]
    let polys_L_row_col = zip_with!(
      par_iter,
      (S, Nis, polys_Z, polys_tau),
      |S, Ni, poly_Z, poly_tau| {
        let mut L_row = vec![poly_tau[0]; *Ni]; // we place mem_row[0] since resized row is appended with 0s
        let mut L_col = vec![poly_Z[Ni - 1]; *Ni]; // we place mem_col[Ni-1] since resized col is appended with Ni-1

        for (i, (val_r, val_c)) in S
          .A
          .iter()
          .chain(S.B.iter())
          .chain(S.C.iter())
          .map(|(r, c, _)| (poly_tau[r], poly_Z[c]))
          .enumerate()
        {
          L_row[i] = val_r;
          L_col[i] = val_c;
        }

        [L_row, L_col]
      }
    )
    .collect::<Vec<_>>();

    let blind_L_row = <E as Engine>::Scalar::random(&mut OsRng);
    let blind_L_col = <E as Engine>::Scalar::random(&mut OsRng);

    let comms_L_row_col = polys_L_row_col
      .par_iter()
      .map(|[L_row, L_col]| {
        let (comm_L_row, comm_L_col) =
          rayon::join(|| E::CE::zkcommit(ck, L_row, &blind_L_row), || E::CE::zkcommit(ck, L_col, &blind_L_col));
        [comm_L_row, comm_L_col]
      })
      .collect::<Vec<_>>();

    // absorb commitments to L_row and L_col in the transcript
    for comms in comms_L_row_col.iter() {
      transcript.absorb(b"e", &comms.as_slice());
    }

    // For each instance, batch Mz = Az + c*Bz + c^2*Cz
    let c = transcript.squeeze(b"c")?;

    let polys_Mz: Vec<_> = polys_Az_Bz_Cz
      .par_iter()
      .map(|polys_Az_Bz_Cz| {
        let poly_vec: Vec<&Vec<_>> = polys_Az_Bz_Cz.iter().collect();
        let w = PolyEvalWitness::<E>::batch(&poly_vec[..], &c);
        w.p
      })
      .collect();

    let evals_Mz: Vec<_> = zip_with!(
      iter,
      (comms_Az_Bz_Cz, evals_Az_Bz_Cz_at_tau),
      |comm_Az_Bz_Cz, evals_Az_Bz_Cz_at_tau| {
        let u = PolyEvalInstance::<E>::batch(
          comm_Az_Bz_Cz.as_slice(),
          vec![], // ignored by the function
          evals_Az_Bz_Cz_at_tau.as_slice(),
          &c,
        );
        u.e
      }
    )
    .collect();

    // we now need to prove three claims for each instance
    // (outer)
    //   0 = \sum_x poly_tau(x) * (poly_Az(x) * poly_Bz(x) - poly_uCz_E(x))
    //   eval_Az_at_tau + c * eval_Bz_at_tau + c^2 * eval_Cz_at_tau = (Az+c*Bz+c^2*Cz)(tau)
    // (inner)
    //   eval_Az_at_tau + c * eval_Bz_at_tau + c^2 * eval_Cz_at_tau = \sum_y L_row(y) * (val_A(y) + c * val_B(y) + c^2 * val_C(y)) * L_col(y)
    // (mem)
    //   L_row(i) = eq(tau, row(i))
    //   L_col(i) = z(col(i))
    let outer_sc_inst = zip_with!(
      (
        polys_Az_Bz_Cz.par_iter(),
        polys_E.par_iter(),
        polys_Mz.into_par_iter(),
        polys_tau.par_iter(),
        evals_Mz.par_iter(),
        us.par_iter()
      ),
      |poly_ABC, poly_E, poly_Mz, poly_tau, eval_Mz, u| {
        let [poly_Az, poly_Bz, poly_Cz] = poly_ABC;
        let poly_uCz_E = zip_with!(par_iter, (poly_Cz, poly_E), |cz, e| *u * cz + e).collect();
        OuterSumcheckInstance::new(
          poly_tau.clone(),
          poly_Az.clone(),
          poly_Bz.clone(),
          poly_uCz_E,
          poly_Mz, // Mz = Az + c * Bz + c^2 * Cz
          eval_Mz, // eval_Az_at_tau + c * eval_Az_at_tau + c^2 * eval_Cz_at_tau
        )
      }
    )
    .collect::<Vec<_>>();

    let inner_sc_inst = zip_with!(
      par_iter,
      (pk.S_repr, evals_Mz, polys_L_row_col),
      |s_repr, eval_Mz, poly_L| {
        let [poly_L_row, poly_L_col] = poly_L;
        let c_square = c.square();
        let val = zip_with!(
          par_iter,
          (s_repr.val_A, s_repr.val_B, s_repr.val_C),
          |v_a, v_b, v_c| *v_a + c * *v_b + c_square * *v_c
        )
        .collect::<Vec<_>>();

        InnerSumcheckInstance::new(
          *eval_Mz,
          MultilinearPolynomial::new(poly_L_row.clone()),
          MultilinearPolynomial::new(poly_L_col.clone()),
          MultilinearPolynomial::new(val),
        )
      }
    )
    .collect::<Vec<_>>();

    // a third sum-check instance to prove the read-only memory claim
    // we now need to prove that L_row and L_col are well-formed
    let (mem_sc_inst, comms_mem_oracles, _polys_mem_oracles) = {
      let gamma = transcript.squeeze(b"g")?;
      let r = transcript.squeeze(b"r")?;

      // We start by computing oracles and auxiliary polynomials to help prove the claim
      // oracles correspond to [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col]
      let (comms_mem_oracles, polys_mem_oracles, mem_aux) = pk
        .S_repr
        .iter()
        .zip_eq(polys_tau.iter())
        .zip_eq(polys_Z.iter())
        .zip_eq(polys_L_row_col.iter())
        .try_fold(
          (Vec::new(), Vec::new(), Vec::new()),
          |(mut comms, mut polys, mut aux), (((s_repr, poly_tau), poly_Z), [L_row, L_col])| {
            let (comm, poly, a) = MemorySumcheckInstance::<E>::compute_oracles(
              ck,
              &r,
              &gamma,
              poly_tau,
              &s_repr.row,
              L_row,
              &s_repr.ts_row,
              poly_Z,
              &s_repr.col,
              L_col,
              &s_repr.ts_col,
            )?;

            comms.push(comm);
            polys.push(poly);
            aux.push(a);

            Ok::<_, NovaError>((comms, polys, aux))
          },
        )?;

      // Commit to oracles
      for comms in comms_mem_oracles.iter() {
        transcript.absorb(b"l", &comms.as_slice());
      }

      // Sample new random variable for eq polynomial
      let rho = transcript.squeeze(b"r")?;
      let all_rhos = PowPolynomial::squares(&rho, N_max.log_2());

      let instances = zip_with!(
        (
          pk.S_repr.par_iter(),
          Nis.par_iter(),
          polys_mem_oracles.par_iter(),
          mem_aux.into_par_iter()
        ),
        |s_repr, Ni, polys_mem_oracles, polys_aux| {
          MemorySumcheckInstance::<E>::new(
            polys_mem_oracles.clone(),
            polys_aux,
            PowPolynomial::evals_with_powers(&all_rhos, Ni.log_2()),
            s_repr.ts_row.clone(),
            s_repr.ts_col.clone(),
          )
        }
      )
      .collect::<Vec<_>>();
      (instances, comms_mem_oracles, polys_mem_oracles)
    };

    let witness_sc_inst = zip_with!(par_iter, (polys_W, S), |poly_W, S| {
      WitnessBoundSumcheck::new(tau, poly_W.clone(), S.num_vars)
    })
    .collect::<Vec<_>>();

    // Run batched Sumcheck for the 3 claims for all instances.
    // Note that the polynomials for claims relating to instance i have size Ni.
    let (sc, rand_sc, claims_outer, claims_inner, claims_mem, claims_witness) = Self::prove_helper(
      ck,
      num_rounds_sc,
      mem_sc_inst,
      outer_sc_inst,
      inner_sc_inst,
      witness_sc_inst,
      &mut transcript,
    )?;

    // number of rounds of sum-check
    let num_rounds = pk.S_comm.iter().map(|s| s.N.log_2()).collect::<Vec<_>>();
    let num_rounds_max = *num_rounds.iter().max().unwrap();

    let (evals_Az_Bz_Cz_W_E, evals_L_row_col, evals_mem_oracle, evals_mem_preprocessed, comm_evals_Az_Bz_Cz_W_E, prod_comm_Az_Bz, comm_evals_L_row_col, prod_comm_L_row_L_col, comm_evals_mem_oracle, comm_evals_mem_preprocessed, _, prod_comm_L_row_L_col_val_A_B_C, comm_z_vec, prods_comms_w_plus_r_inv_row_blind_row, prods_comms_w_plus_r_inv_row_blind_col, prods_comms_w_plus_r_inv_row_L_row, prods_comms_w_plus_r_inv_row_L_col, prods_comms_t_plus_r_inv_col_Z) = {
      let evals_Az_Bz = claims_outer
        .into_iter()
        .map(|claims| [claims[0][0], claims[0][1]])
        .collect::<Vec<_>>();

      let evals_L_row_col = claims_inner
        .into_iter()
        .map(|claims| {
          // [L_row, L_col]
          [claims[0][0], claims[0][1]]
        })
        .collect::<Vec<_>>();

        let (comm_evals_L_row_col, blinds_L_row_L_col, prod_comm_L_row_L_col): (Vec<_>, Vec<_>, Vec<_>) =  evals_L_row_col.iter().map(|[L_row, L_col]| {
          let blind_L_row = <E as Engine>::Scalar::random(&mut OsRng);
          let blind_L_col = <E as Engine>::Scalar::random(&mut OsRng);
          let blind_prod_L_row_L_col = <E as Engine>::Scalar::random(&mut OsRng);

          let (proof_prod_L_row_L_col, comm_L_row, comm_L_col, comm_prod_L_row_L_col): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {
            let prod = *L_row * *L_col;
            ProductProof::prove(
              &pk.sumcheck_gens.ck_1,
              &mut transcript,
              L_row,
              &blind_L_row,
              L_col,
              &blind_L_col,
              &prod,
              &blind_prod_L_row_L_col,
            )
          }.unwrap();
          ([comm_L_row, comm_L_col], [blind_L_row, blind_L_col], (proof_prod_L_row_L_col, comm_prod_L_row_L_col))
        }).collect::<Vec<_>>().into_iter().unzip_n_vec();

      let (evals_mem_oracle, evals_mem_ts): (Vec<_>, Vec<_>) = claims_mem
        .into_iter()
        .map(|claims| {
          (
            // [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col]
            [claims[0][0], claims[0][1], claims[1][0], claims[1][1]],
            // [ts_row, ts_col]
            [claims[0][2], claims[1][2]],
          )
        })
        .unzip();

        let (comm_evals_mem_oracle, blind_evals_mem_oracle): (Vec<_>, Vec<_>) =  evals_mem_oracle.iter().map(|e| {
          let [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col] = e;

          let blind_t_plus_r_inv_row = <E as Engine>::Scalar::random(&mut OsRng);
          let comm_t_plus_r_inv_row = E::CE::zkcommit(ck, &[*t_plus_r_inv_row], &blind_t_plus_r_inv_row).compress();

          let blind_w_plus_r_inv_row = <E as Engine>::Scalar::random(&mut OsRng);
          let comm_w_plus_r_inv_row = E::CE::zkcommit(ck, &[*w_plus_r_inv_row], &blind_w_plus_r_inv_row).compress();

          let blind_t_plus_r_inv_col = <E as Engine>::Scalar::random(&mut OsRng);
          let comm_t_plus_r_inv_col = E::CE::zkcommit(ck, &[*t_plus_r_inv_col], &blind_t_plus_r_inv_col).compress();

          let blind_w_plus_r_inv_col = <E as Engine>::Scalar::random(&mut OsRng);
          let comm_w_plus_r_inv_col = E::CE::zkcommit(ck, &[*w_plus_r_inv_col], &blind_w_plus_r_inv_col).compress();

          ([comm_t_plus_r_inv_row, comm_w_plus_r_inv_row, comm_t_plus_r_inv_col, comm_w_plus_r_inv_col], [blind_w_plus_r_inv_row, blind_t_plus_r_inv_col, blind_w_plus_r_inv_col])
        }).collect::<Vec<_>>().into_iter().unzip();

      let evals_W = claims_witness
        .into_iter()
        .map(|claims| claims[0][0])
        .collect::<Vec<_>>();

      let (evals_Cz_E, evals_mem_val_row_col): (Vec<_>, Vec<_>) = zip_with!(
        iter,
        (polys_Az_Bz_Cz, polys_E, pk.S_repr),
        |ABCzs, poly_E, s_repr| {
          let [_, _, Cz] = ABCzs;
          let log_Ni = s_repr.N.log_2();
          let (_, rand_sc) = rand_sc.split_at(num_rounds_sc - log_Ni);
          let rand_sc_evals = EqPolynomial::evals_from_points(rand_sc);
          let e = [
            Cz,
            poly_E,
            &s_repr.val_A,
            &s_repr.val_B,
            &s_repr.val_C,
            &s_repr.row,
            &s_repr.col,
          ]
          .into_iter()
          .map(|p| {
            // Manually compute evaluation to avoid recomputing rand_sc_evals
            zip_with!(par_iter, (p, rand_sc_evals), |p, eq| *p * eq).sum()
          })
          .collect::<Vec<E::Scalar>>();
          ([e[0], e[1]], [e[2], e[3], e[4], e[5], e[6]])
        }
      )
      .unzip();

      let evals_Az_Bz_Cz_W_E = zip_with!(
        (evals_Az_Bz.into_iter(), evals_Cz_E.into_iter(), evals_W),
        |Az_Bz, Cz_E, W| {
          let [Az, Bz] = Az_Bz;
          let [Cz, E] = Cz_E;
          [Az, Bz, Cz, W, E]
        }
      )
      .collect::<Vec<_>>();

      // let comm_prod_Az_Bz = evals_Az_Bz_Cz_W_E.into_iter().map(|[Az, Bz, _, _, _]|{
      //   let blind_prod_Az_Bz = <E as Engine>::Scalar::random(&mut OsRng);
      //   let (proof_prod_eval_Az_eval_Bz, comm_Az, comm_Bz, comm_prod_eval_Az_eval_Bz): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {
      //     let prod = Az * Bz;
      //     ProductProof::prove(
      //       &pk.sumcheck_gens.ck_1,
      //       &mut transcript,
      //       &Az,
      //       &blind_Az,
      //       &Bz,
      //       &blind_Bz,
      //       &prod,
      //       &blind_prod_Az_Bz,
      //     )
      //   }.unwrap();
      // });



      // let tau = transcript.squeeze(b"t")?;
      // let tau_coords = PowPolynomial::new(&tau, num_rounds_max).coordinates();  

      let (comm_evals_Az_Bz_Cz_W_E, prod_comm_Az_Bz, _): (Vec<_>, Vec<_>, Vec<_>) =  evals_Az_Bz_Cz_W_E.iter().map(|[Az, Bz, Cz, W, E]| {
        // e.iter().map(|eval| {
        //   let blind_eval = <E as Engine>::Scalar::random(&mut OsRng);
        //   E::CE::zkcommit(ck, &[*eval], &blind_eval).compress()
        // }).collect::<Vec<_>>()
        // .try_into()
        // .unwrap()

        let blind_prod_Az_Bz = <E as Engine>::Scalar::random(&mut OsRng);
        let (proof_prod_eval_Az_eval_Bz, comm_Az, comm_Bz, comm_prod_eval_Az_eval_Bz): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {
          let prod = *Az * *Bz;
          ProductProof::prove(
            &pk.sumcheck_gens.ck_1,
            &mut transcript,
            Az,
            &blind_Az,
            Bz,
            &blind_Bz,
            &prod,
            &blind_prod_Az_Bz,
          )
        }.unwrap();

        let blind_Cz = <E as Engine>::Scalar::random(&mut OsRng);
        let comm_Cz = E::CE::zkcommit(ck, &[*Cz], &blind_Cz).compress();

        let blind_W = <E as Engine>::Scalar::random(&mut OsRng);
        let comm_W = E::CE::zkcommit(ck, &[*W], &blind_W).compress();

        let blind_E = <E as Engine>::Scalar::random(&mut OsRng);
        let comm_E = E::CE::zkcommit(ck, &[*E], &blind_E).compress();
        ([comm_Az, comm_Bz, comm_Cz, comm_W, comm_E], (proof_prod_eval_Az_eval_Bz, comm_prod_eval_Az_eval_Bz), [blind_Az, blind_Bz, blind_prod_Az_Bz])
      }).collect::<Vec<_>>().into_iter().unzip_n_vec();

      // [val_A, val_B, val_C, row, col, ts_row, ts_col]
      let evals_mem_preprocessed = zip_with!(
        (evals_mem_val_row_col.into_iter(), evals_mem_ts),
        |eval_mem_val_row_col, eval_mem_ts| {
          let [val_A, val_B, val_C, row, col] = eval_mem_val_row_col;
          let [ts_row, ts_col] = eval_mem_ts;
          [val_A, val_B, val_C, row, col, ts_row, ts_col]
        }
      )
      .collect::<Vec<_>>();



      let (comm_evals_mem_preprocessed, blind_row_col, prod_comm_L_row_L_col_val_A_B_C): (Vec<_>, Vec<_>, Vec<_>) =  evals_mem_preprocessed.iter().enumerate().map(|(i, [val_A, val_B, val_C, row, col, ts_row, ts_col])| {
        let [L_row, L_col] = evals_L_row_col[i];
        let L_row_L_col = L_row * L_col;

        let blind_val_A = <E as Engine>::Scalar::random(&mut OsRng);

        let blind_L_row_L_col = <E as Engine>::Scalar::random(&mut OsRng);

        let blind_prod_L_row_L_col_val_A = <E as Engine>::Scalar::random(&mut OsRng);
        let (proof_prod_L_row_L_col_val_A, _, comm_val_A, comm_prod_L_row_L_col_val_A): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {

          let prod = L_row_L_col * *val_A;
          ProductProof::prove(
            &pk.sumcheck_gens.ck_1,
            &mut transcript,
            &L_row_L_col,
            &blind_L_row_L_col,
            val_A,
            &blind_val_A,
            &prod,
            &blind_prod_L_row_L_col_val_A,
          )
        }.unwrap();

        let blind_val_B = <E as Engine>::Scalar::random(&mut OsRng);

        let blind_prod_L_row_L_col_val_B = <E as Engine>::Scalar::random(&mut OsRng);
        let (proof_prod_L_row_L_col_val_B, _, comm_val_B, comm_prod_L_row_L_col_val_B): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {

          let prod = L_row_L_col * *val_B;
          ProductProof::prove(
            &pk.sumcheck_gens.ck_1,
            &mut transcript,
            &L_row_L_col,
            &blind_L_row_L_col,
            val_B,
            &blind_val_B,
            &prod,
            &blind_prod_L_row_L_col_val_B,
          )
        }.unwrap();

        let blind_val_C = <E as Engine>::Scalar::random(&mut OsRng);

        let blind_prod_L_row_L_col_val_C = <E as Engine>::Scalar::random(&mut OsRng);
        let (proof_prod_L_row_L_col_val_C, _, comm_val_C, comm_prod_L_row_L_col_val_C): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {

          let prod = L_row_L_col * *val_C;
          ProductProof::prove(
            &pk.sumcheck_gens.ck_1,
            &mut transcript,
            &L_row_L_col,
            &blind_L_row_L_col,
            val_A,
            &blind_val_C,
            &prod,
            &blind_prod_L_row_L_col_val_C,
          )
        }.unwrap();

        let blind_row = <E as Engine>::Scalar::random(&mut OsRng);
        let comm_row = E::CE::zkcommit(ck, &[*row], &blind_row).compress();

        let blind_col = <E as Engine>::Scalar::random(&mut OsRng);
        let comm_col = E::CE::zkcommit(ck, &[*col], &blind_col).compress();

        
        let blind_ts_row = <E as Engine>::Scalar::random(&mut OsRng);
        let comm_ts_row = E::CE::zkcommit(ck, &[*ts_row], &blind_ts_row).compress();
        
        let blind_ts_col = <E as Engine>::Scalar::random(&mut OsRng);
        let comm_ts_col = E::CE::zkcommit(ck, &[*ts_col], &blind_ts_col).compress();

        ([comm_val_A, comm_val_B, comm_val_C, comm_row, comm_col, comm_ts_row, comm_ts_col], [blind_row, blind_col], [(proof_prod_L_row_L_col_val_A, comm_prod_L_row_L_col_val_A), (proof_prod_L_row_L_col_val_B, comm_prod_L_row_L_col_val_B), (proof_prod_L_row_L_col_val_C, comm_prod_L_row_L_col_val_C)])
      }).collect::<Vec<_>>().into_iter().unzip_n_vec();

      let num_vars = S.iter().map(|s| s.num_vars).collect::<Vec<_>>();
  
      // Truncated sumcheck randomness for each instance
      let rand_sc_i = num_rounds
        .iter()
        .map(|num_rounds| rand_sc[(num_rounds_max - num_rounds)..].to_vec())
        .collect::<Vec<_>>();
  
      let (comm_z_vec, z_vec, blinds_z): (Vec<_>, Vec<_>, Vec<_>) = zip_with!(
        (
          num_vars.iter(),
          rand_sc_i.iter(),
          U.iter(),
          evals_Az_Bz_Cz_W_E.iter().cloned()
        ),
        |num_vars,
         rand_sc,
         U,
         evals_Az_Bz_Cz_W_E| {
          let [_, _, _, W, _] = evals_Az_Bz_Cz_W_E;

          let num_rounds_i = rand_sc.len();
          let num_vars_log = num_vars.log_2();
  
          let (comm_z, Z, blind_z) = {
            // rand_sc was padded, so we now remove the padding
            let (factor, rand_sc_unpad) = {
              let l = num_rounds_i - (num_vars_log + 1);
  
              let (rand_sc_lo, rand_sc_hi) = rand_sc.split_at(l);
  
              let factor = rand_sc_lo
                .iter()
                .fold(E::Scalar::ONE, |acc, r_p| acc * (E::Scalar::ONE - r_p));
  
              (factor, rand_sc_hi)
            };
  
            let X = {
              // constant term
              let poly_X = std::iter::once(U.u).chain(U.X.iter().cloned()).collect();
              SparsePolynomial::new(num_vars_log, poly_X).evaluate(&rand_sc_unpad[1..])
            };
  
            // W was evaluated as if it was padded to logNi variables,
            // so we don't multiply it by (1-rand_sc_unpad[0])
          //   let comm_z = Commitment::<E>::decompress(&comm_W).unwrap() + <E as Engine>::CE::zkcommit(
          //     &EE::get_scalar_gen_pk(pk.pk_ee.clone()),
          //     &[X],
          //     &<E as Engine>::Scalar::ZERO,
          // ) * factor * rand_sc_unpad[0];

            let z = W + factor * rand_sc_unpad[0] * X;

            let blind_z = <E as Engine>::Scalar::random(&mut OsRng);

            let comm_z = <E as Engine>::CE::zkcommit(
              &EE::get_scalar_gen_pk(pk.pk_ee.clone()),
              &[z],
              &blind_z,
            )
            .compress();

            (comm_z, z, blind_z)
          };
  
          (comm_z, Z, blind_z)
        }
      )
      .collect::<Vec<_>>().into_iter().unzip_n_vec();

      let (prods_comms_w_plus_r_inv_row_blind_row, prods_comms_w_plus_r_inv_row_blind_col, prods_comms_w_plus_r_inv_row_L_row, prods_comms_w_plus_r_inv_row_L_col, prods_comms_t_plus_r_inv_col_Z): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) = zip_with!(
        iter,
        (
          evals_mem_preprocessed,
          blind_row_col,
          evals_mem_oracle,
          blind_evals_mem_oracle,
          evals_L_row_col,
          blinds_L_row_L_col,
          z_vec,
          blinds_z
        ),
        |evals_mem_preprocessed, blind_row_col, evals_mem_oracle, blind_evals_mem_oracle, evals_L_row_col, blinds_L_row_L_col, comm_z_vec, blinds_z| {
          let [_, _, _, row, col, _, _] = evals_mem_preprocessed;
          let [blind_row, blind_col] = blind_row_col;
          let [_, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col] = evals_mem_oracle;
          let [blind_w_plus_r_inv_row, blind_t_plus_r_inv_col, blind_w_plus_r_inv_col] = blind_evals_mem_oracle;
          let [L_row, L_col] = evals_L_row_col;
          let [blind_L_row, blind_L_col] = blinds_L_row_L_col;
          let Z = comm_z_vec;
          let blind_Z = blinds_z;

          let blind_prod_w_plus_r_inv_row_blind_row = <E as Engine>::Scalar::random(&mut OsRng);

          let (prod_w_plus_r_inv_row_blind_row, _, _, comm_prod_w_plus_r_inv_row_blind_row): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {

            let prod = *w_plus_r_inv_row * *row;
            ProductProof::prove(
              &pk.sumcheck_gens.ck_1,
              &mut transcript,
              w_plus_r_inv_row,
              blind_w_plus_r_inv_row,
              row,
              blind_row,
              &prod,
              &blind_prod_w_plus_r_inv_row_blind_row,
            )
          }.unwrap();
          
          let blind_prod_w_plus_r_inv_row_blind_col = <E as Engine>::Scalar::random(&mut OsRng);

          let (prod_w_plus_r_inv_row_blind_col, _, _, comm_prod_w_plus_r_inv_row_blind_col): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {

            let prod = *w_plus_r_inv_row * *col;
            ProductProof::prove(
              &pk.sumcheck_gens.ck_1,
              &mut transcript,
              w_plus_r_inv_col,
              blind_w_plus_r_inv_col,
              col,
              blind_col,
              &prod,
              &blind_prod_w_plus_r_inv_row_blind_col,
            )
          }.unwrap();

          let blind_prod_w_plus_r_inv_row_L_row = <E as Engine>::Scalar::random(&mut OsRng);

          let (prod_w_plus_r_inv_row_L_row, _, _, comm_prod_w_plus_r_inv_row_L_row): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {

            let prod = *w_plus_r_inv_row * *row;
            ProductProof::prove(
              &pk.sumcheck_gens.ck_1,
              &mut transcript,
              w_plus_r_inv_row,
              blind_w_plus_r_inv_row,
              L_row,
              blind_L_row,
              &prod,
              &blind_prod_w_plus_r_inv_row_L_row,
            )
          }.unwrap();
          
          let blind_prod_w_plus_r_inv_row_L_col = <E as Engine>::Scalar::random(&mut OsRng);

          let (prod_w_plus_r_inv_row_L_col, _, _, comm_prod_w_plus_r_inv_row_L_col): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {

            let prod = *w_plus_r_inv_row * *col;
            ProductProof::prove(
              &pk.sumcheck_gens.ck_1,
              &mut transcript,
              w_plus_r_inv_col,
              blind_w_plus_r_inv_col,
              L_col,
              blind_L_col,
              &prod,
              &blind_prod_w_plus_r_inv_row_L_col,
            )
          }.unwrap();

          let blind_prod_t_plus_r_inv_col_Z = <E as Engine>::Scalar::random(&mut OsRng);

          let (prod_t_plus_r_inv_col_Z, _, _, comm_prod_t_plus_r_inv_col_Z): (ProductProof<E>, CompressedCommitment<E>, CompressedCommitment<E>, CompressedCommitment<E>) = {

            let prod = *t_plus_r_inv_col * *Z;
            ProductProof::prove(
              &pk.sumcheck_gens.ck_1,
              &mut transcript,
              w_plus_r_inv_col,
              blind_t_plus_r_inv_col,
              Z,
              blind_Z,
              &prod,
              &blind_prod_t_plus_r_inv_col_Z,
            )
          }.unwrap();

          ((prod_w_plus_r_inv_row_blind_row, comm_prod_w_plus_r_inv_row_blind_row), (prod_w_plus_r_inv_row_blind_col, comm_prod_w_plus_r_inv_row_blind_col), (prod_w_plus_r_inv_row_L_row, comm_prod_w_plus_r_inv_row_L_row), (prod_w_plus_r_inv_row_L_col, comm_prod_w_plus_r_inv_row_L_col), (prod_t_plus_r_inv_col_Z, comm_prod_t_plus_r_inv_col_Z))
        }
      ).collect::<Vec<_>>().into_iter().unzip_n_vec();



      (
        evals_Az_Bz_Cz_W_E,
        evals_L_row_col,
        evals_mem_oracle,
        evals_mem_preprocessed,
        comm_evals_Az_Bz_Cz_W_E,
        prod_comm_Az_Bz,
        comm_evals_L_row_col,
        prod_comm_L_row_L_col,
        comm_evals_mem_oracle,
        comm_evals_mem_preprocessed,
        blind_row_col,
        prod_comm_L_row_L_col_val_A_B_C,
        comm_z_vec,
        prods_comms_w_plus_r_inv_row_blind_row,
        prods_comms_w_plus_r_inv_row_blind_col,
        prods_comms_w_plus_r_inv_row_L_row,
        prods_comms_w_plus_r_inv_row_L_col,
        prods_comms_t_plus_r_inv_col_Z
      )
    };



    let evals_vec = zip_with!(
      iter,
      (
        evals_Az_Bz_Cz_W_E,
        evals_L_row_col,
        evals_mem_oracle,
        evals_mem_preprocessed
      ),
      |Az_Bz_Cz_W_E, L_row_col, mem_oracles, mem_preprocessed| {
        chain![Az_Bz_Cz_W_E, L_row_col, mem_oracles, mem_preprocessed]
          .cloned()
          .collect::<Vec<_>>()
      }
    )
    .collect::<Vec<_>>();

    let comms_vec = zip_with!(
      iter,
      (
        comms_Az_Bz_Cz,
        comms_W_E,
        comms_L_row_col,
        comms_mem_oracles,
        pk.S_comm
      ),
      |Az_Bz_Cz, comms_W_E, L_row_col, mem_oracles, S_comm| {
        chain![
          Az_Bz_Cz,
          comms_W_E,
          L_row_col,
          mem_oracles,
          [
            &S_comm.comm_val_A,
            &S_comm.comm_val_B,
            &S_comm.comm_val_C,
            &S_comm.comm_row,
            &S_comm.comm_col,
            &S_comm.comm_ts_row,
            &S_comm.comm_ts_col,
          ]
        ]
      }
    )
    .flatten()
    .cloned()
    .collect::<Vec<_>>();

    // let w_vec = zip_with!(
    //   (
    //     polys_Az_Bz_Cz.into_iter(),
    //     polys_W.into_iter(),
    //     polys_E.into_iter(),
    //     polys_L_row_col.into_iter(),
    //     polys_mem_oracles.into_iter(),
    //     pk.S_repr.iter()
    //   ),
    //   |Az_Bz_Cz, W, E, L_row_col, mem_oracles, S_repr| {
    //     chain![
    //       Az_Bz_Cz,
    //       [W, E],
    //       L_row_col,
    //       mem_oracles,
    //       [
    //         S_repr.val_A.clone(),
    //         S_repr.val_B.clone(),
    //         S_repr.val_C.clone(),
    //         S_repr.row.clone(),
    //         S_repr.col.clone(),
    //         S_repr.ts_row.clone(),
    //         S_repr.ts_col.clone(),
    //       ]
    //     ]
    //   }
    // )
    // .flatten()
    // .map(|p| PolyEvalWitness::<E> { p })
    // .collect::<Vec<_>>();

    for evals in evals_vec.iter() {
      transcript.absorb(b"e", &evals.as_slice()); // comm_vec is already in the transcript
    }
    let evals_vec = evals_vec.into_iter().flatten().collect::<Vec<_>>();

    let c = transcript.squeeze(b"c")?;



    let tau = transcript.squeeze(b"t")?;
    let tau_coords = PowPolynomial::new(&tau, num_rounds_max).coordinates();

    // Compute eval_Mz = eval_Az_at_tau + c * eval_Bz_at_tau + c^2 * eval_Cz_at_tau
    let evals_Mz: Vec<_> = zip_with!(
      iter,
      (comms_Az_Bz_Cz, evals_Az_Bz_Cz_at_tau),
      |comm_Az_Bz_Cz, evals_Az_Bz_Cz_at_tau| {
        let u = PolyEvalInstance::<E>::batch(
          comm_Az_Bz_Cz.as_slice(),
          tau_coords.clone(),
          evals_Az_Bz_Cz_at_tau.as_slice(),
          &c,
        );
        u.e
      }
    )
    .collect();

  

    let num_claims_per_instance = 10;

    let claims = evals_Mz
    .iter()
    .flat_map(|&eval_Mz| {
      let mut claims = vec![E::Scalar::ZERO; num_claims_per_instance];
      claims[7] = eval_Mz;
      claims[8] = eval_Mz;
      claims.into_iter()
    })
    .collect::<Vec<_>>();


    let comm_claims =  claims.iter().map(|eval| {
        let blind_eval = <E as Engine>::Scalar::random(&mut OsRng);
        E::CE::zkcommit(ck, &[*eval], &blind_eval).compress()
    }).collect::<Vec<_>>();


    // // Compute number of variables for each polynomial
    // let num_vars_u = w_vec.iter().map(|w| w.p.len().log_2()).collect::<Vec<_>>();
    // let u_batch =
    //   PolyEvalInstance::<E>::batch_diff_size(&comms_vec, &evals_vec, &num_vars_u, rand_sc, c);
    // let w_batch =
    //   PolyEvalWitness::<E>::batch_diff_size(&w_vec.iter().by_ref().collect::<Vec<_>>(), c);

    // let eval_arg = EE::prove_batch(
    //   ck,
    //   &pk.pk_ee,
    //   &mut transcript,
    //   &u_batch.c,
    //   &w_batch.p,
    //   &u_batch.x,
    //   &u_batch.e,
    // )?;



    // TODO: need to think about it
    let poly_vec = vec![];

    let blind_poly_vec = vec![];
    let points = vec![];
    let blind_eval_vec = vec![];
    let comm_eval_vec = vec![];

    let eval_arg = EE::prove_batch(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &comms_vec,
      // w_vec.as_slice(),
      &poly_vec,
      &blind_poly_vec,
      &points,
      &evals_vec,
      &blind_eval_vec,
      &comm_eval_vec,
    )?;

    let comms_Az_Bz_Cz = comms_Az_Bz_Cz
      .into_iter()
      .map(|comms| comms.map(|comm| comm.compress()))
      .collect();
    let comms_L_row_col = comms_L_row_col
      .into_iter()
      .map(|comms| comms.map(|comm| comm.compress()))
      .collect();
    let comms_mem_oracles = comms_mem_oracles
      .into_iter()
      .map(|comms| comms.map(|comm| comm.compress()))
      .collect();

    Ok(Self {
      comms_Az_Bz_Cz,
      comms_L_row_col,
      comms_mem_oracles,
      comm_evals_Az_Bz_Cz_at_tau,
      sc,
      comm_evals_Az_Bz_Cz_W_E,
      prod_comm_Az_Bz,
      prod_comm_L_row_L_col,
      comm_evals_L_row_col,
      comm_evals_mem_oracle,
      comm_evals_mem_preprocessed,
      prod_comm_L_row_L_col_val_A_B_C,
      comm_claims,
      comm_z_vec,
      prods_comms_w_plus_r_inv_row_blind_row,
      prods_comms_w_plus_r_inv_row_blind_col,
      prods_comms_w_plus_r_inv_row_L_row,
      prods_comms_w_plus_r_inv_row_L_col,
      prods_comms_t_plus_r_inv_col_Z,
      eval_arg,
    })
  }

  fn verify(&self, vk: &Self::VerifierKey, U: &[RelaxedR1CSInstance<E>]) -> Result<(), NovaError> {
    let num_instances = U.len();
    let num_claims_per_instance = 10;

    // number of rounds of sum-check
    let num_rounds = vk.S_comm.iter().map(|s| s.N.log_2()).collect::<Vec<_>>();
    let num_rounds_max = *num_rounds.iter().max().unwrap();

    let mut transcript = E::TE::new(b"BatchedRelaxedR1CSSNARK");

    transcript.absorb(b"vk", &vk.digest());
    if num_instances > 1 {
      let num_instances_field = E::Scalar::from(num_instances as u64);
      transcript.absorb(b"n", &num_instances_field);
    }
    transcript.absorb(b"U", &U);

    // Decompress commitments
    let comms_Az_Bz_Cz = self
      .comms_Az_Bz_Cz
      .iter()
      .map(|comms| {
        comms
          .iter()
          .map(Commitment::<E>::decompress)
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;

    let comms_L_row_col = self
      .comms_L_row_col
      .iter()
      .map(|comms| {
        comms
          .iter()
          .map(Commitment::<E>::decompress)
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;

      // let comm_z_vec = self
      // .comm_z_vec
      // .iter()
      // .map(|comm| {
      //   Commitment::<E>::decompress(comm).unwrap()
      // })
      // .collect::<Vec<_>>();

    let comms_mem_oracles = self
      .comms_mem_oracles
      .iter()
      .map(|comms| {
        comms
          .iter()
          .map(Commitment::<E>::decompress)
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Add commitments [Az, Bz, Cz] to the transcript
    comms_Az_Bz_Cz
      .iter()
      .for_each(|comms| transcript.absorb(b"c", &comms.as_slice()));


    // // absorb the claimed evaluations into the transcript
    // self.evals_Az_Bz_Cz_at_tau.iter().for_each(|evals| {
    //   // transcript.absorb(b"e", &evals.as_slice());
    // });

    let _comms_evals_Az_Bz_Cz_at_tau = self
    .comm_evals_Az_Bz_Cz_at_tau
    .iter()
    .map(|comms| {
      comms
        .iter()
        .map(Commitment::<E>::decompress)
        .collect::<Result<Vec<_>, _>>()
    })
    .collect::<Result<Vec<_>, _>>()?;

    
    let comm_evals_Az_Bz_Cz_W_E = self
      .comm_evals_Az_Bz_Cz_W_E
      .iter()
      .map(|comms| {
        let arr: [_; 5] = comms
          .iter()
          .map(Commitment::<E>::decompress)
          .collect::<Result<Vec<_>, _>>().unwrap().try_into().unwrap();
        arr
      })
      .collect::<Vec<_>>();

    let comm_evals_mem_preprocessed = self
      .comm_evals_mem_preprocessed
      .iter()
      .map(|comms| {
        comms
          .iter()
          .map(Commitment::<E>::decompress)
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;

    let comm_evals_L_row_col = self
      .comm_evals_L_row_col
      .iter()
      .map(|comms| {
        let arr: [_; 2] = comms
          .iter()
          .map(Commitment::<E>::decompress)
          .collect::<Result<Vec<_>, _>>().unwrap().try_into().unwrap();
        arr
      })
      .collect::<Vec<_>>();

    let comm_evals_mem_oracle = self
      .comm_evals_mem_oracle
      .iter()
      .map(|comms| {
        let arr: [_; 4] = comms
          .iter()
          .map(Commitment::<E>::decompress)
          .collect::<Result<Vec<_>, _>>().unwrap().try_into().unwrap();
        arr
      })
      .collect::<Vec<_>>();
 
    // let comms_w_plus_r_inv_row_blind_row = self.prods_comms_w_plus_r_inv_row_blind_row.iter().map(|(prod, comm)| {
    //   self.prod.verify(
    //       &vk.sumcheck_gens.ck_1,
    //       &mut transcript,
    //       &self.comm_eval_w_plus_r_inv_col,
    //       &self.comm_eval_col,
    //       &comm,
    //   ).unwrap();
    //   Commitment::<E>::decompress(&comm).unwrap()
    // });

    let comms_w_plus_r_inv_row_row = zip_with!(iter, (self.prods_comms_w_plus_r_inv_row_blind_row, self.comm_evals_mem_oracle, self.comm_evals_mem_preprocessed), |prod_comm, comm_evals_mem_oracle, comm_evals_mem_preprocessed| {
      let [_, comm_w_plus_r_inv_row, _, _] = comm_evals_mem_oracle;
      let [_, _, _, comm_row, _, _, _] = comm_evals_mem_preprocessed;
      let (prod, comm) = prod_comm;
      prod.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        &comm_w_plus_r_inv_row,
        &comm_row,
        &comm,
    )?;
    Ok(Commitment::<E>::decompress(&comm)?)
    })
    .collect::<Result<Vec<_>, NovaError>>()?;

    // [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col]      let [_, _, _, comm_w_plus_r_inv_col] = comm_evals_mem_oracle;


    let comms_w_plus_r_inv_col_col = zip_with!(iter, (self.prods_comms_w_plus_r_inv_row_blind_row, self.comm_evals_mem_oracle, self.comm_evals_mem_preprocessed), |prod_comm, comm_evals_mem_oracle, comm_evals_mem_preprocessed| {
      let [_, _, _, comm_w_plus_r_inv_col] = comm_evals_mem_oracle;
      let [_, _, _, _, comm_col, _, _] = comm_evals_mem_preprocessed;
      let (prod, comm) = prod_comm;
      prod.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        &comm_w_plus_r_inv_col ,
        &comm_col,
        &comm,
    )?;
    Ok(Commitment::<E>::decompress(&comm)?)
    })
    .collect::<Result<Vec<_>, NovaError>>()?;


    let comms_w_plus_r_inv_row_L_row = zip_with!(iter, (self.prods_comms_w_plus_r_inv_row_blind_row, self.comm_evals_mem_oracle, self.comm_evals_L_row_col), |prod_comm, comm_evals_mem_oracle, comm_evals_L_row_col| {
      let [_, comm_w_plus_r_inv_row, _, _] = comm_evals_mem_oracle;
      let [comm_L_row, _] = comm_evals_L_row_col;
      let (prod, comm) = prod_comm;
      prod.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        &comm_w_plus_r_inv_row,
        &comm_L_row,
        &comm,
    )?;
    Ok(Commitment::<E>::decompress(&comm)?)
    })
    .collect::<Result<Vec<_>, NovaError>>()?;


    let comms_w_plus_r_inv_col_L_col = zip_with!(iter, (self.prods_comms_w_plus_r_inv_row_blind_row, self.comm_evals_mem_oracle, self.comm_evals_L_row_col), |prod_comm, comm_evals_mem_oracle, comm_evals_L_row_col| {
      let [_, _, _, comm_w_plus_r_inv_col] = comm_evals_mem_oracle;
      let [_, comm_L_col] = comm_evals_L_row_col;
      let (prod, comm) = prod_comm;
      prod.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        &comm_w_plus_r_inv_col ,
        &comm_L_col,
        &comm,
    )?;
    Ok(Commitment::<E>::decompress(&comm)?)
    })
    .collect::<Result<Vec<_>, NovaError>>()?;

    let comms_t_plus_r_inv_col_z = zip_with!(iter, (self.prods_comms_w_plus_r_inv_row_blind_row, self.comm_evals_mem_oracle, self.comm_z_vec), |prod_comm, comm_evals_mem_oracle, comm_z_vec| {
      let [_, _, comm_t_plus_r_inv_col, _] = comm_evals_mem_oracle;
      let comm_z = comm_z_vec;
      let (prod, comm) = prod_comm;
      prod.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        comm_t_plus_r_inv_col,
        comm_z,
        comm,
    )?;
    Ok(Commitment::<E>::decompress(&comm)?)
    })
    .collect::<Result<Vec<_>, NovaError>>()?;

    let comms_Az_Bz = zip_with!(iter, (self.prod_comm_Az_Bz, self.comm_evals_Az_Bz_Cz_W_E), |prod_comm, comm_evals_Az_Bz_Cz_W_E| {
      let [comm_Az, comm_Bz, _, _, _] = comm_evals_Az_Bz_Cz_W_E;
      let (prod, comm) = prod_comm;
      prod.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        comm_Az,
        comm_Bz,
        comm,
    )?;
    Ok(Commitment::<E>::decompress(&comm)?)
    })
    .collect::<Result<Vec<_>, NovaError>>()?;

  
    let comms_L_row_L_col = zip_with!((self.prod_comm_L_row_L_col.iter().cloned(), self.comm_evals_L_row_col.iter().cloned()), |prod_comm, comm_evals_L_row_col| {
      let [comm_L_row, comm_L_col] = comm_evals_L_row_col;
      let (prod, comm) = prod_comm;
      prod.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        &comm_L_row,
        &comm_L_col,
        &comm,
    )?;
    Ok(comm)
    })
    .collect::<Result<Vec<_>, NovaError>>()?;


    // let comms_t_plus_r_inv_col_Z = zip_with!(iter, (self.prods_comms_t_plus_r_inv_col_Z, self.comm_evals_mem_oracle, self.comm_z_vec), |prod_comm, comm_evals_mem_oracle, comm_z_vec| {
    //   let [_, _, comm_t_plus_r_inv_col, _] = comm_evals_mem_oracle;
    //   let comm_z = comm_z_vec;
    //   let (prod, comm) = prod_comm;
    //   prod.verify(
    //     &vk.sumcheck_gens.ck_1,
    //     &mut transcript,
    //     comm_t_plus_r_inv_col,
    //     comm_z,
    //     comm,
    // )?;
    // Ok(Commitment::<E>::decompress(&comm)?)
    // })
    // .collect::<Result<Vec<_>, NovaError>>()?;


    let comms_L_row_L_col_val_A_B_C = zip_with!(iter, (self.prod_comm_L_row_L_col_val_A_B_C, comms_L_row_L_col, self.comm_evals_mem_preprocessed), |prod_comm, comms_L_row_L_col, comm_evals_mem_preprocessed| {
      let comm_L_row_L_col = comms_L_row_L_col;
      let [comm_val_A, comm_val_B, comm_val_C, _, _, _, _] = comm_evals_mem_preprocessed;
      let [(prod_A, comm_A), (prod_B, comm_B), (prod_C, comm_C)] = prod_comm;
      prod_A.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        comm_L_row_L_col,
        comm_val_A,
        comm_A,
    )?;
      prod_B.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        comm_L_row_L_col,
        comm_val_B,
        comm_B,
    )?;
      prod_C.verify(
        &vk.sumcheck_gens.ck_1,
        &mut transcript,
        comm_L_row_L_col,
        comm_val_C,
        comm_C,
    )?;
    Ok([Commitment::<E>::decompress(&comm_A)?, Commitment::<E>::decompress(&comm_B)?, Commitment::<E>::decompress(&comm_C)?])
    })
    .collect::<Result<Vec<_>, NovaError>>()?;

    // absorb commitments to L_row and L_col in the transcript
    for comms in comms_L_row_col.iter() {
      transcript.absorb(b"e", &comms.as_slice());
    }

    // Batch at tau for each instance
    let c = transcript.squeeze(b"c")?;

    // // Compute eval_Mz = eval_Az_at_tau + c * eval_Bz_at_tau + c^2 * eval_Cz_at_tau
    // let evals_Mz: Vec<_> = zip_with!(
    //   iter,
    //   (comms_Az_Bz_Cz, self.comm_evals_Az_Bz_Cz_at_tau),
    //   |comm_Az_Bz_Cz, comm_evals_Az_Bz_Cz_at_tau| {
    //     let u = PolyEvalInstance::<E>::batch(
    //       comm_Az_Bz_Cz.as_slice(),
    //       tau_coords.clone(),
    //       comm_evals_Az_Bz_Cz_at_tau.as_slice(),
    //       &c,
    //     );
    //     u.e
    //   }
    // )
    // .collect();
    // let comm_claims = self
    //   .comm_claims
    //   .iter()
    //   .map(|comms| {
    //     Commitment::<E>::decompress(comms)
    //   })
    //   .collect::<Result<Vec<_>, _>>()?;

    let gamma = transcript.squeeze(b"g")?;
    let r = transcript.squeeze(b"r")?;

    for comms in comms_mem_oracles.iter() {
      transcript.absorb(b"l", &comms.as_slice());
    }

    let rho = transcript.squeeze(b"r")?;

    let s = transcript.squeeze(b"r")?;
    let s_powers = powers(&s, num_instances * num_claims_per_instance);

    let (claim_sc_final, rand_sc) = {
      // // Gather all claims into a single vector
      // let claims = comm_evals_Mz
      //   .iter()
      //   .flat_map(|&eval_Mz| {
      //     let mut claims = vec![E::Scalar::ZERO; num_claims_per_instance];
      //     claims[7] = eval_Mz;
      //     claims[8] = eval_Mz;
      //     claims.into_iter()
      //   })
      //   .collect::<Vec<_>>();

      // Number of rounds for each claim
      let num_rounds_by_claim = num_rounds
        .iter()
        .flat_map(|num_rounds_i| vec![*num_rounds_i; num_claims_per_instance].into_iter())
        .collect::<Vec<_>>();

      self
        .sc
        .verify_batch(&self.comm_claims, &num_rounds_by_claim, &s_powers, 3, &vk.sumcheck_gens.ck_1, &vk.sumcheck_gens.ck_4, &mut transcript)?
    };

    let tau = transcript.squeeze(b"t")?;
    // let tau_coords = PowPolynomial::new(&tau, num_rounds_max).coordinates();

    // Truncated sumcheck randomness for each instance
    let rand_sc_i = num_rounds
      .iter()
      .map(|num_rounds| rand_sc[(num_rounds_max - num_rounds)..].to_vec())
      .collect::<Vec<_>>();

    let claim_zero =
    <E as Engine>::CE::zkcommit(&vk.sumcheck_gens.ck_1, &[<E as Engine>::Scalar::ZERO], &<E as Engine>::Scalar::ZERO);

    let claim_sc_final_expected = zip_with!(
      (
        vk.num_vars.iter(),
        rand_sc_i.iter(),
        U.iter(),
        comm_evals_Az_Bz_Cz_W_E.iter().cloned(),
        comm_evals_L_row_col.iter().cloned(),
        comm_evals_mem_oracle.iter().cloned(),
        comm_evals_mem_preprocessed.iter().cloned(),
        comms_w_plus_r_inv_row_row.iter().cloned(),
        comms_w_plus_r_inv_row_L_row.iter().cloned(),
        comms_t_plus_r_inv_col_z.iter().cloned(),
        comms_w_plus_r_inv_col_col.iter().cloned(),
        comms_w_plus_r_inv_col_L_col.iter().cloned(),
        comms_Az_Bz.iter().cloned(),
        comms_L_row_L_col_val_A_B_C.iter().cloned()
      ),
      |num_vars,
       rand_sc,
       U,
       comm_evals_Az_Bz_Cz_W_E,
       comm_evals_L_row_col,
       comm_eval_mem_oracle,
       comm_eval_mem_preprocessed,
       comm_w_plus_r_inv_row_row,
       comm_w_plus_r_inv_row_L_row,
       comm_t_plus_r_inv_col_z,
       comm_w_plus_r_inv_col_col,
       comm_w_plus_r_inv_col_L_col,
       comm_Az_Bz,
       comms_L_row_L_col_val_A_B_C| {
        let [Az, Bz, Cz, W, E] = comm_evals_Az_Bz_Cz_W_E;
        let [_, _] = comm_evals_L_row_col;
        let [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col] =
          comm_eval_mem_oracle;
        let [_, _, _, _, _, ts_row, ts_col] = comm_eval_mem_preprocessed.try_into().unwrap();
        let [comm_L_row_L_col_val_A, comm_L_row_L_col_val_B, comm_L_row_L_col_val_C] = comms_L_row_L_col_val_A_B_C.try_into().unwrap();


        let num_rounds_i = rand_sc.len();
        let num_vars_log = num_vars.log_2();

        let eq_rho = PowPolynomial::new(&rho, num_rounds_i).evaluate(rand_sc);

        let (eq_tau, eq_masked_tau) = {
          let eq_tau: EqPolynomial<_> = PowPolynomial::new(&tau, num_rounds_i).into();

          let eq_tau_at_rand = eq_tau.evaluate(rand_sc);
          let eq_masked_tau = MaskedEqPolynomial::new(&eq_tau, num_vars_log).evaluate(rand_sc);

          (eq_tau_at_rand, eq_masked_tau)
        };

        // Evaluate identity polynomial
        let id = IdentityPolynomial::new(num_rounds_i).evaluate(rand_sc);

        // let Z = {
        //   // rand_sc was padded, so we now remove the padding
        //   let (factor, rand_sc_unpad) = {
        //     let l = num_rounds_i - (num_vars_log + 1);

        //     let (rand_sc_lo, rand_sc_hi) = rand_sc.split_at(l);

        //     let factor = rand_sc_lo
        //       .iter()
        //       .fold(E::Scalar::ONE, |acc, r_p| acc * (E::Scalar::ONE - r_p));

        //     (factor, rand_sc_hi)
        //   };

        //   let X = {
        //     // constant term
        //     let poly_X = std::iter::once(U.u).chain(U.X.iter().cloned()).collect();
        //     SparsePolynomial::new(num_vars_log, poly_X).evaluate(&rand_sc_unpad[1..])
        //   };

        //   // W was evaluated as if it was padded to logNi variables,
        //   // so we don't multiply it by (1-rand_sc_unpad[0])
        //   W + <E as Engine>::CE::zkcommit(
        //     &EE::get_scalar_gen_vk(vk.vk_ee.clone()),
        //     &[X],
        //     &<E as Engine>::Scalar::ZERO,
        // ) * factor * rand_sc_unpad[0]
        // };

        let t_plus_row = {
          let addr_row = id;
          let val_row = eq_tau;
          let t = addr_row + gamma * val_row;
          t
        };

        // let w_plus_row = {
        //   let addr_row = row;
        //   let val_row = L_row;
        //   let w = addr_row + gamma * val_row;
        //   w
        // };

        // let t_plus_col = {
        //   let addr_col = id;
        //   let val_col = Z;
        //   let t = addr_col + gamma * val_col;
        //   t
        // };

        // let w_plus_col = {
        //   let addr_col = col;
        //   let val_col = L_col;
        //   let w = addr_col + gamma * val_col;
        //   w
        // };

        let claim_one =
        <E as Engine>::CE::zkcommit(&vk.sumcheck_gens.ck_1, &[<E as Engine>::Scalar::ONE], &<E as Engine>::Scalar::ZERO);
  
  
        let claims_mem = [
          t_plus_r_inv_row - w_plus_r_inv_row,
          t_plus_r_inv_col - w_plus_r_inv_col,
          (t_plus_r_inv_row * t_plus_row + t_plus_r_inv_row * r - ts_row) * eq_rho,
          (comm_w_plus_r_inv_row_row + comm_w_plus_r_inv_row_L_row * gamma + w_plus_r_inv_row * r - claim_one) * eq_rho,
          (t_plus_r_inv_col * id + comm_t_plus_r_inv_col_z * gamma + t_plus_r_inv_col * r - ts_col) * eq_rho,
          (comm_w_plus_r_inv_col_col + comm_w_plus_r_inv_col_L_col * gamma + w_plus_r_inv_col * r - claim_one) * eq_rho,
        ];

        let claims_outer = [
          (comm_Az_Bz - Cz * U.u - E) * eq_tau,
          (Az + Bz * c + Cz * c * c) * eq_tau,
        ];
        let claims_inner = [comm_L_row_L_col_val_A + comm_L_row_L_col_val_B * c + comm_L_row_L_col_val_C * c * c];

        let claims_witness = [W * eq_masked_tau];

        chain![claims_mem, claims_outer, claims_inner, claims_witness]
      }
    )
    .flatten()
    .zip_eq(s_powers)
    .fold(claim_zero, |acc, (claim, s)| acc + claim * s);

    if claim_sc_final_expected != Commitment::<E>::decompress(&claim_sc_final)? {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // let evals_vec = zip_with!(
    //   iter,
    //   (
    //     self.evals_Az_Bz_Cz_W_E,
    //     self.evals_L_row_col,
    //     self.evals_mem_oracle,
    //     self.evals_mem_preprocessed
    //   ),
    //   |Az_Bz_Cz_W_E, L_row_col, mem_oracles, mem_preprocessed| {
    //     chain![Az_Bz_Cz_W_E, L_row_col, mem_oracles, mem_preprocessed]
    //       .cloned()
    //       .collect::<Vec<_>>()
    //   }
    // )
    // .collect::<Vec<_>>();

    // // Add all Sumcheck evaluations to the transcript
    // for evals in evals_vec.iter() {
    //   transcript.absorb(b"e", &evals.as_slice()); // comm_vec is already in the transcript
    // }




    let _c = transcript.squeeze(b"c")?;

    // // Compute batched polynomial evaluation instance at rand_sc
    // let u = {
    //   let num_evals = evals_vec[0].len();

    //   let evals_vec = evals_vec.into_iter().flatten().collect::<Vec<_>>();

    //   let num_vars = num_rounds
    //     .iter()
    //     .flat_map(|num_rounds| vec![*num_rounds; num_evals].into_iter())
    //     .collect::<Vec<_>>();

    //   let comms_vec = zip_with!(
    //     (
    //       comms_Az_Bz_Cz.into_iter(),
    //       U.iter(),
    //       comms_L_row_col.into_iter(),
    //       comms_mem_oracles.into_iter(),
    //       vk.S_comm.iter()
    //     ),
    //     |Az_Bz_Cz, U, L_row_col, mem_oracles, S_comm| {
    //       chain![
    //         Az_Bz_Cz,
    //         [U.comm_W, U.comm_E],
    //         L_row_col,
    //         mem_oracles,
    //         [
    //           S_comm.comm_val_A,
    //           S_comm.comm_val_B,
    //           S_comm.comm_val_C,
    //           S_comm.comm_row,
    //           S_comm.comm_col,
    //           S_comm.comm_ts_row,
    //           S_comm.comm_ts_col,
    //         ]
    //       ]
    //     }
    //   )
    //   .flatten()
    //   .collect::<Vec<_>>();

    //   PolyEvalInstance::<E>::batch_diff_size(&comms_vec, &evals_vec, &num_vars, rand_sc, c)
    // };

    // verify
    // EE::verify_batch(&vk.vk_ee, &mut transcript, &u.c, &u.x, &u.e, &self.eval_arg)?;

    let comm_vec = vec![];
    let points = vec![];

    EE::verify_batch(
      &vk.vk_ee,
      &mut transcript,
      &comm_vec,
      &points,
      &self.eval_arg,
    )?;

    Ok(())
  }
}

impl<E: Engine + Serialize + for<'de> Deserialize<'de>, EE: EvaluationEngineTrait<E>> BatchedRelaxedR1CSSNARK<E, EE> 
where 
    EE: EvaluationEngineTrait<E>,
    <E as Engine>::CE: ZKCommitmentEngineTrait<E>,
    <E as Engine>::GE: DlogGroup<ScalarExt = E::Scalar>,
    // E::CE: CommitmentEngineTrait<E>,
    E::CE: CommitmentEngineTrait<E, Commitment = zk_pedersen::Commitment<E>, CommitmentKey = zk_pedersen::CommitmentKey<E>>,
    <E::CE as CommitmentEngineTrait<E>>::Commitment: Add<Output = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment>, 
    <E::CE as CommitmentEngineTrait<E>>::Commitment: Sub<Output = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment>,
    <<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as Sub>::Output as Mul<<E as Engine>::Scalar>>::Output: Add<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment> 
{
  /// Runs the batched Sumcheck protocol for the claims of multiple instance of possibly different sizes.
  ///
  /// # Details
  ///
  /// In order to avoid padding all polynomials to the same maximum size, we adopt the following strategy.
  ///
  /// Let n be the number of variables for the largest instance,
  /// and let m be the number of variables for a shorter one.
  /// Let P(X_{0},...,X_{m-1}) be one of the MLEs of the short instance, which has been committed to
  /// by taking the MSM of its evaluations with the first 2^m basis points of the commitment key.
  ///
  /// This Sumcheck prover will interpret it as the polynomial
  ///   P'(X_{0},...,X_{n-1}) = P(X_{n-m},...,X_{n-1}),
  /// whose MLE evaluations over {0,1}^m is equal to 2^{n-m} repetitions of the evaluations of P.
  ///
  /// In order to account for these "imagined" repetitions, the initial claims for this short instances
  /// are scaled by 2^{n-m}.
  ///
  /// For the first n-m rounds, the univariate polynomials relating to this shorter claim will be constant,
  /// and equal to the initial claims, scaled by 2^{n-m-i}, where i is the round number.
  /// By definition, P' does not depend on X_i, so binding P' to r_i has no effect on the evaluations.
  /// The Sumcheck prover will then interpret the polynomial P' as having half as many repetitions
  /// in the next round.
  ///
  /// When we get to round n-m, the Sumcheck proceeds as usual since the polynomials are the expected size
  /// for the round.
  ///
  /// Note that at the end of the protocol, the prover returns the evaluation
  ///   u' = P'(r_{0},...,r_{n-1}) = P(r_{n-m},...,r_{n-1})
  /// However, the polynomial we actually committed to over {0,1}^n is
  ///   P''(X_{0},...,X_{n-1}) = L_0(X_{0},...,X_{n-m-1}) * P(X_{n-m},...,X_{n-1})
  /// The SNARK prover/verifier will need to rescale the evaluation by the first Lagrange polynomial
  ///   u'' = L_0(r_{0},...,r_{n-m-1}) * u'
  /// in order batch all evaluations with a single PCS call.
  fn prove_helper<T1, T2, T3, T4>(
    ck: &CommitmentKey<E>,
    num_rounds: usize,
    mut mem: Vec<T1>,
    mut outer: Vec<T2>,
    mut inner: Vec<T3>,
    mut witness: Vec<T4>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      ZKSumcheckProof<E>,
      Vec<E::Scalar>,
      Vec<Vec<Vec<E::Scalar>>>,
      Vec<Vec<Vec<E::Scalar>>>,
      Vec<Vec<Vec<E::Scalar>>>,
      Vec<Vec<Vec<E::Scalar>>>,
    ),
    NovaError,
  >
  where
    T1: ZKSumcheckEngine<E>,
    T2: ZKSumcheckEngine<E>,
    T3: ZKSumcheckEngine<E>,
    T4: ZKSumcheckEngine<E>,
  {
    // sanity checks
    let num_instances = mem.len();
    assert_eq!(outer.len(), num_instances);
    assert_eq!(inner.len(), num_instances);
    assert_eq!(witness.len(), num_instances);

    for inst in mem.iter_mut() {
      assert!(inst.size().is_power_of_two());
    }
    for inst in outer.iter() {
      assert!(inst.size().is_power_of_two());
    }
    for inst in inner.iter() {
      assert!(inst.size().is_power_of_two());
    }
    for inst in witness.iter() {
      assert!(inst.size().is_power_of_two());
    }

    let degree = mem[0].degree();
    assert!(mem.iter().all(|inst| inst.degree() == degree));
    assert!(outer.iter().all(|inst| inst.degree() == degree));
    assert!(inner.iter().all(|inst| inst.degree() == degree));
    assert!(witness.iter().all(|inst| inst.degree() == degree));

    // Collect all claims from the instances. If the instances is defined over `m` variables,
    // which is less that the total number of rounds `n`,
    // the individual claims σ are scaled by 2^{n-m}.
    let claims = zip_with!(
      iter,
      (mem, outer, inner, witness),
      |mem, outer, inner, witness| {
        Self::scaled_claims(mem, num_rounds)
          .into_iter()
          .chain(Self::scaled_claims(outer, num_rounds))
          .chain(Self::scaled_claims(inner, num_rounds))
          .chain(Self::scaled_claims(witness, num_rounds))
      }
    )
    .flatten()
    .collect::<Vec<E::Scalar>>();

    // Sample a challenge for the random linear combination of all scaled claims
    let s = transcript.squeeze(b"r")?;
    let coeffs = powers(&s, claims.len());

    // At the start of each round, the running claim is equal to the random linear combination
    // of the Sumcheck claims, evaluated over the bound polynomials.
    // Initially, it is equal to the random linear combination of the scaled input claims.
    let mut running_claim = zip_with!(iter, (claims, coeffs), |c_1, c_2| *c_1 * c_2).sum();

    // Keep track of the verifier challenges r, and the univariate polynomials sent by the prover
    // in each round
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();

    let mut comm_polys = Vec::new();
    let mut comm_evals = Vec::new();
    let proofs = Vec::new();

    let (blinds_poly, blinds_evals) = {
      (
          (0..num_rounds)
              .map(|_i| <E as Engine>::Scalar::random(&mut OsRng))
              .collect::<Vec<<E as Engine>::Scalar>>(),
          (0..num_rounds)
              .map(|_i| <E as Engine>::Scalar::random(&mut OsRng))
              .collect::<Vec<<E as Engine>::Scalar>>(),
      )
    };

    for i in 0..num_rounds {
      // At the start of round i, there input polynomials are defined over at most n-i variables.
      let remaining_variables = num_rounds - i;

      // For each claim j, compute the evaluations of its univariate polynomial S_j(X_i)
      // at X = 0, 2, 3. The polynomial is such that S_{j-1}(r_{j-1}) = S_j(0) + S_j(1).
      // If the number of variable m of the claim is m < n-i, then the polynomial is
      // constants and equal to the initial claim σ_j scaled by 2^{n-m-i-1}.
      let evals = zip_with!(
        par_iter,
        (mem, outer, inner, witness),
        |mem, outer, inner, witness| {
          let ((evals_mem, evals_outer), (evals_inner, evals_witness)) = rayon::join(
            || {
              rayon::join(
                || Self::get_evals(mem, remaining_variables),
                || Self::get_evals(outer, remaining_variables),
              )
            },
            || {
              rayon::join(
                || Self::get_evals(inner, remaining_variables),
                || Self::get_evals(witness, remaining_variables),
              )
            },
          );
          evals_mem
            .into_par_iter()
            .chain(evals_outer.into_par_iter())
            .chain(evals_inner.into_par_iter())
            .chain(evals_witness.into_par_iter())
        }
      )
      .flatten()
      .collect::<Vec<_>>();

      assert_eq!(evals.len(), claims.len());

      // Random linear combination of the univariate evaluations at X_i = 0, 2, 3
      let evals_combined_0 = (0..evals.len()).map(|i| evals[i][0] * coeffs[i]).sum();
      let evals_combined_2 = (0..evals.len()).map(|i| evals[i][1] * coeffs[i]).sum();
      let evals_combined_3 = (0..evals.len()).map(|i| evals[i][2] * coeffs[i]).sum();

      let evals = vec![
        evals_combined_0,
        running_claim - evals_combined_0,
        evals_combined_2,
        evals_combined_3,
      ];
      // Coefficient representation of S(X_i)
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      // derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);

      // Bind the variable X_i of polynomials across all claims to r_i.
      // If the claim is defined over m variables and m < n-i, then
      // binding has no effect on the polynomial.
      zip_with_for_each!(
        par_iter_mut,
        (mem, outer, inner, witness),
        |mem, outer, inner, witness| {
          rayon::join(
            || {
              rayon::join(
                || Self::bind(mem, remaining_variables, &r_i),
                || Self::bind(outer, remaining_variables, &r_i),
              )
            },
            || {
              rayon::join(
                || Self::bind(inner, remaining_variables, &r_i),
                || Self::bind(witness, remaining_variables, &r_i),
              )
            },
          );
        }
      );

      running_claim = poly.evaluate(&r_i);
      let comm_e: crate::provider::zk_pedersen::CompressedCommitment<E> = <E as Engine>::CE::zkcommit(ck, &[running_claim], &blinds_evals[i]).compress();
      cubic_polys.push(poly.compress());

      let comm_poly: crate::provider::zk_pedersen::CompressedCommitment<E> = <E as Engine>::CE::zkcommit(ck, &poly.coeffs, &blinds_poly[i]).compress();
      comm_polys.push(comm_poly);
      comm_evals.push(comm_e);
    }

    // Collect evaluations at (r_{n-m}, ..., r_{n-1}) of polynomials over all claims,
    // where m is the initial number of variables the individual claims are defined over.
    let claims_outer = outer.into_iter().map(|inst| inst.final_claims()).collect();
    let claims_inner = inner.into_iter().map(|inst| inst.final_claims()).collect();
    let claims_mem = mem.into_iter().map(|inst| inst.final_claims()).collect();
    let claims_witness = witness
      .into_iter()
      .map(|inst| inst.final_claims())
      .collect();

    Ok((
      ZKSumcheckProof::new(comm_polys, comm_evals, proofs),
      r,
      claims_outer,
      claims_inner,
      claims_mem,
      claims_witness,
    ))
  }

  /// In round i, computes the evaluations at X_i = 0, 2, 3 of the univariate polynomials S(X_i)
  /// for each claim in the instance.
  /// Let `n` be the total number of Sumcheck rounds, and assume the instance is defined over `m` variables.
  /// We define `remaining_variables` as n-i.
  /// If m < n-i, then the polynomials in the instance are not defined over X_i, so the univariate
  /// polynomial is constant and equal to 2^{n-m-i-1}*σ, where σ is the initial claim.
  fn get_evals<T: ZKSumcheckEngine<E>>(inst: &T, remaining_variables: usize) -> Vec<Vec<E::Scalar>> {
    let num_instance_variables = inst.size().log_2(); // m
    if num_instance_variables < remaining_variables {
      let deg = inst.degree();

      // The evaluations at X_i = 0, 2, 3 are all equal to the scaled claim
      Self::scaled_claims(inst, remaining_variables - 1)
        .into_iter()
        .map(|scaled_claim| vec![scaled_claim; deg])
        .collect()
    } else {
      inst.evaluation_points()
    }
  }

  /// In round i after receiving challenge r_i, we partially evaluate all polynomials in the instance
  /// at X_i = r_i. If the instance is defined over m variables m which is less than n-i, then
  /// the polynomials do not depend on X_i, so binding them to r_i has no effect.  
  fn bind<T: ZKSumcheckEngine<E>>(inst: &mut T, remaining_variables: usize, r: &E::Scalar) {
    let num_instance_variables = inst.size().log_2(); // m
    if remaining_variables <= num_instance_variables {
      inst.bound(r)
    }
  }

  /// Given an instance defined over m variables, the sum over n = `remaining_variables` is equal
  /// to the initial claim scaled by 2^{n-m}, when m ≤ n.   
  fn scaled_claims<T: ZKSumcheckEngine<E>>(inst: &T, remaining_variables: usize) -> Vec<E::Scalar> {
    let num_instance_variables = inst.size().log_2(); // m
    let num_repetitions = 1 << (remaining_variables - num_instance_variables);
    let scaling = E::Scalar::from(num_repetitions as u64);
    inst
      .initial_claims()
      .iter()
      .map(|claim| scaling * claim)
      .collect()
  }
}

impl<E: Engine + Serialize + for<'de> Deserialize<'de>, EE: EvaluationEngineTrait<E>> RelaxedR1CSSNARKTrait<E>
  for BatchedRelaxedR1CSSNARK<E, EE> 
  where 
    E::CE: ZKCommitmentEngineTrait<E>, 
    <E as Engine>::GE: DlogGroup<ScalarExt = <E as Engine>::Scalar>,
    E::CE: CommitmentEngineTrait<E, Commitment = zk_pedersen::Commitment<E>, CommitmentKey = zk_pedersen::CommitmentKey<E>>,

{
  type ProverKey = ProverKey<E, EE>;

  type VerifierKey = VerifierKey<E, EE>;

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
