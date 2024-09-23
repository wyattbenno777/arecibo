//! batched zk pp snark
//!
//!

use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::{
    powers,
    math::Math,
    polys::{
      eq::EqPolynomial,
      power::PowPolynomial,
      univariate::{CompressedUniPoly, UniPoly},
      multilinear::{MultilinearPolynomial},
    },
    zksumcheck::{
      engine::{
        ZKSumcheckEngine,
        WitnessBoundSumcheck,
      },
      ZKSumcheckProof,
    },
    sumcheck::{
      engine::{
        SumcheckEngine,
        OuterSumcheckInstance, InnerSumcheckInstance, MemorySumcheckInstance
      },
      SumcheckProof
    },
    zkppsnark::{ZKR1CSShapeSparkCommitment, R1CSShapeSparkRepr},
    PolyEvalInstance, PolyEvalWitness
  },
  traits::{
    commitment::{CommitmentEngineTrait,  CommitmentTrait, Len, ZKCommitmentEngineTrait},
    zkevaluation::EvaluationEngineTrait,
    snark::{BatchedRelaxedR1CSSNARKTrait, DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine, TranscriptEngineTrait,
  },
  Commitment, CommitmentKey, CompressedCommitment,
  zip_with, zip_with_for_each
};
//use crate::spartan::nizk::ProductProof;
use core::ops::{Add, Sub, Mul};
use crate::spartan::zksnark::SumcheckGens;
use crate::provider::zk_pedersen;
use crate::provider::zk_ipa_pc;
use crate::provider::traits::DlogGroup;
use core::slice;
use ff::Field;
use itertools::{chain, Itertools as _};
use once_cell::sync::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rand_core::OsRng;
use crate::provider::zk_pedersen::CommitmentKeyExtTrait;
use unzip_n::unzip_n;
use std::fs;
use tracing::error;
use ff::PrimeField;
use crate::traits;
use std::iter;

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

/// The remove untrusted prover will be given all of the data that it
/// needs to verify the batched proofs from the smaller trusted prover.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "E: Engine")]
pub struct DataForUntrustedRemote<E: Engine> {
  polys_Az_Bz_Cz: Vec<[Vec<E::Scalar>; 3]>,
  ck: CommitmentKey<E>,
  //transcript: E::TE,
  pk_ee: CommitmentKey<E>,
  vk_digest: E::Scalar,
  polys_tau: Vec<Vec<E::Scalar>>, 
  coords_tau: Vec<Vec<E::Scalar>>,
  polys_L_row_col: Vec<[Vec<E::Scalar>; 2]>,
  polys_E: Vec<Vec<E::Scalar>>, 
  polys_Mz: Vec<Vec<E::Scalar>>,
  comms_Az_Bz_Cz: Vec<[Commitment<E>; 3]>,
  S_repr: Vec<R1CSShapeSparkRepr<E>>,
  S_comm: Vec<ZKR1CSShapeSparkCommitment<E>>,
  us: Vec<E::Scalar>, 
  polys_Z: Vec<Vec<E::Scalar>>,
  N_max: usize,
  Nis: Vec<usize>,
  num_rounds_sc: usize,
  rand_sc: Vec<E::Scalar>,
  blinded_witness_comms: Vec<Commitment<E>>,
  u_batch_witness: PolyEvalInstance<E>,
  w_batch_witness: PolyEvalWitness<E>,
  claims_witness: Vec<Vec<Vec<E::Scalar>>>,
  sumcheck_comm_polys: Vec<CompressedCommitment<E>>,
  sumcheck_comm_evals: Vec<CompressedCommitment<E>>,
  sumcheck_deltas: Vec<CompressedCommitment<E>>,
  sumcheck_betas: Vec<CompressedCommitment<E>>,
  sumcheck_zs: Vec<Vec<<E as Engine>::Scalar>>,
  sumcheck_z_deltas: Vec<<E as Engine>::Scalar>,
  sumcheck_z_betas: Vec<<E as Engine>::Scalar>,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  deserialize = "E: Deserialize<'de>"
))]
pub struct BatchedRelaxedR1CSSNARK<E: Engine + Serialize> {
  data: DataForUntrustedRemote<E>
}

#[allow(unused)]
fn serialize_field<T: serde::Serialize>(field_name: &str, field_value: &T) {
  match serde_json::to_string(field_value) {
      Ok(json_string) => {
          let file_name = format!("{}.json", field_name);
          if let Err(e) = fs::write(&file_name, json_string) {
              error!("Failed to write file {}: {}", file_name, e);
          } else {
              println!("Successfully wrote {}", file_name);
          }
      },
      Err(e) => {
          error!("Failed to serialize {}: {}", field_name, e);
      }
  }
}

#[allow(unused)]
fn serialize_data_for_remote<E: Engine>(data: &DataForUntrustedRemote<E>) 
where 
    E::Scalar: Serialize,
    Commitment<E>: Serialize,
    PolyEvalInstance<E>: Serialize,
    PolyEvalWitness<E>: Serialize,
{
    serialize_field("polys_Az_Bz_Cz", &data.polys_Az_Bz_Cz);
    serialize_field("ck", &data.ck);
    serialize_field("pk_ee", &data.pk_ee);
    serialize_field("vk_digest", &data.vk_digest);
    serialize_field("polys_tau", &data.polys_tau);
    serialize_field("coords_tau", &data.coords_tau);
    serialize_field("polys_L_row_col", &data.polys_L_row_col);
    serialize_field("polys_E", &data.polys_E);
    serialize_field("polys_Mz", &data.polys_Mz);
    serialize_field("comms_Az_Bz_Cz", &data.comms_Az_Bz_Cz);
    serialize_field("S_repr", &data.S_repr);
    serialize_field("S_comm", &data.S_comm);
    serialize_field("us", &data.us);
    serialize_field("polys_Z", &data.polys_Z);
    serialize_field("N_max", &data.N_max);
    serialize_field("Nis", &data.Nis);
    serialize_field("num_rounds_sc", &data.num_rounds_sc);
    serialize_field("rand_sc", &data.rand_sc);
    serialize_field("blinded_witness_comms", &data.blinded_witness_comms);
    serialize_field("u_batch_witness", &data.u_batch_witness);
    serialize_field("w_batch_witness", &data.w_batch_witness);
    serialize_field("claims_witness", &data.claims_witness);
    serialize_field("sumcheck_comm_polys", &data.sumcheck_comm_polys);
    serialize_field("sumcheck_comm_evals", &data.sumcheck_comm_evals);
    serialize_field("sumcheck_deltas", &data.sumcheck_deltas);
    serialize_field("sumcheck_betas", &data.sumcheck_betas);
    serialize_field("sumcheck_zs", &data.sumcheck_zs);
    serialize_field("sumcheck_z_deltas", &data.sumcheck_z_deltas);
    serialize_field("sumcheck_z_betas", &data.sumcheck_z_betas);

    println!("Serialization of DataForUntrustedRemote complete.");
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

    // Append public inputs to Wᵢ: Zᵢ = [Wᵢ, uᵢ, Xᵢ]
    let polys_Z = zip_with!(par_iter, (W, U, Nis), |W, U, Ni| {
      // poly_Z will be resized later, so we preallocate the correct capacity
      let mut poly_Z = Vec::with_capacity(*Ni);
      poly_Z.extend(W.W.iter().chain([&U.u]).chain(U.X.iter()));
      poly_Z
    })
    .collect::<Vec<Vec<E::Scalar>>>();

    // Move polys_W and polys_E, as well as U.u out of U
    let (_comms_W_E, us): (Vec<_>, Vec<_>) = U.iter().map(|U| ([U.comm_W, U.comm_E], U.u)).unzip();
    let (polys_W, polys_E): (Vec<_>, Vec<_>) = W.into_iter().map(|w| (w.W, w.E)).unzip();

    // Compute [Az, Bz, Cz]
    // Step 1, reducing the R1CS satisfiability to claims about polynomials.
    let mut polys_Az_Bz_Cz = zip_with!(par_iter, (polys_Z, S), |z, s| {
      let (Az, Bz, Cz) = s.multiply_vec(z)?;
      Ok([Az, Bz, Cz])
    })
    .collect::<Result<Vec<_>, NovaError>>()?;

    // Commit to [Az, Bz, Cz] and add to transcript
    let comms_Az_Bz_Cz = polys_Az_Bz_Cz
    .par_iter()
    .map(|[Az, Bz, Cz]| {
      let (comm_Az, (comm_Bz, comm_Cz)) = rayon::join(
        || E::CE::commit(ck, Az),
        || rayon::join(|| E::CE::commit(ck, Bz), || E::CE::commit(ck, Cz)),
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
    // This is sent to the untrusted prover for futher processing.
    polys_Az_Bz_Cz
    .par_iter_mut()
    .zip_eq(Nis.par_iter())
    .for_each(|(az_bz_cz, &Ni)| {
      az_bz_cz
        .par_iter_mut()
        .for_each(|mz| mz.resize(Ni, E::Scalar::ZERO))
    });


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

    let witness_sc_inst = zip_with!(par_iter, (polys_W, S), |poly_W, S| {
      WitnessBoundSumcheck::new(tau, poly_W.clone(), S.num_vars)
    })
    .collect::<Vec<_>>();


    let (sumcheck_result, rand_sc, claims_witness) = Self::prove_witness(
      ck,
      num_rounds_sc,
      witness_sc_inst,
      &mut transcript,
    )?;

    //need to start ipa proof here for witness.
    let evals_W = claims_witness.clone()
    .into_iter()
    .map(|claims| claims[0][0])
    .collect::<Vec<_>>();

    let evals_E = zip_with!(
      iter,
      (polys_E, pk.S_repr),
      |poly_E, s_repr| {
          let log_Ni = s_repr.N.log_2();
          let (_, rand_sc_i) = rand_sc.split_at(num_rounds_sc - log_Ni);
          let rand_sc_evals = EqPolynomial::evals_from_points(rand_sc_i);
          zip_with!(par_iter, (poly_E, rand_sc_evals), |p, eq| *p * eq).sum()
      }
    )
    .collect::<Vec<E::Scalar>>();

    let evals_W_E = zip_with!(
      (evals_W.into_iter(), evals_E.into_iter()),
      |W, E| [W, E]
    )
    .collect::<Vec<_>>();

    // Prepare witness polynomials for IPA
    let witness_polys: Vec<PolyEvalWitness<E>> = zip_with!(
      (polys_W.into_iter(), polys_E.clone().into_iter()),
      |W, E| vec![PolyEvalWitness { p: W }, PolyEvalWitness { p: E }]
    )
    .flatten()
    .collect();

    let mut rng = rand::thread_rng();
    let blinding_factors: Vec<E::Scalar> = (0..witness_polys.len())
        .map(|_| E::Scalar::random(&mut rng))
        .collect();
    
    // Blind the witness polynomials
    let blinded_witness_polys: Vec<PolyEvalWitness<E>> = witness_polys
    .into_iter()
    .zip(blinding_factors.iter())
    .map(|(mut poly, &blind)| {
        for coeff in poly.p.iter_mut() {
            *coeff += blind;
        }
        poly
    })
    .collect();

    // Commit to the blinded witness polynomials
    let blinded_witness_comms = blinded_witness_polys
    .iter()
    .map(|poly| E::CE::commit(ck, &poly.p))
    .collect::<Vec<_>>();

    // Absorb each commitment individually
    for comm in &blinded_witness_comms {
        transcript.absorb(b"c", comm);
    }

    // Prepare u_batch for blinded witness polynomials
    let c_witness = transcript.squeeze(b"c_witness")?;
    let num_vars_witness = blinded_witness_polys.iter().map(|p| p.p.len().log_2()).collect::<Vec<_>>();
    let u_batch_witness = PolyEvalInstance::<E>::batch_diff_size(
      &blinded_witness_comms,
      &evals_W_E.iter().flatten().cloned().collect::<Vec<_>>(),
      &num_vars_witness,
      rand_sc.clone(),
      c_witness
    );

    // Create a vector of references to blinded PolyEvalWitness
    let blinded_witness_poly_refs: Vec<&PolyEvalWitness<E>> = blinded_witness_polys.iter().collect();

    // Create w_batch for blinded witness polynomials
    let w_batch_witness = PolyEvalWitness::<E>::batch_diff_size(&blinded_witness_poly_refs, c_witness);

    let data_for_remote: DataForUntrustedRemote<E> = DataForUntrustedRemote {
      polys_Az_Bz_Cz: polys_Az_Bz_Cz.clone(), // Clone the polynomials
      ck: ck.clone(),
      pk_ee: pk.pk_ee.ck_s.clone(),
      vk_digest: pk.vk_digest.clone(),
      polys_tau: polys_tau.clone(),
      coords_tau: coords_tau.clone(),
      polys_L_row_col: polys_L_row_col.clone(),
      polys_E: polys_E.clone(),
      polys_Mz: polys_Mz.clone(),
      comms_Az_Bz_Cz: comms_Az_Bz_Cz.clone(),
      S_repr: pk.S_repr.clone(),
      S_comm: pk.S_comm.clone(),
      us: us.clone(),
      polys_Z: polys_Z.clone(),
      N_max: N_max,
      Nis: Nis.clone(),
      num_rounds_sc: num_rounds_sc.clone(),
      rand_sc: rand_sc.clone(),
      blinded_witness_comms,
      u_batch_witness,
      w_batch_witness,
      claims_witness,
      sumcheck_comm_polys: sumcheck_result.comm_polys,
      sumcheck_comm_evals: sumcheck_result.comm_evals,
      sumcheck_deltas: sumcheck_result.proofs.iter().map(|p| p.delta.clone()).collect(),
      sumcheck_betas: sumcheck_result.proofs.iter().map(|p| p.beta.clone()).collect(),
      sumcheck_zs: sumcheck_result.proofs.iter().map(|p| p.z.clone()).collect(),
      sumcheck_z_deltas: sumcheck_result.proofs.iter().map(|p| p.z_delta).collect(),
      sumcheck_z_betas: sumcheck_result.proofs.iter().map(|p| p.z_beta).collect(),
    };

    /*let _ = Self::prove_unstrusted(
      &data_for_remote,
    )?;*/

    //serialize_data_for_remote(&data_for_remote);

    println!("done tiny prover");
    
    Ok(Self {
      data: data_for_remote
    })
  }

  fn verify(&self, _vk: &Self::VerifierKey, _U: &[RelaxedR1CSInstance<E>]) -> Result<(), NovaError> {
    Ok(())
  }

}

impl<E: Engine + Serialize + for<'de> Deserialize<'de>> BatchedRelaxedR1CSSNARK<E> 
where
    E::CE: ZKCommitmentEngineTrait<E>,
    E::GE: DlogGroup<ScalarExt = E::Scalar>,
    E::CE: CommitmentEngineTrait<E, Commitment = zk_pedersen::Commitment<E>, CommitmentKey = zk_pedersen::CommitmentKey<E>>,
    <<E::CE as CommitmentEngineTrait<E>>::Commitment as CommitmentTrait<E>>::CompressedCommitment: Into<CompressedCommitment<E>>,
    zk_pedersen::CommitmentKey<E>: zk_pedersen::CommitmentKeyExtTrait<E>,
{
  pub fn prove_unstrusted(
      data: &DataForUntrustedRemote<E>,
  ) -> Result<(), NovaError>
  where
      E::Scalar: PrimeField,
  {
    let mut transcript = E::TE::new(b"UntrustedProver");
    transcript.absorb(b"vk", &data.vk_digest);

    // Evaluate and commit to [Az(tau), Bz(tau), Cz(tau)]
    let evals_Az_Bz_Cz_at_tau = zip_with!(
      par_iter,
      (data.polys_Az_Bz_Cz, data.coords_tau),
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

    // absorb the claimed evaluations into the transcript
    for evals in evals_Az_Bz_Cz_at_tau.iter() {
      transcript.absorb(b"e", &evals.as_slice());
    }

    let untrusted_c = transcript.squeeze(b"untrusted_c")?;

    let evals_Mz: Vec<_> = zip_with!(
      iter,
      (data.comms_Az_Bz_Cz, evals_Az_Bz_Cz_at_tau),
      |comm_Az_Bz_Cz, evals_Az_Bz_Cz_at_tau| {
        let u = PolyEvalInstance::<E>::batch(
          comm_Az_Bz_Cz.as_slice(),
          vec![], // ignored by the function
          evals_Az_Bz_Cz_at_tau.as_slice(),
          &untrusted_c,
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
        data.polys_Az_Bz_Cz.par_iter(),
        data.polys_E.par_iter(),
        <Vec<Vec<<E as traits::Engine>::Scalar>> as Clone>::clone(&data.polys_Mz).into_par_iter(),
        data.polys_tau.par_iter(),
        evals_Mz.par_iter(),
        data.us.par_iter()
      ),
      |poly_ABC, poly_E, poly_Mz, poly_tau, eval_Mz, u| {
        let [poly_Az, poly_Bz, poly_Cz] = poly_ABC;
        let poly_uCz_E = zip_with!(par_iter, (poly_Cz, poly_E), |cz, e| *u * cz + e).collect();
        OuterSumcheckInstance::<E>::new(
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
      (data.S_repr, evals_Mz, data.polys_L_row_col),
      |s_repr, eval_Mz, poly_L| {
        let [poly_L_row, poly_L_col] = poly_L;
        let c_square = untrusted_c.square();
        let val = zip_with!(
          par_iter,
          (s_repr.val_A, s_repr.val_B, s_repr.val_C),
          |v_a, v_b, v_c| *v_a + untrusted_c * *v_b + c_square * *v_c
        )
        .collect::<Vec<_>>();

        InnerSumcheckInstance::<E>::new(
          *eval_Mz,
          MultilinearPolynomial::new(poly_L_row.clone()),
          MultilinearPolynomial::new(poly_L_col.clone()),
          MultilinearPolynomial::new(val),
        )
      }
    )
    .collect::<Vec<_>>();

    let comms_L_row_col = data.polys_L_row_col
    .par_iter()
    .map(|[L_row, L_col]| {
      let (comm_L_row, comm_L_col) =
        rayon::join(|| E::CE::commit(&data.ck, L_row), || E::CE::commit(&data.ck, L_col));
      [comm_L_row, comm_L_col]
    })
    .collect::<Vec<_>>();

    // absorb commitments to L_row and L_col in the transcript
    for comms in comms_L_row_col.iter() {
      transcript.absorb(b"e", &comms.as_slice());
    }

    // a third sum-check instance to prove the read-only memory claim
    // we now need to prove that L_row and L_col are well-formed
    let (mem_sc_inst, comms_mem_oracles, _polys_mem_oracles) = {
      let gamma = transcript.squeeze(b"g")?;
      let r = transcript.squeeze(b"r")?;

      // We start by computing oracles and auxiliary polynomials to help prove the claim
      // oracles correspond to [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col]
      let (comms_mem_oracles, polys_mem_oracles, mem_aux) = data.
        S_repr
        .iter()
        .zip_eq(data.polys_tau.iter())
        .zip_eq(data.polys_Z.iter())
        .zip_eq(data.polys_L_row_col.iter())
        .try_fold(
          (Vec::new(), Vec::new(), Vec::new()),
          |(mut comms, mut polys, mut aux), (((s_repr, poly_tau), poly_Z), [L_row, L_col])| {
            let (comm, poly, a) = MemorySumcheckInstance::<E>::compute_oracles(
              &data.ck,
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
      let all_rhos = PowPolynomial::squares(&rho, data.N_max.log_2());

      let instances = zip_with!(
        (
          data.S_repr.par_iter(),
          data.Nis.par_iter(),
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

    
    // Run batched Sumcheck for the 3 claims for all instances.
    // Note that the polynomials for claims relating to instance i have size Ni.
    let (_sc, rand_sc, claims_outer, claims_inner, claims_mem) = Self::prove_helper(
      data.num_rounds_sc,
      mem_sc_inst,
      outer_sc_inst,
      inner_sc_inst,
      &mut transcript,
    )?;

    let (evals_Az_Bz_Cz_E, evals_L_row_col, evals_mem_oracle, evals_mem_preprocessed) = {
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

      let (evals_Cz_E, evals_mem_val_row_col): (Vec<_>, Vec<_>) = zip_with!(
        iter,
        (data.polys_Az_Bz_Cz, data.polys_E, data.S_repr),
        |ABCzs, poly_E, s_repr| {
          let [_, _, Cz] = ABCzs;
          let log_Ni = s_repr.N.log_2();
          let (_, rand_sc) = rand_sc.split_at(data.num_rounds_sc - log_Ni);
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

      let evals_Az_Bz_Cz_E = zip_with!(
        (evals_Az_Bz.into_iter(), evals_Cz_E.into_iter()),
        |Az_Bz, Cz_E| {
          let [Az, Bz] = Az_Bz;
          let [Cz, E] = Cz_E;
          [Az, Bz, Cz, E]
        }
      )
      .collect::<Vec<_>>();

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
      (
        evals_Az_Bz_Cz_E,
        evals_L_row_col,
        evals_mem_oracle,
        evals_mem_preprocessed,
      )
    };

    let evals_vec = zip_with!(
      iter,
      (
        evals_Az_Bz_Cz_E,
        evals_L_row_col,
        evals_mem_oracle,
        evals_mem_preprocessed
      ),
      |Az_Bz_Cz_E, L_row_col, mem_oracles, mem_preprocessed| {
        chain![Az_Bz_Cz_E, L_row_col, mem_oracles, mem_preprocessed]
          .cloned()
          .collect::<Vec<_>>()
      }
    )
    .collect::<Vec<_>>();

    let comms_vec = zip_with!(
      iter,
      (
        data.comms_Az_Bz_Cz,
        comms_L_row_col,
        comms_mem_oracles,
        data.S_comm
      ),
      |Az_Bz_Cz, L_row_col, mem_oracles, S_comm| {
        chain![
          Az_Bz_Cz,
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

    for evals in evals_vec.iter() {
      transcript.absorb(b"e", &evals.as_slice()); // comm_vec is already in the transcript
    }
    let evals_vec = evals_vec.into_iter().flatten().collect::<Vec<_>>();

    let untrusted_c = transcript.squeeze(b"untrusted_c")?;

    // Compute number of variables for each polynomial
    let num_vars_u: Vec<usize> = [
      Self::repeat_log2(&data.S_repr, 3), // For Az, Bz, Cz (3 repetitions)
      Self::repeat_log2(&data.S_repr, 1), // For E (1 repetition)
      Self::repeat_log2(&data.S_repr, 2), // For L_row, L_col (2 repetitions)
      Self::repeat_log2(&data.S_repr, 4), // For mem_oracles (4 repetitions)
      Self::repeat_log2(&data.S_repr, 7)  // For preprocessed values (7 repetitions)
    ]
    .into_iter()
    .flatten()
    .collect();

    assert_eq!(num_vars_u.len(), comms_vec.len(), "Number of variables must match number of commitments");
    assert_eq!(evals_vec.len(), comms_vec.len(), "Number of evaluations must match number of commitments");

    let _u_batch = PolyEvalInstance::<E>::batch_diff_size(&comms_vec, &evals_vec, &num_vars_u, rand_sc, untrusted_c);

    // Perform IPA for batched polynomials
    /*let eval_arg = ipa_pc::EvaluationEngine::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &u_batch.c,
      &w_batch.p,
      &u_batch.x,
      &u_batch.e,
    )?;*/
        
    // Perform IPA for batched witness polynomials
    let _eval_arg_witness = zk_ipa_pc::EvaluationEngine::<E>::prove_not_zk(
        &data.ck,
        &data.pk_ee,
        &mut transcript,
        &data.u_batch_witness.c,
        &data.w_batch_witness.p,
        &data.u_batch_witness.x,
        &data.u_batch_witness.e,
    )?;

    let _comms_Az_Bz_Cz: Vec<[CompressedCommitment<E>; 3]> = <Vec<[zk_pedersen::Commitment<E>; 3]> as Clone>::clone(&data.comms_Az_Bz_Cz)
    .into_iter()
    .map(|comms| [comms[0].compress(), comms[1].compress(), comms[2].compress()])
    .collect();

    let _comms_L_row_col: Vec<[CompressedCommitment<E>; 2]> = comms_L_row_col
        .into_iter()
        .map(|comms| [comms[0].compress(), comms[1].compress()])
        .collect();

    let _comms_mem_oracles: Vec<[CompressedCommitment<E>; 4]> = comms_mem_oracles
        .into_iter()
        .map(|comms| [comms[0].compress(), comms[1].compress(), comms[2].compress(), comms[3].compress()])
        .collect();

    Ok(())
  }

  fn repeat_log2(s_repr: &[R1CSShapeSparkRepr<E>], repetitions: usize) -> impl Iterator<Item = usize> + '_ {
    s_repr.iter().flat_map(move |s| iter::repeat(s.N.log_2()).take(repetitions))
  }
  
  //Prove only the witness claims.
  #[allow(unused)]
  fn prove_witness<T1>(
    ck: &CommitmentKey<E>,
    num_rounds: usize,
    mut witness: Vec<T1>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      ZKSumcheckProof<E>,
      Vec<E::Scalar>,
      Vec<Vec<Vec<E::Scalar>>>,
    ),
    NovaError,
  >
  where
    T1: ZKSumcheckEngine<E>,
  {

    for inst in witness.iter() {
      assert!(inst.size().is_power_of_two());
    }

    let claims = witness
    .iter()
    .flat_map(|witness| Self::scaled_claims(witness, num_rounds))
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
      // At the start of round i, the input polynomials are defined over at most n-i variables.
      let remaining_variables = num_rounds - i;
    
      // For each claim j, compute the evaluations of its univariate polynomial S_j(X_i)
      // at X = 0, 2, 3. The polynomial is such that S_{j-1}(r_{j-1}) = S_j(0) + S_j(1).
      // If the number of variable m of the claim is m < n-i, then the polynomial is
      // constant and equal to the initial claim σ_j scaled by 2^{n-m-i-1}.
      let evals = witness
        .par_iter()
        .flat_map(|witness| Self::get_evals(witness, remaining_variables))
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
      witness
        .par_iter_mut()
        .for_each(|witness| Self::bind(witness, remaining_variables, &r_i));
    
      running_claim = poly.evaluate(&r_i);

      let comm_e = <E::CE as ZKCommitmentEngineTrait<E>>::zkcommit(ck, &[running_claim], &blinds_evals[i]).compress();
      let comm_poly = <E::CE as ZKCommitmentEngineTrait<E>>::zkcommit(ck, &poly.coeffs, &blinds_poly[i]).compress();

      comm_polys.push(comm_poly);
      comm_evals.push(comm_e);
      cubic_polys.push(poly.compress());
    }

    // Collect evaluations at (r_{n-m}, ..., r_{n-1}) of polynomials over all claims,
    // where m is the initial number of variables the individual claims are defined over.
    let claims_witness = witness
      .into_iter()
      .map(|inst| inst.final_claims())
      .collect();


    Ok((
      ZKSumcheckProof::<E>::new(comm_polys, comm_evals, proofs),
      r,
      claims_witness,
    ))
  }

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
  fn prove_helper<T1, T2, T3>(
    num_rounds: usize,
    mut mem: Vec<T1>,
    mut outer: Vec<T2>,
    mut inner: Vec<T3>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      SumcheckProof<E>,
      Vec<E::Scalar>,
      Vec<Vec<Vec<E::Scalar>>>,
      Vec<Vec<Vec<E::Scalar>>>,
      Vec<Vec<Vec<E::Scalar>>>,
    ),
    NovaError,
  >
  where
    T1: SumcheckEngine<E>,
    T2: SumcheckEngine<E>,
    T3: SumcheckEngine<E>,
  {
    // sanity checks
    let num_instances = mem.len();
    assert_eq!(outer.len(), num_instances);
    assert_eq!(inner.len(), num_instances);

    for inst in mem.iter_mut() {
      assert!(SumcheckEngine::size(inst).is_power_of_two());
    }
    for inst in outer.iter() {
      assert!(SumcheckEngine::size(inst).is_power_of_two());
    }
    for inst in inner.iter() {
      assert!(SumcheckEngine::size(inst).is_power_of_two());
    }

    let degree = SumcheckEngine::degree(&mem[0]);
    assert!(mem.iter().all(|inst|  SumcheckEngine::degree(inst) == degree));
    assert!(outer.iter().all(|inst| SumcheckEngine::degree(inst) == degree));
    assert!(inner.iter().all(|inst| SumcheckEngine::degree(inst) == degree));

    // Collect all claims from the instances. If the instances is defined over `m` variables,
    // which is less that the total number of rounds `n`,
    // the individual claims σ are scaled by 2^{n-m}.
    let claims = zip_with!(
      iter,
      (mem, outer, inner),
      |mem, outer, inner| {
        Self::scaled_claims_not_zk(mem, num_rounds)
          .into_iter()
          .chain(Self::scaled_claims_not_zk(outer, num_rounds))
          .chain(Self::scaled_claims_not_zk(inner, num_rounds))
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

    for i in 0..num_rounds {
      // At the start of round i, there input polynomials are defined over at most n-i variables.
      let remaining_variables = num_rounds - i;

      // For each claim j, compute the evaluations of its univariate polynomial S_j(X_i)
      // at X = 0, 2, 3. The polynomial is such that S_{j-1}(r_{j-1}) = S_j(0) + S_j(1).
      // If the number of variable m of the claim is m < n-i, then the polynomial is
      // constants and equal to the initial claim σ_j scaled by 2^{n-m-i-1}.
      let evals = zip_with!(
        par_iter,
        (mem, outer, inner),
        |mem, outer, inner| {
          let ((evals_mem, evals_outer), evals_inner) = rayon::join(
            || rayon::join(
              || Self::get_evals_not_zk(mem, remaining_variables),
              || Self::get_evals_not_zk(outer, remaining_variables)
            ),
            || Self::get_evals_not_zk(inner, remaining_variables)
          );
          evals_mem
            .into_par_iter()
            .chain(evals_outer.into_par_iter())
            .chain(evals_inner.into_par_iter())
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
        (mem, outer, inner),
        |mem, outer, inner| {
          rayon::join(
            || Self::bind_not_zk(mem, remaining_variables, &r_i),
            || rayon::join(
              || Self::bind_not_zk(outer, remaining_variables, &r_i),
              || Self::bind_not_zk(inner, remaining_variables, &r_i),
            ),
          );
        }
      );

      running_claim = poly.evaluate(&r_i);
      cubic_polys.push(poly.compress());
    }

    // Collect evaluations at (r_{n-m}, ..., r_{n-1}) of polynomials over all claims,
    // where m is the initial number of variables the individual claims are defined over.
    let claims_outer = outer.into_iter().map(|inst| SumcheckEngine::final_claims(&inst)).collect();
    let claims_inner = inner.into_iter().map(|inst| SumcheckEngine::final_claims(&inst)).collect();
    let claims_mem = mem.into_iter().map(|inst| SumcheckEngine::final_claims(&inst)).collect();

    Ok((
      SumcheckProof::new(cubic_polys),
      r,
      claims_outer,
      claims_inner,
      claims_mem,
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

  fn get_evals_not_zk<T: SumcheckEngine<E>>(inst: &T, remaining_variables: usize) -> Vec<Vec<E::Scalar>> {
    let num_instance_variables = inst.size().log_2(); // m
    if num_instance_variables < remaining_variables {
      let deg = inst.degree();

      // The evaluations at X_i = 0, 2, 3 are all equal to the scaled claim
      Self::scaled_claims_not_zk(inst, remaining_variables - 1)
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

  fn bind_not_zk<T: SumcheckEngine<E>>(inst: &mut T, remaining_variables: usize, r: &E::Scalar) {
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

    /// Given an instance defined over m variables, the sum over n = `remaining_variables` is equal
  /// to the initial claim scaled by 2^{n-m}, when m ≤ n.   
  fn scaled_claims_not_zk<T: SumcheckEngine<E>>(inst: &T, remaining_variables: usize) -> Vec<E::Scalar> {
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
