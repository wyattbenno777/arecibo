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
    },
    zksumcheck::{
      engine::{
        ZKSumcheckEngine,
        WitnessBoundSumcheck,
      },
      ZKSumcheckProof,
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
  zip_with
};
//use crate::spartan::nizk::ProductProof;
use core::ops::{Add, Sub, Mul};
use crate::spartan::zksnark::SumcheckGens;
use crate::provider::zk_pedersen;
use crate::provider::zk_ipa_pc;
use crate::provider::traits::DlogGroup;
use core::slice;
use ff::Field;
use itertools::{Itertools as _};
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
  polys_tau: Vec<Vec<E::Scalar>>, 
  coords_tau: Vec<Vec<E::Scalar>>,
  polys_L_row_col: Vec<[Vec<E::Scalar>; 2]>,
  polys_E: Vec<Vec<E::Scalar>>, 
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
    serialize_field("polys_tau", &data.polys_tau);
    serialize_field("coords_tau", &data.coords_tau);
    serialize_field("polys_L_row_col", &data.polys_L_row_col);
    serialize_field("polys_E", &data.polys_E);
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

    println!("In tiny prover");

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

    //let evals_Az_Bz_Cz_at_tau: Vec<[E::Scalar; 3]> = vec![[E::Scalar::zero(); 3]; polys_Az_Bz_Cz.len()];

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
      polys_tau: polys_tau.clone(),
      coords_tau: coords_tau.clone(),
      polys_L_row_col: polys_L_row_col.clone(),
      polys_E: polys_E.clone(),
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
        let mut transcript = E::TE::new(b"BatchedRelaxedR1CSSNARK");
        
        // Perform IPA for witness polynomials
        let _eval_arg_witness = zk_ipa_pc::EvaluationEngine::<E>::prove_not_zk(
            &data.ck,
            &data.pk_ee,
            &mut transcript,
            &data.u_batch_witness.c,
            &data.w_batch_witness.p,
            &data.u_batch_witness.x,
            &data.u_batch_witness.e,
        )?;


        Ok(())
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
