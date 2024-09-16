use crate::spartan::math::Math;
use crate::spartan::verify_circuit::gadgets::{
  poseidon_transcript::PoseidonTranscriptCircuit, sumcheck::SCVerifyBatchedGadget,
};
use crate::traits::commitment::CommitmentTrait;
use crate::{
  gadgets::{alloc_negate, alloc_one, alloc_zero},
  provider::PallasEngine,
  spartan::PolyEvalInstance,
  traits::snark::DigestHelperTrait,
  zip_with,
};
use crate::{r1cs::RelaxedR1CSInstance, traits::Engine, Commitment};
use generic_array::typenum::U24;
use itertools::chain;
use poseidon_sponge::sponge::{circuit::SpongeCircuit, vanilla::Sponge};
use poseidon_sponge::{sponge::vanilla::SpongeTrait, Strength};

use crate::spartan::verify_circuit::gadgets::poly::{
  alloc_powers, AllocatedEqPolynomial, AllocatedIdentityPolynomial, AllocatedMaskedEqPolynomial,
  AllocatedPowPolynomial, AllocatedSparsePolynomial,
};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, Namespace, SynthesisError};

use itertools::Itertools as _;

use crate::spartan::verify_circuit::ipa_prover_poseidon::batched_ppsnark::{
  BatchedRelaxedR1CSSNARK, VerifierKey,
};
use poseidon_sponge::poseidon::PoseidonConstants;

#[cfg(test)]
mod tests;

type E = PallasEngine;
type SNARK<E> = BatchedRelaxedR1CSSNARK<E>;

type Scalar = <E as Engine>::Scalar;

pub struct SpartanVerifyCircuit {
  snark: SNARK<E>,
  constants: PoseidonConstants<Scalar, U24>,
}

impl SpartanVerifyCircuit {
  /// Create new instance of SpartanVerifyCircuit
  pub fn new(snark: SNARK<E>) -> Self {
    let constants = Sponge::<Scalar, U24>::api_constants(Strength::Standard);
    Self { snark, constants }
  }

  /// Allocate witnesses and return
  fn alloc_witness<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    alloc_c: &AllocatedNum<Scalar>,
    alloc_tau_coords: &[AllocatedNum<Scalar>],
  ) -> Result<
    (
      Vec<AllocatedNum<Scalar>>,      // alloc_evals_Mz
      Vec<Vec<AllocatedNum<Scalar>>>, // alloc_evals_Az_Bz_Cz_W_E
      Vec<Vec<AllocatedNum<Scalar>>>, // alloc_evals_L_row_col
      Vec<Vec<AllocatedNum<Scalar>>>, // alloc_evals_mem_oracle
      Vec<Vec<AllocatedNum<Scalar>>>, // alloc_evals_mem_preprocessed
    ),
    SynthesisError,
  > {
    // decompress comms_Az_Bz_Cz.
    // used to get evals_Mz
    let comms_Az_Bz_Cz = self
      .snark
      .comms_Az_Bz_Cz
      .iter()
      .map(|comms| {
        comms
          .iter()
          .map(Commitment::<E>::decompress)
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()
      .expect("decompress comms_Az_Bz_Cz");

    // Compute eval_Mz = eval_Az_at_tau + c * eval_Bz_at_tau + c^2 * eval_Cz_at_tau
    let tau_coords = alloc_tau_coords
      .iter()
      .map(|tau_coord| {
        tau_coord
          .get_value()
          .ok_or(SynthesisError::AssignmentMissing)
      })
      .collect::<Result<Vec<Scalar>, _>>()?;

    let c = alloc_c
      .get_value()
      .ok_or(SynthesisError::AssignmentMissing)?;

    let evals_Mz: Vec<_> = zip_with!(
      iter,
      (comms_Az_Bz_Cz, self.snark.evals_Az_Bz_Cz_at_tau),
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

    let alloc_evals_Mz = evals_Mz
      .iter()
      .enumerate()
      .map(|(i, eval)| AllocatedNum::alloc(cs.namespace(|| format!("eval_Mz_{i}")), || Ok(*eval)))
      .collect::<Result<Vec<_>, _>>()?;

    // evals_Az_Bz_Cz_W_E
    let alloc_evals_Az_Bz_Cz_W_E = self
      .snark
      .evals_Az_Bz_Cz_W_E
      .iter()
      .enumerate()
      .map(|(i, evals)| {
        evals
          .iter()
          .enumerate()
          .map(|(j, eval)| {
            AllocatedNum::alloc(
              cs.namespace(|| format!("evals_Az_Bz_Cz_W_E {i}, {j}")),
              || Ok(*eval),
            )
          })
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Allocate evals_L_row_col
    let alloc_evals_L_row_col = self
      .snark
      .evals_L_row_col
      .iter()
      .enumerate()
      .map(|(i, evals)| {
        evals
          .iter()
          .enumerate()
          .map(|(j, eval)| {
            AllocatedNum::alloc(cs.namespace(|| format!("evals_L_row_col {i}, {j}")), || {
              Ok(*eval)
            })
          })
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Allocate evals_mem_oracles
    let alloc_evals_mem_oracle = self
      .snark
      .evals_mem_oracle
      .iter()
      .enumerate()
      .map(|(i, evals)| {
        evals
          .iter()
          .enumerate()
          .map(|(j, eval)| {
            AllocatedNum::alloc(
              cs.namespace(|| format!("evals_mem_oracles {i}, {j}")),
              || Ok(*eval),
            )
          })
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Allocate evals_mem_preprocessed
    let alloc_evals_mem_preprocessed = self
      .snark
      .evals_mem_preprocessed
      .iter()
      .enumerate()
      .map(|(i, evals)| {
        evals
          .iter()
          .enumerate()
          .map(|(j, eval)| {
            AllocatedNum::alloc(
              cs.namespace(|| format!("evals_mem_preprocessed {i}, {j}")),
              || Ok(*eval),
            )
          })
          .collect::<Result<Vec<_>, _>>()
      })
      .collect::<Result<Vec<_>, _>>()?;
    Ok((
      alloc_evals_Mz,
      alloc_evals_Az_Bz_Cz_W_E,
      alloc_evals_L_row_col,
      alloc_evals_mem_oracle,
      alloc_evals_mem_preprocessed,
    ))
  }

  /// Instantiate the Spartan Poseidon transcript circuit and its required sponge function
  fn sponge_transcript_circuit<'a, CS: ConstraintSystem<Scalar>>(
    &'a self,
    ns: &mut Namespace<'a, Scalar, CS>,
    _num_rounds_max: usize,
  ) -> (
    PoseidonTranscriptCircuit<E>,
    SpongeCircuit<'a, Scalar, U24, CS>,
  ) {
    let mut sponge = SpongeCircuit::<Scalar, U24, _>::new_with_constants(
      &self.constants,
      poseidon_sponge::sponge::vanilla::Mode::Simplex,
    );

    let transcript_circuit = PoseidonTranscriptCircuit::<E>::new(&mut sponge, ns);

    (transcript_circuit, sponge)
  }

  /// Run circuit through constraint system to build R1CS shapes, instance and witness
  pub fn synthesize<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    vk: &VerifierKey<E>,
    U: &[RelaxedR1CSInstance<E>],
  ) -> Result<(), SynthesisError> {
    // Allocated constants that will be used throughout circuit for different computations that require a an allocated zero or one variable
    let alloc_one = alloc_one(cs.namespace(|| "one"));
    let alloc_zero = alloc_zero(cs.namespace(|| "zero"));

    // number of rounds of sum-check
    let num_rounds = vk.S_comm.iter().map(|s| s.N.log_2()).collect::<Vec<_>>();
    let num_rounds_max = *num_rounds.iter().max().unwrap();

    // Instantiate transcript circuit
    let mut cs = cs.namespace(|| "ns");
    let (mut transcript, mut sponge) = self.sponge_transcript_circuit(&mut cs, num_rounds_max);

    // Absorb vk
    let alloc_vk_digest = AllocatedNum::alloc(cs.namespace(|| "vk_digest"), || Ok(vk.digest()))?;
    transcript.absorb(alloc_vk_digest, &mut sponge, &mut cs);

    // Squeeze out tau
    let alloc_tau = transcript.squeeze(String::from("tau"), &mut sponge, &mut cs)?;
    let alloc_tau_coords = AllocatedPowPolynomial::new(
      cs.namespace(|| "Pow polynomials"),
      &alloc_tau,
      num_rounds_max,
    )?
    .coordinates();

    // Squeeze c (used with tau coords to compute evals_Mz in Self::alloc_witness)
    let alloc_c = transcript.squeeze(String::from("c"), &mut sponge, &mut cs)?;

    // Alloc witnesses + (computes values for evals_Mz and allocates it)
    let (
      alloc_evals_Mz,
      alloc_evals_Az_Bz_Cz_W_E,
      alloc_evals_L_row_col,
      alloc_evals_mem_oracle,
      alloc_evals_mem_preprocessed,
    ) = self.alloc_witness(
      cs.namespace(|| "alloc witnesses"),
      &alloc_c,
      &alloc_tau_coords,
    )?;

    // Squeeze gamma, r, rho, s
    let alloc_gamma = transcript.squeeze(String::from("gamma"), &mut sponge, &mut cs)?;
    let alloc_r = transcript.squeeze(String::from("r"), &mut sponge, &mut cs)?;
    let alloc_rho = transcript.squeeze(String::from("rho"), &mut sponge, &mut cs)?;
    let alloc_s = transcript.squeeze(String::from("s"), &mut sponge, &mut cs)?;

    // Get powers of s
    let num_instances = U.len();
    let num_claims_per_instance = 10;
    let alloc_s_powers = alloc_powers(
      cs.namespace(|| "s powers"),
      &alloc_s,
      num_instances * num_claims_per_instance,
    )?;

    // Run sumcheck verify
    let (alloc_claim_sc_final, alloc_rand_sc) = {
      // Gather all claims into a single vector
      let alloc_claims = alloc_evals_Mz
        .iter()
        .flat_map(|eval_Mz| {
          let mut claims = (0..num_claims_per_instance)
            .map(|_| alloc_zero.clone())
            .collect::<Vec<_>>();

          claims[7] = eval_Mz.clone();
          claims[8] = eval_Mz.clone();
          claims.into_iter()
        })
        .collect::<Vec<_>>();

      // Number of rounds for each claim
      let num_rounds_by_claim = num_rounds
        .iter()
        .flat_map(|num_rounds_i| vec![*num_rounds_i; num_claims_per_instance].into_iter())
        .collect::<Vec<_>>();

      let alloc_sc = SCVerifyBatchedGadget::new(self.snark.sc.clone());
      let alloc_degree_bound =
        AllocatedNum::alloc(cs.namespace(|| "degree_bound"), || Ok(Scalar::from(3_u64)))?;

      alloc_sc.verify_batched(
        &mut cs,
        &alloc_claims,
        &num_rounds_by_claim,
        &alloc_s_powers,
        &alloc_degree_bound,
        &mut transcript,
        &mut sponge,
        "sc_verify",
      )?
    };

    let alloc_rand_sc_i = num_rounds
      .iter()
      .map(|num_rounds| alloc_rand_sc[(num_rounds_max - num_rounds)..].to_vec())
      .collect::<Vec<_>>();

    let mut alloc_chained_claims = vec![];

    for (i, (num_vars, U)) in vk.num_vars.iter().zip_eq(U.iter()).enumerate() {
      let alloc_rand_sc = &alloc_rand_sc_i[i];

      // get evals_Az_Bz_Cz_W_E
      let alloc_eval_Az_Bz_Cz_W_E = &alloc_evals_Az_Bz_Cz_W_E[i];
      let alloc_Az = &alloc_eval_Az_Bz_Cz_W_E[0];
      let alloc_Bz = &alloc_eval_Az_Bz_Cz_W_E[1];
      let alloc_Cz = &alloc_eval_Az_Bz_Cz_W_E[2];
      let alloc_W = &alloc_eval_Az_Bz_Cz_W_E[3];
      let alloc_E = &alloc_eval_Az_Bz_Cz_W_E[4];

      // get evals_L_row_col
      let alloc_eval_L_row_col = &alloc_evals_L_row_col[i];
      let alloc_L_row = &alloc_eval_L_row_col[0];
      let alloc_L_col = &alloc_eval_L_row_col[1];

      // get evals_mem_preprocessed
      let alloc_eval_mem_oracle = &alloc_evals_mem_oracle[i];
      let alloc_t_plus_r_inv_row = &alloc_eval_mem_oracle[0];
      let alloc_w_plus_r_inv_row = &alloc_eval_mem_oracle[1];
      let alloc_t_plus_r_inv_col = &alloc_eval_mem_oracle[2];
      let alloc_w_plus_r_inv_col = &alloc_eval_mem_oracle[3];

      // get evals_mem_preprocessed
      let alloc_eval_mem_preprocessed = &alloc_evals_mem_preprocessed[i];
      let alloc_val_A = &alloc_eval_mem_preprocessed[0];
      let alloc_val_B = &alloc_eval_mem_preprocessed[1];
      let alloc_val_C = &alloc_eval_mem_preprocessed[2];
      let alloc_row = &alloc_eval_mem_preprocessed[3];
      let alloc_col = &alloc_eval_mem_preprocessed[4];
      let alloc_ts_row = &alloc_eval_mem_preprocessed[5];
      let alloc_ts_col = &alloc_eval_mem_preprocessed[6];

      let num_rounds_i = alloc_rand_sc.len();
      let num_vars_log = num_vars.log_2();

      let alloc_eq_rho = AllocatedPowPolynomial::new(
        cs.namespace(|| format!("eq_rho_poly_{i}")),
        &alloc_rho,
        num_rounds_i,
      )?
      .evaluate(
        cs.namespace(|| format!("eq_rho_poly_eval_{i}")),
        &alloc_rand_sc,
      )?;

      let (alloc_eq_tau, alloc_eq_masked_tau) = {
        let alloc_eq_tau: AllocatedEqPolynomial<_> = AllocatedPowPolynomial::new(
          cs.namespace(|| format!("eq_tau_pow_poly_{i}")),
          &alloc_tau,
          num_rounds_i,
        )?
        .into();

        let alloc_eq_tau_at_rand = alloc_eq_tau.evaluate(
          cs.namespace(|| format!("eq_tau_eval()_{i}")),
          &alloc_rand_sc,
        )?;

        let alloc_eq_masked_tau = AllocatedMaskedEqPolynomial::new(&alloc_eq_tau, num_vars_log)
          .evaluate(
            cs.namespace(|| format!("eq_masked_tau_eval()_{i}")),
            &alloc_rand_sc,
          )?;

        (alloc_eq_tau_at_rand, alloc_eq_masked_tau)
      };

      // Evaluate identity polynomial
      let alloc_id = AllocatedIdentityPolynomial::new(num_rounds_i).evaluate(
        cs.namespace(|| format!("id_poly_{i}.eval()")),
        &alloc_rand_sc,
      )?;

      let alloc_Z = {
        // rand_sc was padded, so we now remove the padding
        let (alloc_factor, alloc_rand_sc_unpad) = {
          let l = num_rounds_i - (num_vars_log + 1);

          let (alloc_rand_sc_lo, alloc_rand_sc_hi) = alloc_rand_sc.split_at(l);
          let mut alloc_factor = alloc_one.clone();
          // let factor = rand_sc_lo
          //   .iter()
          //   .fold(E::Scalar::ONE, |acc, r_p| acc * (E::Scalar::ONE - r_p));
          for alloc_r_p in alloc_rand_sc_lo.iter() {
            let alloc_neg_r_p = alloc_negate(cs.namespace(|| format!("r_p_{i}")), alloc_r_p)?;
            let alloc_one_minus_r_p = alloc_one.add(
              cs.namespace(|| format!("one_minus_r_p_{i}")),
              &alloc_neg_r_p,
            )?;
            alloc_factor =
              alloc_factor.mul(cs.namespace(|| format!("factor_{i}")), &alloc_one_minus_r_p)?;
          }
          (alloc_factor, alloc_rand_sc_hi)
        };

        let alloc_X = {
          // constant term
          let poly_X: Vec<Scalar> = std::iter::once(U.u).chain(U.X.iter().cloned()).collect();
          let alloc_poly_X = poly_X
            .iter()
            .map(|x| AllocatedNum::alloc(cs.namespace(|| format!("poly_X_{i}")), || Ok(*x)))
            .collect::<Result<Vec<_>, _>>()?;

          AllocatedSparsePolynomial::new(num_vars_log, alloc_poly_X).evaluate(
            cs.namespace(|| format!("evaluate sparse_ml poly_{i}")),
            &alloc_rand_sc_unpad[1..],
          )?
        };

        // W was evaluated as if it was padded to logNi variables,
        // so we don't multiply it by (1-rand_sc_unpad[0])
        // W + factor * rand_sc_unpad[0] * X
        let alloc_rand_sc_unpad_times_X = alloc_rand_sc_unpad[0].mul(
          cs.namespace(|| format!("rand_sc_unpad_times_X_{i}")),
          &alloc_X,
        )?;
        let factor_times_rand_sc_unpad_times_X = alloc_factor.mul(
          cs.namespace(|| format!("factor_times_rand_sc_unpad_times_X_{i}")),
          &alloc_rand_sc_unpad_times_X,
        )?;

        alloc_W.add(
          cs.namespace(|| format!("W_plus_factor_times_rand_sc_unpad_times_X_{i}")),
          &factor_times_rand_sc_unpad_times_X,
        )?
      };

      // let t_plus_r_row = {
      //   let addr_row = id;
      //   let val_row = eq_tau;
      //   let t = addr_row + gamma * val_row;
      //   t + r
      // };
      let alloc_t_plus_r_row = {
        let alloc_addr_row = alloc_id.clone();
        let alloc_val_row = alloc_eq_tau.clone();

        // let t = addr_row + gamma * val_row;
        let alloc_gamma_times_val_row = alloc_gamma.mul(
          cs.namespace(|| format!("t_plus_r_row: gamma * val_row {i}")),
          &alloc_val_row,
        )?;
        let alloc_t = alloc_addr_row.add(
          cs.namespace(|| format!("t_plus_r_row: addr_row + gamma * val_row {i}")),
          &alloc_gamma_times_val_row,
        )?;

        // t + r
        alloc_t.add(
          cs.namespace(|| format!("t_plus_r_row: t + r {i}")),
          &alloc_r,
        )?
      };

      // let w_plus_r_row = {
      //   let addr_row = row;
      //   let val_row = L_row;
      //   let w = addr_row + gamma * val_row;
      //   w + r
      // };
      let alloc_w_plus_r_row = {
        let alloc_addr_row = alloc_row;
        let alloc_val_row = alloc_L_row.clone();

        // let w = addr_row + gamma * val_row;
        let alloc_gamma_times_val_row = alloc_gamma.mul(
          cs.namespace(|| format!("w_plus_r_row: gamma * val_row {i}")),
          &alloc_val_row,
        )?;
        let alloc_w = alloc_addr_row.add(
          cs.namespace(|| format!("w_plus_r_row: addr_row + gamma * val_row {i}")),
          &alloc_gamma_times_val_row,
        )?;
        // w + r
        alloc_w.add(
          cs.namespace(|| format!("w_plus_r_row: w + r {i}")),
          &alloc_r,
        )?
      };

      // let t_plus_r_col = {
      //   let addr_col = id;
      //   let val_col = Z;
      //   let t = addr_col + gamma * val_col;
      //   t + r
      // };
      let alloc_t_plus_r_col = {
        let alloc_addr_col = alloc_id;
        let alloc_val_col = alloc_Z;

        // let t = addr_col + gamma * val_col;
        let alloc_gamma_times_val_col = alloc_gamma.mul(
          cs.namespace(|| format!("t_plus_r_col: gamma * val_col {i}")),
          &alloc_val_col,
        )?;
        let alloc_t = alloc_addr_col.add(
          cs.namespace(|| format!("t_plus_r_col: addr_col + gamma * val_col {i}")),
          &alloc_gamma_times_val_col,
        )?;

        // t + r
        alloc_t.add(
          cs.namespace(|| format!("t_plus_r_col: t + r {i}")),
          &alloc_r,
        )?
      };

      // let w_plus_r_col = {
      //   let addr_col = col;
      //   let val_col = L_col;
      //   let w = addr_col + gamma * val_col;
      //   w + r
      // };
      let alloc_w_plus_r_col = {
        let alloc_addr_col = alloc_col;
        let alloc_val_col = alloc_L_col.clone();

        // let w = addr_col + gamma * val_col;
        let alloc_gamma_times_val_col = alloc_gamma.mul(
          cs.namespace(|| format!("w_plus_r_col: gamma * val_col {i}")),
          &alloc_val_col,
        )?;
        let alloc_w = alloc_addr_col.add(
          cs.namespace(|| format!("w_plus_r_col: addr_col + gamma * val_col {i}")),
          &alloc_gamma_times_val_col,
        )?;

        // w + r
        alloc_w.add(
          cs.namespace(|| format!("w_plus_r_col: w + r {i}")),
          &alloc_r,
        )?
      };

      // let claims_mem = [
      //   t_plus_r_inv_row - w_plus_r_inv_row,
      //   t_plus_r_inv_col - w_plus_r_inv_col,
      //   eq_rho * (t_plus_r_inv_row * t_plus_r_row - ts_row),
      //   eq_rho * (w_plus_r_inv_row * w_plus_r_row - E::Scalar::ONE),
      //   eq_rho * (t_plus_r_inv_col * t_plus_r_col - ts_col),
      //   eq_rho * (w_plus_r_inv_col * w_plus_r_col - E::Scalar::ONE),
      // ];
      let alloc_claims_mem_0 = {
        let alloc_neg_w_plus_r_inv_row = alloc_negate(
          cs.namespace(|| format!("neg_w_plus_r_inv_row_{i}")),
          &alloc_w_plus_r_inv_row,
        )?;
        alloc_t_plus_r_inv_row.add(
          cs.namespace(|| format!("t_plus_r_inv_row - w_plus_r_inv_row_{i}")),
          &alloc_neg_w_plus_r_inv_row,
        )?
      };

      // alloc_claims_mem_1
      let alloc_claims_mem_1 = {
        //   t_plus_r_inv_col - w_plus_r_inv_col,
        let alloc_neg_w_plus_r_inv_col = alloc_negate(
          cs.namespace(|| format!("neg_w_plus_r_inv_col_{i}")),
          &alloc_w_plus_r_inv_col,
        )?;
        alloc_t_plus_r_inv_col.add(
          cs.namespace(|| format!("t_plus_r_inv_col - w_plus_r_inv_col_{i}")),
          &alloc_neg_w_plus_r_inv_col,
        )?
      };

      // alloc_claims_mem_2
      let alloc_neg_ts_row =
        alloc_negate(cs.namespace(|| format!("neg_ts_row_{i}")), &alloc_ts_row)?;
      let alloc_t_plus_r_inv_row_times_t_plus_r_row = alloc_t_plus_r_inv_row.mul(
        cs.namespace(|| format!("t_plus_r_inv_row * t_plus_r_row_{i}")),
        &alloc_t_plus_r_row,
      )?;
      // (t_plus_r_inv_row * t_plus_r_row - ts_row)
      let alloc_term = alloc_t_plus_r_inv_row_times_t_plus_r_row.add(
        cs.namespace(|| format!("t_plus_r_inv_row * t_plus_r_row - ts_row_{i}")),
        &alloc_neg_ts_row,
      )?;
      let alloc_claims_mem_2 = alloc_eq_rho.mul(
        cs.namespace(|| format!("eq_rho * (t_plus_r_inv_row * t_plus_r_row - ts_row_{i})")),
        &alloc_term,
      )?;

      // alloc_claims_mem_3
      let alloc_neg_one = alloc_negate(cs.namespace(|| format!("neg_one_{i}")), &alloc_one)?;
      let alloc_w_plus_r_inv_row_times_w_plus_r_row = alloc_w_plus_r_inv_row.mul(
        cs.namespace(|| format!("w_plus_r_inv_row * w_plus_r_row_{i}")),
        &alloc_w_plus_r_row,
      )?;
      // (w_plus_r_inv_row * w_plus_r_row - E::Scalar::ONE)
      let alloc_term = alloc_w_plus_r_inv_row_times_w_plus_r_row.add(
        cs.namespace(|| format!("w_plus_r_inv_row * w_plus_r_row - E::Scalar::ONE_{i}")),
        &alloc_neg_one,
      )?;
      let alloc_claims_mem_3 = alloc_eq_rho.mul(
        cs.namespace(|| format!("eq_rho * (w_plus_r_inv_row * w_plus_r_row - E::Scalar::ONE_{i})")),
        &alloc_term,
      )?;

      // alloc_claims_mem_4
      let alloc_neg_ts_col =
        alloc_negate(cs.namespace(|| format!("neg_ts_col_{i}")), &alloc_ts_col)?;
      let alloc_t_plus_r_inv_col_times_t_plus_r_col = alloc_t_plus_r_inv_col.mul(
        cs.namespace(|| format!("t_plus_r_inv_col * t_plus_r_col_{i}")),
        &alloc_t_plus_r_col,
      )?;
      // (t_plus_r_inv_col * t_plus_r_col - ts_col)
      let alloc_term = alloc_t_plus_r_inv_col_times_t_plus_r_col.add(
        cs.namespace(|| format!("t_plus_r_inv_col * t_plus_r_col - ts_col_{i}")),
        &alloc_neg_ts_col,
      )?;
      let alloc_claims_mem_4 = alloc_eq_rho.mul(
        cs.namespace(|| format!("eq_rho * (t_plus_r_inv_col * t_plus_r_col - ts_col_{i})")),
        &alloc_term,
      )?;

      // alloc_claims_mem_5
      let alloc_w_plus_r_inv_col_times_w_plus_r_col = alloc_w_plus_r_inv_col.mul(
        cs.namespace(|| format!("w_plus_r_inv_col * w_plus_r_col_{i}")),
        &alloc_w_plus_r_col,
      )?;
      // (w_plus_r_inv_col * w_plus_r_col - E::Scalar::ONE)
      let alloc_term = alloc_w_plus_r_inv_col_times_w_plus_r_col.add(
        cs.namespace(|| format!("w_plus_r_inv_col * w_plus_r_col - E::Scalar::ONE_{i}")),
        &alloc_neg_one,
      )?;
      let alloc_claims_mem_5 = alloc_eq_rho.mul(
        cs.namespace(|| format!("eq_rho * (w_plus_r_inv_col * w_plus_r_col - E::Scalar::ONE_{i})")),
        &alloc_term,
      )?;

      let alloc_claims_mem = [
        //   t_plus_r_inv_row - w_plus_r_inv_row,
        alloc_claims_mem_0,
        //   t_plus_r_inv_col - w_plus_r_inv_col,
        alloc_claims_mem_1,
        //   eq_rho * (t_plus_r_inv_row * t_plus_r_row - ts_row),
        alloc_claims_mem_2,
        //   eq_rho * (w_plus_r_inv_row * w_plus_r_row - E::Scalar::ONE),
        alloc_claims_mem_3,
        //   eq_rho * (t_plus_r_inv_col * t_plus_r_col - ts_col),
        alloc_claims_mem_4,
        //   eq_rho * (w_plus_r_inv_col * w_plus_r_col - E::Scalar::ONE),
        alloc_claims_mem_5,
      ];

      // let claims_outer = [
      //   eq_tau * (Az * Bz - U.u * Cz - E),
      //   eq_tau * (Az + c * Bz + c * c * Cz),
      // ];

      // claims_outer 1
      // -E
      let alloc_neg_E = alloc_negate(cs.namespace(|| format!("neg_E_{i}")), &alloc_E)?;
      // U.u
      let alloc_U_u = AllocatedNum::alloc(cs.namespace(|| format!("U_u_{i}")), || Ok(U.u))?;

      // U.u * Cz
      let alloc_U_u_times_Cz =
        alloc_U_u.mul(cs.namespace(|| format!("U_u * Cz_{i}")), &alloc_Cz)?;

      // Az * Bz
      let alloc_Az_times_Bz = alloc_Az.mul(cs.namespace(|| format!("Az * Bz_{i}")), &alloc_Bz)?;

      // - (U.u * Cz)
      let alloc_neg_U_u_times_Cz = alloc_negate(
        cs.namespace(|| format!("neg_U_u_times_Cz_{i}")),
        &alloc_U_u_times_Cz,
      )?;

      // Az * Bz - U.u * Cz
      let alloc_term_0 = alloc_Az_times_Bz.add(
        cs.namespace(|| format!("Az * Bz_{i} - U.u * Cz")),
        &alloc_neg_U_u_times_Cz,
      )?;

      // Az * Bz - U.u * Cz - E
      let alloc_term_0 = alloc_term_0.add(
        cs.namespace(|| format!("(Az * Bz - U.u * Cz - E)_{i}")),
        &alloc_neg_E,
      )?;

      let alloc_claims_outer_0 = alloc_eq_tau.mul(
        cs.namespace(|| format!("eq_tau * (Az * Bz - U.u * Cz - E_{i})")),
        &alloc_term_0,
      )?;

      // claims_outer 2
      let alloc_c_times_Bz = alloc_c.mul(cs.namespace(|| format!("c * Bz_{i}")), &alloc_Bz)?;
      let alloc_c_times_Cz = alloc_c.mul(cs.namespace(|| format!("c * Cz_{i}")), &alloc_Cz)?;
      let alloc_c_times_c_times_Cz = alloc_c.mul(
        cs.namespace(|| format!("c * c * Cz_{i}")),
        &alloc_c_times_Cz,
      )?;
      let alloc_term = alloc_Az.add(cs.namespace(|| format!("Az_{i}")), &alloc_c_times_Bz)?;
      let alloc_term = alloc_term.add(
        cs.namespace(|| format!("(Az + c * Bz + c * c * Cz) {i}")),
        &alloc_c_times_c_times_Cz,
      )?;
      let alloc_claims_outer_1 = alloc_eq_tau.mul(
        cs.namespace(|| format!("eq_tau * (Az + c * Bz + c * c * Cz_{i})")),
        &alloc_term,
      )?;

      let alloc_claims_outer = [
        // eq_tau * (Az * Bz - U.u * Cz - E),
        alloc_claims_outer_0,
        // eq_tau * (Az + c * Bz + c * c * Cz),
        alloc_claims_outer_1,
      ];

      // let claims_inner = [L_row * L_col * (val_A + c * val_B + c * c * val_C)];

      // L_row * L_col * (val_A + c * val_B + c * c * val_C)
      let alloc_claims_inner_elem = {
        // c * val_C
        let alloc_c_times_val_C =
          alloc_c.mul(cs.namespace(|| format!("c * val_C_{i}")), &alloc_val_C)?;

        // c * c * val_C
        let alloc_c_times_c_times_val_C = alloc_c.mul(
          cs.namespace(|| format!("c * c * val_C_{i}")),
          &alloc_c_times_val_C,
        )?;

        // c * val_B
        let alloc_c_times_val_B =
          alloc_c.mul(cs.namespace(|| format!("c * val_B_{i}")), &alloc_val_B)?;

        // val_A + c * val_B
        let alloc_term = alloc_val_A.add(
          cs.namespace(|| format!("val_A + c * val_B_{i}")),
          &alloc_c_times_val_B,
        )?;

        // val_A + c * val_B + c * c * val_C
        let alloc_term = alloc_term.add(
          cs.namespace(|| format!("val_A + c * val_B + c * c * val_C_{i}")),
          &alloc_c_times_c_times_val_C,
        )?;

        // L_col * L_row * (val_A + c * val_B + c * c * val_C)
        let res = alloc_L_row.mul(
          cs.namespace(|| format!("L_row * (val_A + c * val_B + c * c * val_C_{i})")),
          &alloc_term,
        )?;

        res.mul(
          cs.namespace(|| format!("L_col * L_row * (val_A + c * val_B + c * c * val_C_{i})")),
          alloc_L_col,
        )?
      };

      let alloc_claims_inner = [alloc_claims_inner_elem];

      // let claims_witness = [eq_masked_tau * W];
      let alloc_claims_witness =
        [alloc_eq_masked_tau.mul(cs.namespace(|| format!("eq_masked_tau * W_{i}")), &alloc_W)?];

      alloc_chained_claims.push(chain![
        alloc_claims_mem,
        alloc_claims_outer,
        alloc_claims_inner,
        alloc_claims_witness
      ]);
    }

    // Mirror of this
    //
    // let claim_sc_final_expected = chained_claims
    // .into_iter()
    // .flatten()
    // .zip_eq(s_powers)
    // .fold(E::Scalar::ZERO, |acc, (claim, s)| acc + s * claim);
    let mut alloc_claim_sc_final_expected = alloc_zero.clone();
    for (i, (claim, s)) in alloc_chained_claims
      .into_iter()
      .flatten()
      .zip_eq(alloc_s_powers)
      .enumerate()
    {
      let claim_times_s = claim.mul(cs.namespace(|| format!("claim * s_{i}")), &s)?;
      alloc_claim_sc_final_expected = alloc_claim_sc_final_expected.add(
        cs.namespace(|| format!("final_claim: claim * s_{i}")),
        &claim_times_s,
      )?;
    }

    // Check:
    //
    // if claim_sc_final_expected != claim_sc_final {
    //   return Err(NovaError::InvalidSumcheckProof);
    // }
    cs.enforce(
      || "enforce claim_sc_final",
      |lc| lc + alloc_claim_sc_final.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + alloc_claim_sc_final_expected.get_variable(),
    );

    Ok(())
  }
}
