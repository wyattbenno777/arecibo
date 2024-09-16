use crate::gadgets::{alloc_one, alloc_zero};
use crate::r1cs::{RelaxedR1CSInstance, SparseMatrix};
use crate::spartan::verify_circuit::gadgets::poly::{alloc_powers, AllocatedPowPolynomial};
use crate::spartan::verify_circuit::gadgets::poseidon_transcript::PoseidonTranscriptCircuit;
use crate::traits::snark::DigestHelperTrait;
use crate::{spartan::verify_circuit::ipa_prover_poseidon::batched, traits::Engine};
use bellpepper_core::num::AllocatedNum;
use bellpepper_core::{ConstraintSystem, Namespace, SynthesisError};
use ff::Field;
use generic_array::typenum::U24;
use poseidon_sponge::sponge::circuit::SpongeCircuit;
use poseidon_sponge::sponge::vanilla::SpongeTrait;
use poseidon_sponge::{poseidon::PoseidonConstants, sponge::vanilla::Sponge, Strength};

use crate::gadgets::alloc_negate;
use crate::spartan::math::Math;
use crate::spartan::verify_circuit::gadgets::poly::AllocMultilinearPolynomial;
use crate::spartan::verify_circuit::gadgets::poly::AllocatedEqPolynomial;

use crate::spartan::verify_circuit::gadgets::poly::AllocatedSparsePolynomial;
use crate::spartan::verify_circuit::gadgets::sumcheck::SCVerifyBatchedGadget;
use itertools::Itertools as _;
use std::iter;
use std::marker::PhantomData;

type SNARK<E> = batched::BatchedRelaxedR1CSSNARK<E>;
type VK<E> = batched::VerifierKey<E>;

#[cfg(test)]
mod tests;

pub struct SpartanVerifyCircuit<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> SpartanVerifyCircuit<E> {
  /// Instantiate the Spartan Poseidon transcript circuit and its required sponge function
  fn sponge_transcript_circuit<'a, CS: ConstraintSystem<E::Scalar>>(
    ns: &mut Namespace<'a, E::Scalar, CS>,
    constants: &'a PoseidonConstants<E::Scalar, U24>,
  ) -> (
    PoseidonTranscriptCircuit<E>,
    SpongeCircuit<'a, E::Scalar, U24, CS>,
  ) {
    let mut sponge = SpongeCircuit::<E::Scalar, U24, _>::new_with_constants(
      constants,
      poseidon_sponge::sponge::vanilla::Mode::Simplex,
    );

    let transcript_circuit = PoseidonTranscriptCircuit::<E>::new(&mut sponge, ns);

    (transcript_circuit, sponge)
  }

  /// Run circuit through constraint system to build R1CS shapes, instance and witness
  pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    vk: &VK<E>,
    U: &[RelaxedR1CSInstance<E>],
    snark: &SNARK<E>,
  ) -> Result<(), SynthesisError> {
    // Allocated constants that will be used throughout circuit for different computations that require a an allocated zero or one variable
    let alloc_one = alloc_one(cs.namespace(|| "one"));
    let alloc_zero = alloc_zero(cs.namespace(|| "zero"));

    let (num_rounds_x, num_rounds_y): (Vec<_>, Vec<_>) = vk
      .S
      .iter()
      .map(|s| (s.num_cons.log_2(), s.num_vars.log_2() + 1))
      .unzip();
    let num_rounds_x_max = *num_rounds_x.iter().max().unwrap();
    let num_rounds_y_max = *num_rounds_y.iter().max().unwrap();
    let num_instances = U.len();

    // Instantiate transcript circuit
    let constants = Sponge::<E::Scalar, U24>::api_constants(Strength::Standard);
    let mut cs = cs.namespace(|| "ns");
    let (mut transcript, mut sponge) = Self::sponge_transcript_circuit(&mut cs, &constants);

    let alloc_vk_digest = AllocatedNum::alloc(cs.namespace(|| "vk_digest"), || Ok(vk.digest()))?;
    transcript.absorb(alloc_vk_digest, &mut sponge, &mut cs);

    let alloc_tau = transcript.squeeze(String::from("tau"), &mut sponge, &mut cs)?;
    let alloc_all_taus =
      AllocatedPowPolynomial::squares(cs.namespace(|| "all_taus"), &alloc_tau, num_rounds_x_max)?;

    let mut alloc_polys_tau = Vec::new();

    for (i, num_rounds_x) in num_rounds_x.iter().enumerate() {
      let alloc_pow_poly_tau = AllocatedPowPolynomial::evals_with_powers(
        cs.namespace(|| format!("polys_tau_{}", i)),
        &alloc_all_taus,
        *num_rounds_x,
      )?;

      let alloc_poly_tau = AllocMultilinearPolynomial::new(alloc_pow_poly_tau);
      alloc_polys_tau.push(alloc_poly_tau);
    }

    let alloc_outer_r = transcript.squeeze(String::from("outer_r"), &mut sponge, &mut cs)?;

    let alloc_outer_r_powers = alloc_powers(
      cs.namespace(|| "outer_r_powers"),
      &alloc_outer_r,
      num_instances,
    )?;

    let (alloc_claim_outer_final, alloc_r_x) = {
      let alloc_sc = SCVerifyBatchedGadget::new(snark.sc_proof_outer.clone());
      let alloc_degree_bound = AllocatedNum::alloc(cs.namespace(|| "degree_bound"), || {
        Ok(E::Scalar::from(3_u64))
      })?;

      let alloc_claims = vec![alloc_zero.clone(); num_instances];

      alloc_sc.verify_batched(
        &mut cs,
        &alloc_claims,
        &num_rounds_x,
        &alloc_outer_r_powers,
        &alloc_degree_bound,
        &mut transcript,
        &mut sponge,
        "outer sc",
      )?
    };

    let alloc_r_x = num_rounds_x
      .iter()
      .map(|num_rounds| alloc_r_x[(num_rounds_x_max - num_rounds)..].to_vec())
      .collect::<Vec<_>>();

    // Extract evaluations into a vector [(Azᵢ, Bzᵢ, Czᵢ, Eᵢ)]
    let ABCE_evals = || snark.claims_outer.iter().zip_eq(snark.evals_E.iter());

    let alloc_ABCE_evals = ABCE_evals()
      .enumerate()
      .map(|(i, ((claim_Az, claim_Bz, claim_Cz), eval_E))| {
        (
          (
            AllocatedNum::alloc(cs.namespace(|| format!("claim_Az_{i}")), || Ok(*claim_Az))
              .expect("Allocated variable into cs"),
            AllocatedNum::alloc(cs.namespace(|| format!("claim_Bz_{i}")), || Ok(*claim_Bz))
              .expect("Allocated variable into cs"),
            AllocatedNum::alloc(cs.namespace(|| format!("claim_Cz_{i}")), || Ok(*claim_Cz))
              .expect("Allocated variable into cs"),
          ),
          AllocatedNum::alloc(cs.namespace(|| format!("eval_E_{i}")), || Ok(*eval_E))
            .expect("Allocated variable into cs"),
        )
      })
      .collect::<Vec<_>>();

    let alloc_chis_r_x = alloc_r_x
      .iter()
      .enumerate()
      .map(|(i, r_x)| {
        AllocatedEqPolynomial::evals_from_points(cs.namespace(|| format!("chis_r_x_{i}")), r_x)
      })
      .collect::<Result<Vec<_>, _>>()?;

    let mut alloc_evals_tau = Vec::new();

    for (i, (alloc_poly_tau, alloc_er_x)) in alloc_polys_tau
      .iter()
      .zip_eq(alloc_chis_r_x.iter())
      .enumerate()
    {
      let alloc_eval_tau = AllocMultilinearPolynomial::evaluate_with_chis(
        cs.namespace(|| format!("eval_tau_{}", i)),
        alloc_poly_tau.evaluations(),
        &alloc_er_x,
      )?;

      alloc_evals_tau.push(alloc_eval_tau);
    }

    let alloc_claim_outer_final_expected = {
      let mut final_claim = alloc_zero.clone();
      for (i, (((alloc_ABCE_eval, u), alloc_eval_tau), alloc_r)) in alloc_ABCE_evals
        .iter()
        .zip_eq(U.iter())
        .zip_eq(alloc_evals_tau.iter())
        .zip_eq(alloc_outer_r_powers.iter())
        .enumerate()
      {
        let ((alloc_claim_Az, alloc_claim_Bz, alloc_claim_Cz), alloc_eval_E) = alloc_ABCE_eval;
        let alloc_U_u = AllocatedNum::alloc(cs.namespace(|| format!("U_u_{}", i)), || Ok(u.u))?;

        // claim_Az * claim_Bz
        let alloc_AzBz =
          alloc_claim_Az.mul(cs.namespace(|| format!("AzBz_{}", i)), alloc_claim_Bz)?;

        // u.u * claim_Cz
        let alloc_uCz = alloc_U_u.mul(cs.namespace(|| format!("uCz_{}", i)), alloc_claim_Cz)?;

        // - u.u * claim_Cz
        let neg_alloc_uCz =
          alloc_negate(cs.namespace(|| format!("- u.u * claim_Cz {i}")), &alloc_uCz)?;

        // claim_Az * claim_Bz - u.u * claim_Cz
        let alloc_AzBz_uCz =
          alloc_AzBz.add(cs.namespace(|| format!("AzBz_uCz_{}", i)), &neg_alloc_uCz)?;

        // - eval_E
        let neg_alloc_eval_E =
          alloc_negate(cs.namespace(|| format!("- eval_E {i}")), alloc_eval_E)?;

        // claim_Az * claim_Bz - u.u * claim_Cz - eval_E
        let alloc_AzBz_uCz_E = alloc_AzBz_uCz.add(
          cs.namespace(|| format!("AzBz_uCz_E_{}", i)),
          &neg_alloc_eval_E,
        )?;

        // r * eval_tau
        let alloc_r_eval_tau = alloc_r.mul(
          cs.namespace(|| format!("r_eval_tau_{}", i)),
          &alloc_eval_tau,
        )?;

        // r * eval_tau * (claim_Az * claim_Bz - u.u * claim_Cz - eval_E)
        let alloc_claim =
          alloc_r_eval_tau.mul(cs.namespace(|| format!("claim_{}", i)), &alloc_AzBz_uCz_E)?;

        final_claim =
          final_claim.add(cs.namespace(|| format!("final_claim_{}", i)), &alloc_claim)?;
      }

      final_claim
    };

    // Enforce the following check:
    //
    // if claim_outer_final != claim_outer_final_expected {
    //   return Err(NovaError::InvalidSumcheckProof);
    // }
    cs.enforce(
      || "claim_outer_final_expected == alloc_claim_outer_final_expected",
      |lc| lc + alloc_claim_outer_final_expected.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + alloc_claim_outer_final.get_variable(),
    );

    let alloc_inner_r = transcript.squeeze(String::from("inner_r"), &mut sponge, &mut cs)?;
    let alloc_inner_r_square = alloc_inner_r.square(cs.namespace(|| "inner_r_square"))?;
    let alloc_inner_r_cube =
      alloc_inner_r_square.mul(cs.namespace(|| "inner_r_cube"), &alloc_inner_r)?;
    let alloc_inner_r_powers = alloc_powers(
      cs.namespace(|| "inner_r_powers"),
      &alloc_inner_r_cube,
      num_instances,
    )?;

    let mut alloc_claims_inner = Vec::new();

    for (i, ((claim_Az, claim_Bz, claim_Cz), _)) in alloc_ABCE_evals.iter().enumerate() {
      // inner_r_square * claim_Cz
      let alloc_inner_r_square_Cz = alloc_inner_r_square.mul(
        cs.namespace(|| format!("inner_r_square_Cz_{}", i)),
        claim_Cz,
      )?;

      // inner_r * claim_Bz
      let alloc_inner_r_Bz =
        alloc_inner_r.mul(cs.namespace(|| format!("inner_r_Bz_{}", i)), claim_Bz)?;

      // inner_r * claim_Bz + inner_r_square * claim_Cz
      let alloc_inner_r_Bz_inner_r_square_Cz = alloc_inner_r_Bz.add(
        cs.namespace(|| format!("inner_r_Bz_inner_r_square_Cz_{}", i)),
        &alloc_inner_r_square_Cz,
      )?;

      // claim_Az + inner_r * claim_Bz + inner_r_square * claim_Cz
      let claim_inner = claim_Az.add(
        cs.namespace(|| format!("claim_inner_{}", i)),
        &alloc_inner_r_Bz_inner_r_square_Cz,
      )?;

      alloc_claims_inner.push(claim_inner);
    }

    let (alloc_claim_inner_final, alloc_r_y) = {
      let alloc_inner_degree_bound =
        AllocatedNum::alloc(cs.namespace(|| "inner_degree_bound"), || {
          Ok(E::Scalar::from(2_u64))
        })?;
      let alloc_sc = SCVerifyBatchedGadget::new(snark.sc_proof_inner.clone());

      alloc_sc.verify_batched(
        &mut cs,
        &alloc_claims_inner,
        &num_rounds_y,
        &alloc_inner_r_powers,
        &alloc_inner_degree_bound,
        &mut transcript,
        &mut sponge,
        "inner cs",
      )?
    };

    let alloc_r_y: Vec<Vec<AllocatedNum<E::Scalar>>> = num_rounds_y
      .iter()
      .map(|num_rounds| alloc_r_y[(num_rounds_y_max - num_rounds)..].to_vec())
      .collect();

    let alloc_evals_W = snark
      .evals_W
      .iter()
      .enumerate()
      .map(|(i, eval)| AllocatedNum::alloc(cs.namespace(|| format!("eval_W_{}", i)), || Ok(*eval)))
      .collect::<Result<Vec<_>, _>>()?;

    let mut alloc_evals_Z = Vec::new();

    for (i, ((alloc_eval_W, U), alloc_r_y)) in alloc_evals_W
      .iter()
      .zip_eq(U.iter())
      .zip_eq(alloc_r_y.iter())
      .enumerate()
    {
      let alloc_eval_X = {
        let alloc_poly_X = {
          let poly_X: Vec<E::Scalar> = iter::once(U.u).chain(U.X.iter().cloned()).collect();
          poly_X
            .iter()
            .enumerate()
            .map(|(j, x)| AllocatedNum::alloc(cs.namespace(|| format!("X_{}_{}", i, j)), || Ok(*x)))
            .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?
        };

        AllocatedSparsePolynomial::new(alloc_r_y.len() - 1, alloc_poly_X)
          .evaluate(cs.namespace(|| format!("eval_X_{}", i)), &alloc_r_y[1..])?
      };

      // r_y[0] * eval_X
      let alloc_r_y0_eval_X =
        alloc_r_y[0].mul(cs.namespace(|| format!("r_y0_eval_X_{}", i)), &alloc_eval_X)?;

      //  - r_y[0]
      let neg_alloc_r_y0 = alloc_negate(cs.namespace(|| format!("- r_y0_{}", i)), &alloc_r_y[0])?;

      // 1 - r_y[0]
      let alloc_1_r_y0 =
        alloc_one.add(cs.namespace(|| format!("1 - r_y0_{}", i)), &neg_alloc_r_y0)?;

      // (E::Scalar::ONE - r_y[0]) * eval_W
      let alloc_1_r_y0_eval_W = alloc_1_r_y0.mul(
        cs.namespace(|| format!("1 - r_y0_eval_W_{}", i)),
        alloc_eval_W,
      )?;

      // (E::Scalar::ONE - r_y[0]) * eval_W + r_y[0] * eval_X
      let alloc_eval_Z =
        alloc_1_r_y0_eval_W.add(cs.namespace(|| format!("eval_Z_{}", i)), &alloc_r_y0_eval_X)?;

      alloc_evals_Z.push(alloc_eval_Z);
    }

    let mut alloc_claim_inner_final_expected = alloc_zero.clone();

    for (i, ((((S, alloc_r_x), alloc_r_y), alloc_eval_Z), alloc_r_i)) in vk
      .S
      .iter()
      .zip_eq(alloc_chis_r_x.iter())
      .zip_eq(alloc_r_y.iter())
      .zip_eq(alloc_evals_Z.iter())
      .zip_eq(alloc_inner_r_powers.iter())
      .enumerate()
    {
      let alloc_evals = alloc_multi_evaluate::<_, E>(
        cs.namespace(|| format!("multi eval {i}")),
        &[&S.A, &S.B, &S.C],
        &alloc_r_x,
        &alloc_r_y,
      )?;

      // inner_r_square * evals[2]
      let alloc_inner_r_square_evals_2 = alloc_inner_r_square.mul(
        cs.namespace(|| format!("inner_r_square_evals_2_{}", i)),
        &alloc_evals[2],
      )?;

      // inner_r * evals[1]
      let alloc_inner_r_evals_1 = alloc_inner_r.mul(
        cs.namespace(|| format!("inner_r_evals_1_{}", i)),
        &alloc_evals[1],
      )?;

      // inner_r * evals[1] + inner_r_square * evals[2]
      let alloc_inner_r_evals_1_inner_r_square_evals_2 = alloc_inner_r_evals_1.add(
        cs.namespace(|| format!("inner_r_evals_1_inner_r_square_evals_2_{}", i)),
        &alloc_inner_r_square_evals_2,
      )?;

      // evals[0] + inner_r * evals[1] + inner_r_square * evals[2]
      let alloc_eval = alloc_evals[0].add(
        cs.namespace(|| format!("eval_{}", i)),
        &alloc_inner_r_evals_1_inner_r_square_evals_2,
      )?;

      // eval * r_i
      let alloc_eval_r_i = alloc_eval.mul(cs.namespace(|| format!("eval_r_i_{}", i)), alloc_r_i)?;

      // eval * r_i * eval_Z
      let alloc_claim = alloc_eval_r_i.mul(
        cs.namespace(|| format!("final inner claim_{}", i)),
        alloc_eval_Z,
      )?;

      alloc_claim_inner_final_expected = alloc_claim_inner_final_expected.add(
        cs.namespace(|| format!("claim_inner_final_expected_{}", i)),
        &alloc_claim,
      )?;
    }

    // Enforce this check:
    //
    // if claim_inner_final != claim_inner_final_expected {
    //   return Err(NovaError::InvalidSumcheckProof);
    // }

    cs.enforce(
      || "claim_inner_final_expected == alloc_claim_inner_final_expected",
      |lc| lc + alloc_claim_inner_final_expected.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + alloc_claim_inner_final.get_variable(),
    );

    Ok(())
  }
}

fn alloc_multi_evaluate<CS: ConstraintSystem<E::Scalar>, E: Engine>(
  mut cs: CS,
  M_vec: &[&SparseMatrix<E::Scalar>],
  chi_r_x: &[AllocatedNum<E::Scalar>],
  r_y: &[AllocatedNum<E::Scalar>],
) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
  let T_x = chi_r_x;

  let T_y =
    AllocatedEqPolynomial::evals_from_points(cs.namespace(|| "eq poly eval from points"), r_y)?;

  M_vec
    .iter()
    .enumerate()
    .map(|(i, &M_vec)| {
      alloc_evaluate_with_table::<_, E>(
        cs.namespace(|| format!("evaluate with table {i}")),
        M_vec,
        T_x,
        &T_y,
      )
    })
    .collect()
}

fn alloc_evaluate_with_table<CS: ConstraintSystem<E::Scalar>, E: Engine>(
  mut cs: CS,
  M: &SparseMatrix<E::Scalar>,
  T_x: &[AllocatedNum<E::Scalar>],
  T_y: &[AllocatedNum<E::Scalar>],
) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
  let alloc_zero = AllocatedNum::alloc(cs.namespace(|| "zero"), || Ok(E::Scalar::ZERO))?;

  let outer_vals = M
    .iter_rows()
    .enumerate()
    .map(|(row_idx, row)| {
      let inner_vals = M
        .get_row(row)
        .enumerate()
        .map(|(i, (val, col_idx))| {
          // T_x[row_idx] * T_y[*col_idx]
          let alloc_T_row_col = T_x[row_idx].mul(
            cs.namespace(|| format!("[{}][{}] * {}", row_idx, i, col_idx)),
            &T_y[*col_idx],
          )?;

          let val = &AllocatedNum::alloc(
            cs.namespace(|| format!("val_{}_{}", row_idx, col_idx)),
            || Ok(*val),
          )?;

          // T_x[row_idx] * T_y[*col_idx] * val
          alloc_T_row_col.mul(
            cs.namespace(|| format!("[{}][{}] * val", row_idx, col_idx)),
            val,
          )
        })
        .collect::<Result<Vec<_>, _>>()?;

      let mut inner_res = alloc_zero.clone();

      for (i, val) in inner_vals.iter().enumerate() {
        inner_res = inner_res.add(cs.namespace(|| format!("inner_res_{}_{}", row_idx, i)), val)?;
      }

      Ok(inner_res)
    })
    .collect::<Result<Vec<_>, SynthesisError>>()?;

  let mut outer_res = alloc_zero.clone();

  for (i, val) in outer_vals.iter().enumerate() {
    outer_res = outer_res.add(cs.namespace(|| format!("outer_res_{}", i)), val)?;
  }

  Ok(outer_res)
}
