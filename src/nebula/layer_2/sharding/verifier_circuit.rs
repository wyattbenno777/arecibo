use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_CHALLENGE_BITS, NUM_FE_IN_EMULATED_POINT},
  cyclefold::gadgets::{emulated::AllocatedEmulRelaxedR1CSInstance, AllocatedCycleFoldInstance},
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{emulated::AllocatedEmulPoint, le_bits_to_num},
  nebula::{
    augmented_circuit::AugmentedCircuitParams,
    layer_2::{
      gadgets::{
        r1cs::AllocatedRelaxedR1CSInstanceBn, CycleFoldNIFSVerifierGadget,
        CycleFoldRelaxedNIFSVerifierGadget, NIFSVerifierGadget, PrimaryNIFSVerifierGadget,
      },
      utils::Layer2FoldingData,
    },
    rs::StepCircuit,
  },
  r1cs::RelaxedR1CSInstance,
  traits::{
    commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROCircuitTrait,
    ROConstantsCircuit,
  },
};
use ff::Field;

#[derive(Clone)]
pub struct VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  folding_data_F: Option<Layer2FoldingData<E>>,
  folding_data_ops: Option<Layer2FoldingData<E>>,
  folding_data_scan: Option<Layer2FoldingData<E>>,
  U1_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
  C_IS: Option<E::Scalar>,
  C_FS: Option<E::Scalar>,
  params: AugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<Dual<E>>,
}

impl<E> StepCircuit<E::Scalar> for VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  fn arity(&self) -> usize {
    2
  }
  // TODO: complete all the checks in specified in Nebula 4.4.3
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    //  check that Ci =? CIS // finalized proof starts with previous memory
    let C_i = z[0].clone();
    let C_IS = AllocatedNum::alloc(cs.namespace(|| "prev_IC"), || {
      Ok(self.C_IS.unwrap_or(E::Scalar::ZERO))
    })?;
    let C_FS = AllocatedNum::alloc(cs.namespace(|| "prev_IC"), || {
      Ok(self.C_FS.unwrap_or(E::Scalar::ZERO))
    })?;
    cs.enforce(
      || "C_i = C_IS",
      |lc| lc + C_i.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + C_IS.get_variable(),
    );

    // Get running CycleFold instance
    let U1_secondary = self.alloc_cyclefold_running_instance(cs.namespace(|| "U1_secondary"))?;

    // alloc witness for F
    let (pp_digest_F, U1_F, U2_F, E_new_F, W_new_F, U2_secondary_F, nifs_F) =
      VerifierCircuit::alloc_folding_data(
        cs.namespace(|| "alloc folding data F"),
        &self.params,
        &self.folding_data_F,
      )?;

    // alloc witness for ops
    let (pp_digest_ops, U1_ops, U2_ops, E_new_ops, W_new_ops, U2_secondary_ops, nifs_ops) =
      VerifierCircuit::alloc_folding_data(
        cs.namespace(|| "alloc folding data ops"),
        &self.params,
        &self.folding_data_ops,
      )?;

    // alloc witness for scan
    let (pp_digest_scan, U1_scan, U2_scan, E_new_scan, W_new_scan, U2_secondary_scan, nifs_scan) =
      VerifierCircuit::alloc_folding_data(
        cs.namespace(|| "alloc folding data_scan"),
        &self.params,
        &self.folding_data_scan,
      )?;

    // i/o hash check
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      3 * (2 * NUM_FE_IN_EMULATED_POINT + 3) + (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS), // 3 * (U.W + U.comm_E + U.X + U.u) + U_cyclefold
    );
    U1_F.absorb_in_ro(cs.namespace(|| "absorb U1_F"), &mut ro)?;
    U1_ops.absorb_in_ro(cs.namespace(|| "absorb U1_ops"), &mut ro)?;
    U1_scan.absorb_in_ro(cs.namespace(|| "absorb U1_scan"), &mut ro)?;
    U1_secondary.absorb_in_ro(cs.namespace(|| "absorb U1_secondary"), &mut ro)?;
    let hash_U_bits = ro.squeeze(cs.namespace(|| "hash_U bits"), NUM_CHALLENGE_BITS)?;
    let hash_U = le_bits_to_num(cs.namespace(|| "hash_U"), &hash_U_bits)?;
    let expected_hash_U = z[1].clone();
    cs.enforce(
      || "hash_U == z0",
      |lc| lc + hash_U.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + expected_hash_U.get_variable(),
    );

    // NIFS.V for F
    let (U_F, U_secondary_temp_1) = nifs_F.verify(
      cs.namespace(|| "nifs_F"),
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
      &U1_F,
      &U2_F,
      &U1_secondary,
      &U2_secondary_F,
      &pp_digest_F,
      E_new_F,
      W_new_F,
    )?;

    // NIFS.V for ops
    let (U_ops, U_secondary_temp_2) = nifs_ops.verify(
      cs.namespace(|| "nifs_ops"),
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
      &U1_ops,
      &U2_ops,
      &U_secondary_temp_1,
      &U2_secondary_ops,
      &pp_digest_ops,
      E_new_ops,
      W_new_ops,
    )?;

    // NIFS.V for scan
    let (U_scan, U_secondary) = nifs_scan.verify(
      cs.namespace(|| "nifs_scan"),
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
      &U1_scan,
      &U2_scan,
      &U_secondary_temp_2,
      &U2_secondary_scan,
      &pp_digest_scan,
      E_new_scan,
      W_new_scan,
    )?;

    // output hash
    let mut ro = <Dual<E> as Engine>::ROCircuit::new(
      self.ro_consts.clone(),
      3 * (2 * NUM_FE_IN_EMULATED_POINT + 3) + (3 + 3 + BN_N_LIMBS + NIO_CYCLE_FOLD * BN_N_LIMBS), // (U.W + U.comm_E + U.X + U.u) + U_cyclefold
    );
    U_F.absorb_in_ro(cs.namespace(|| "absorb folded U_F"), &mut ro)?;
    U_ops.absorb_in_ro(cs.namespace(|| "absorb folded U_ops"), &mut ro)?;
    U_scan.absorb_in_ro(cs.namespace(|| "absorb folded U_scan"), &mut ro)?;
    U_secondary.absorb_in_ro(cs.namespace(|| "absorb folded U_secondary"), &mut ro)?;
    let hash_U_bits = ro.squeeze(cs.namespace(|| "hash_folded_U bits"), NUM_CHALLENGE_BITS)?;
    let hash_U = le_bits_to_num(cs.namespace(|| "hash_folded_U"), &hash_U_bits)?;
    Ok(vec![C_FS, hash_U])
  }

  fn non_deterministic_advice(&self) -> Vec<E::Scalar> {
    vec![]
  }
}

impl<E> VerifierCircuit<E>
where
  E: CurveCycleEquipped,
{
  pub fn new(
    params: AugmentedCircuitParams,
    ro_consts: ROConstantsCircuit<Dual<E>>,
    folding_data_F: Option<Layer2FoldingData<E>>,
    folding_data_ops: Option<Layer2FoldingData<E>>,
    folding_data_scan: Option<Layer2FoldingData<E>>,
    U1_secondary: Option<RelaxedR1CSInstance<Dual<E>>>,
    C_IS: Option<E::Scalar>,
    C_FS: Option<E::Scalar>,
  ) -> Self {
    Self {
      params,
      ro_consts,
      folding_data_F,
      folding_data_ops,
      folding_data_scan,
      U1_secondary,
      C_IS,
      C_FS,
    }
  }

  fn alloc_cyclefold_running_instance<CS>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    AllocatedRelaxedR1CSInstanceBn::alloc(
      cs.namespace(|| "U1_secondary"),
      self.U1_secondary.as_ref(),
      self.params.limb_width,
      self.params.n_limbs,
    )
  }
  fn alloc_folding_data<CS>(
    mut cs: CS,
    params: &AugmentedCircuitParams,
    folding_data: &Option<Layer2FoldingData<E>>,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,                                 // pp_digest
      AllocatedEmulRelaxedR1CSInstance<Dual<E>>,               // U1
      AllocatedEmulRelaxedR1CSInstance<Dual<E>>,               // U2
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,             // E_new
      AllocatedEmulPoint<<Dual<E> as Engine>::GE>,             // W_new
      AllocatedRelaxedR1CSInstanceBn<Dual<E>, NIO_CYCLE_FOLD>, // U2_secondary
      NIFSVerifierGadget<E>,                                   // nifs
    ),
    SynthesisError,
  >
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    // Primary folding data
    let pp_digest = AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || {
      Ok(
        folding_data
          .as_ref()
          .and_then(|data| data.pp_digest)
          .map_or(E::Scalar::ZERO, |pp_digest| pp_digest),
      )
    })?;
    let U1 = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      folding_data.as_ref().and_then(|data| data.U1.as_ref()),
      params.limb_width,
      params.n_limbs,
    )?;
    let U2 = AllocatedEmulRelaxedR1CSInstance::alloc(
      cs.namespace(|| "allocate U"),
      folding_data.as_ref().and_then(|data| data.U2.as_ref()),
      params.limb_width,
      params.n_limbs,
    )?;
    let nifs_primary = PrimaryNIFSVerifierGadget::alloc(
      cs.namespace(|| "primary_nifs"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_primary),
      params.limb_width,
      params.n_limbs,
    )?;
    let E_new = AllocatedEmulPoint::alloc(
      cs.namespace(|| "E_new"),
      folding_data
        .as_ref()
        .and_then(|data| data.E_new)
        .map(|E_new| E_new.to_coordinates()),
      params.limb_width,
      params.n_limbs,
    )?;
    let W_new = AllocatedEmulPoint::alloc(
      cs.namespace(|| "W_new"),
      folding_data
        .as_ref()
        .and_then(|data| data.W_new)
        .map(|W_new| W_new.to_coordinates()),
      params.limb_width,
      params.n_limbs,
    )?;

    // First CycleFold data
    let nifs_E1 = CycleFoldNIFSVerifierGadget::alloc(
      cs.namespace(|| "nifs_E1"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_E1),
    )?;
    let l_u_cyclefold_E1 = AllocatedCycleFoldInstance::alloc(
      cs.namespace(|| "l_u_cyclefold_E1"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.l_u_cyclefold_E1),
      params.limb_width,
      params.n_limbs,
    )?;

    // Second CycleFold data
    let nifs_E2 = CycleFoldNIFSVerifierGadget::alloc(
      cs.namespace(|| "nifs_E2"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_E2),
    )?;
    let l_u_cyclefold_E2 = AllocatedCycleFoldInstance::alloc(
      cs.namespace(|| "l_u_cyclefold_E2"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.l_u_cyclefold_E2),
      params.limb_width,
      params.n_limbs,
    )?;

    // Third CycleFold data
    let nifs_W = CycleFoldNIFSVerifierGadget::alloc(
      cs.namespace(|| "nifs_W"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_W),
    )?;
    let l_u_cyclefold_W = AllocatedCycleFoldInstance::alloc(
      cs.namespace(|| "l_u_cyclefold_W"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.l_u_cyclefold_W),
      params.limb_width,
      params.n_limbs,
    )?;

    // fourth CycleFold data
    let U2_secondary = AllocatedRelaxedR1CSInstanceBn::alloc(
      cs.namespace(|| "U2_secondary"),
      folding_data
        .as_ref()
        .and_then(|data| data.U2_secondary.as_ref()),
      params.limb_width,
      params.n_limbs,
    )?;
    let nifs_final_cyclefold = CycleFoldRelaxedNIFSVerifierGadget::alloc(
      cs.namespace(|| "nifs_final_cyclefold"),
      folding_data
        .as_ref()
        .and_then(|data| data.nifs.as_ref())
        .map(|nifs| &nifs.nifs_final_cyclefold),
    )?;
    let nifs = NIFSVerifierGadget {
      nifs_primary,
      nifs_E1,
      nifs_E2,
      nifs_W,
      nifs_final_cyclefold,
      l_u_cyclefold_E1,
      l_u_cyclefold_E2,
      l_u_cyclefold_W,
    };
    Ok((pp_digest, U1, U2, E_new, W_new, U2_secondary, nifs))
  }
}
