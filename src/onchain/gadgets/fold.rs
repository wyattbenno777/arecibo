use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_FE_IN_EMULATED_POINT},
  cyclefold::gadgets::emulated::{
    AllocatedEmulPoint, AllocatedEmulR1CSInstance, AllocatedEmulRelaxedR1CSInstance,
  },
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  nebula::nifs::NIFS,
  provider::kzg_commitment::UVKZGCommitment,
  traits::{
    commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROCircuitTrait,
    ROConstantsCircuit,
  },
};
use group::Curve;
use pairing::Engine as PairingEngine;
pub struct FoldGadget {}

impl FoldGadget {
  pub fn fold_group_elements_native<E: PairingEngine>(
    U_commitments: (UVKZGCommitment<E>, UVKZGCommitment<E>),
    u_commitments: UVKZGCommitment<E>,
    cmT: E::G1,
    r: E::Fr,
  ) -> Result<(UVKZGCommitment<E>, UVKZGCommitment<E>), SynthesisError> {
    let U_cmW = U_commitments.0;
    let U_cmE = U_commitments.1;
    let u_cmW = u_commitments;

    let cmW = E::G1::from(U_cmW.0) + E::G1::from(u_cmW.0) * r;
    let cmE = E::G1::from(U_cmE.0) + cmT * r;

    Ok((
      UVKZGCommitment::<E>::new(cmW.to_affine()),
      UVKZGCommitment::<E>::new(cmE.to_affine()),
    ))
  }

  pub fn fold_field_elements_gadget<CS, E: CurveCycleEquipped>(
    cs: &mut CS,
    pp_hash: AllocatedNum<<E as Engine>::Scalar>,
    U: AllocatedEmulRelaxedR1CSInstance<Dual<E>>,
    u: AllocatedEmulR1CSInstance<Dual<E>>,
    nifs_proof: NIFS<E>,
    r: AllocatedNum<<E as Engine>::Scalar>,
  ) -> Result<AllocatedEmulRelaxedR1CSInstance<Dual<E>>, SynthesisError>
  where
    CS: ConstraintSystem<<E as Engine>::Scalar>,
  {
    let mut ro_circuit = <Dual<E> as Engine>::ROCircuit::new(
      ROConstantsCircuit::<Dual<E>>::default(),
      1 + NUM_FE_IN_EMULATED_POINT + 2 + NUM_FE_IN_EMULATED_POINT, // pp_digest + u.W + u.X + T
    );
    ro_circuit.absorb(&pp_hash);
    u.absorb_in_ro(cs.namespace(|| "u"), &mut ro_circuit)?;
    let cm_T: AllocatedEmulPoint<<Dual<E> as Engine>::GE> = AllocatedEmulPoint::alloc(
      cs.namespace(|| "cm_T"),
      Some(nifs_proof.nifs_primary.comm_T.to_coordinates()),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;
    cm_T.absorb_in_ro(cs.namespace(|| "cm_T"), &mut ro_circuit)?;

    // TODO: Make this constraint work
    // let r_1_bits = ro_circuit.squeeze(cs.namespace(|| "squeeze_r_1"), NUM_HASH_BITS)?;
    // let r_1 = le_bits_to_num(cs.namespace(|| "bits_to_num r_1"), &r_1_bits)?;

    // cs.enforce(
    //   || "r_1 is a valid scalar",
    //   |lc| lc,
    //   |lc| lc,
    //   |lc| lc + r.get_variable() - r_1.get_variable(),
    // );

    let r_mul_x0 = r.mul(cs.namespace(|| "mul x0"), &u.x0)?;
    let x0 = U.x0.add(cs.namespace(|| "add x0"), &r_mul_x0)?;
    let r_mul_x1 = r.mul(cs.namespace(|| "mul x1"), &u.x1)?;
    let x1 = U.x1.add(cs.namespace(|| "add x1"), &r_mul_x1)?;
    let u = U.u.add(cs.namespace(|| "add u"), &r)?;
    let folded_U = AllocatedEmulRelaxedR1CSInstance {
      comm_W: AllocatedEmulPoint::default(cs.namespace(|| "comm_W"), BN_LIMB_WIDTH, BN_N_LIMBS)?,
      comm_E: AllocatedEmulPoint::default(cs.namespace(|| "comm_E"), BN_LIMB_WIDTH, BN_N_LIMBS)?,
      u,
      x0,
      x1,
    };
    Ok(folded_U)
  }
}
