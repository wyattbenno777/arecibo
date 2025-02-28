use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_FE_IN_EMULATED_POINT},
  cyclefold::gadgets::emulated::{
    AllocatedEmulR1CSInstance, AllocatedEmulRelaxedR1CSInstance,
  },
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  nebula::nifs::NIFS,
  traits::{
    commitment::CommitmentTrait, CurveCycleEquipped, Dual, Engine, ROCircuitTrait,
    ROConstantsCircuit,
  },
  gadgets::emulated::AllocatedEmulPoint,
  Commitment,
};
/// A gadget for folding group elements.
pub struct FoldGadget {}

impl FoldGadget {
  /// Fold group elements natively.
  pub fn fold_group_elements_native<E: CurveCycleEquipped>(
    U_cmW: Commitment<E>,
    U_cmE: Commitment<E>,
    u_cmW: Commitment<E>,
    cmT: Commitment<E>,
    r: E::Scalar,
  ) -> Result<(Commitment<E>, Commitment<E>), SynthesisError> {
    let cmW = U_cmW + u_cmW * r;
    let cmE = U_cmE + cmT * r;
    Ok((cmW, cmE))
  }

  /// Fold in-circuit field elements.
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
