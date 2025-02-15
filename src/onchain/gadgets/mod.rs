use crate::frontend::SynthesisError;
use crate::traits::CurveCycleEquipped;
use crate::Commitment;

pub mod domain;
pub mod eval;
pub mod hash;
pub mod kzg;

pub use domain::*;
pub use eval::*;
pub use hash::*;
pub use kzg::*;
pub struct DeciderNovaGadget {}

impl DeciderNovaGadget {
  pub fn fold_group_elements_native<E: CurveCycleEquipped>(
    U_commitments: (Commitment<E>, Commitment<E>),
    u_commitments: Commitment<E>,
    // cmT: Commitment<E>,
    r: E::Scalar,
  ) -> Result<(Commitment<E>, Commitment<E>), SynthesisError> {
    let U_cmW = U_commitments.0;
    let U_cmE = U_commitments.1;
    let u_cmW = u_commitments;
    // *comm_E_1 + *comm_T * *r;
    let cmW = U_cmW + u_cmW * r;
    let cmE = U_cmE; // + cmT * r;

    Ok((cmW, cmE))
  }
}
