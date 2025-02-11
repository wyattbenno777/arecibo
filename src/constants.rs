//! Global Nova constants

pub(crate) const NUM_CHALLENGE_BITS: usize = 128;
pub(crate) const BN_LIMB_WIDTH: usize = 64;
pub(crate) const BN_N_LIMBS: usize = 4;
pub(crate) const NUM_FE_IN_EMULATED_POINT: usize = 2 * BN_N_LIMBS + 1;
pub(crate) const NIO_CYCLE_FOLD: usize = 4; // 1 per point (3) + scalar
pub(crate) const DEFAULT_ABSORBS: usize = 0;

/// Bit size of Nova field element hashes
pub const NUM_HASH_BITS: usize = 250;

/*
 * *** HyperNova constants ***
*/

/// Number of Matrices for HyperNova
pub(crate) const NUM_MATRICES: usize = 3;
/// Number of base constraints for augmented circuit
pub(crate) const BASE_CONSTRAINTS: usize = 17;
/// Maximum number of constraints per step circuit input
pub(crate) const MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT: usize = 487;
/// Maximum number of constraints per sumcheck round
pub(crate) const MAX_CONSTRAINTS_PER_SUMCHECK_ROUND: usize = 1481;
