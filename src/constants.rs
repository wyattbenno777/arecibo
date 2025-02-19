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

/// Base number of constraints for the augmented circuit.
///
/// The total constraint count depends on the number of sumcheck rounds:
///
/// - **0 rounds:** The circuit would have 46967 constraints.
/// - **Edge Case (0 → 1 round):** Transitioning from 0 rounds to 1 round increases the constraint count by 2295.
/// - **Subsequent rounds:** Each additional round after the first increases the constraints by 1518.
///
/// Since the circuit always uses at least one round (the 0-round case never happens), we
/// incorporate the extra cost of the 0-to-1 round transition by adding the difference between
/// the first round increase and a regular round increase: (2295 - 1518) = 777.
///
/// Therefore, the base constraint count is:
///     BASE_CONSTRAINTS = 46967 (for 0 rounds) + 777 (extra for the edge case to 1 round)
pub(crate) const EDGE_CASE_CONSTRAINTS: usize = 777;
pub(crate) const BASE_CONSTRAINTS: usize = 46967 + EDGE_CASE_CONSTRAINTS;

/// Maximum number of constraints per step circuit input
pub(crate) const MAX_CONSTRAINTS_PER_STEP_CIRCUIT_INPUT: usize = 1;

/// Maximum number of constraints per sumcheck round
pub(crate) const MAX_CONSTRAINTS_PER_SUMCHECK_ROUND: usize = 1518;

/// Number of univariate coefficients in a HyperNova sumcheck proof
pub(crate) const NUM_UNIVARIATE_COEFFS: usize = 4;
