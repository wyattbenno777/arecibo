//! Nova verifier
use askama::Template;
use crate::constants::{BN_N_LIMBS, BN_LIMB_WIDTH};
use crate::frontend::num::AllocatedNum;
use crate::onchain::compressed::CompressedVK;
use crate::provider::kzg_commitment::KZGVerifierKey;
use crate::provider::Bn256EngineKZG;

use crate::frontend::groth16;
use super::groth16::{SolidityGroth16Verifier, SolidityGroth16VerifierKey};
use super::kzg::{SolidityKZGVerifier, SolidityKZGVerifierKey};
use crate::onchain::utils::HeaderInclusion;
use crate::onchain::verifiers::{ProtocolVerifierKey, PRAGMA_GROTH16_VERIFIER};
use serde::{Deserialize, Serialize};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use ff::PrimeField;
use crate::gadgets::BigNat;

/// NovaCycleFold verifier key
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NovaCycleFoldVerifierKey {
    pp_hash: DisplayFr,
    g16_vk: SolidityGroth16VerifierKey,
    kzg_vk: SolidityKZGVerifierKey,
    z_len: usize,
}

// TODO: Remove this hack to display Fr
#[derive(Default, Serialize, Deserialize, Debug, Clone)]
struct DisplayFr(Fr);

impl std::fmt::Display for DisplayFr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// NovaCycleFold decider
#[derive(Template, Default)]
#[template(path = "nova_cyclefold_decider.askama.sol", ext = "sol")]
pub struct NovaCycleFoldDecider {
    pp_hash: DisplayFr,
    groth16_verifier: SolidityGroth16Verifier,
    kzg10_verifier: SolidityKZGVerifier,
    // z_len denotes the FCircuit state (z_i) length
    z_len: usize,
    public_inputs_len: usize,
    num_limbs: usize,
    bits_per_limb: usize,
}

/// Get the decider template for the NovaCycleFold decider
pub fn get_decider_template_for_cyclefold_decider(
    nova_cyclefold_vk: NovaCycleFoldVerifierKey,
) -> String {
    HeaderInclusion::<NovaCycleFoldDecider>::builder()
        .template(nova_cyclefold_vk)
        .build()
        .render()
        .unwrap()
}

impl From<NovaCycleFoldVerifierKey> for NovaCycleFoldDecider {
    fn from(value: NovaCycleFoldVerifierKey) -> Self {
        let solidity_groth16_verifier_key = value.g16_vk;
        let solidity_kzg_verifier_key = value.kzg_vk;
        let public_inputs_len = solidity_groth16_verifier_key.vk.ic.len();
        // let bits_per_limb = NonNativeUintVar::<Fq>::bits_per_limb();
        Self {
            pp_hash: value.pp_hash,
            groth16_verifier: SolidityGroth16Verifier::from(solidity_groth16_verifier_key),
            kzg10_verifier: SolidityKZGVerifier::from(solidity_kzg_verifier_key),
            z_len: value.z_len,
            public_inputs_len,
            num_limbs: BN_N_LIMBS,   //: (250_f32 / (bits_per_limb as f32)).ceil() as usize,
            bits_per_limb: BN_LIMB_WIDTH,
        }
    }
}

/// `AllocatedLimb` represents a single limb of a non-native unsigned integer in the
/// circuit.
/// The limb value `v` should be small enough to fit into `AllocatedNum`, and we also
/// store an upper bound `ub` for the limb value, which is treated as a constant
/// in the circuit and is used for efficient equality checks and some arithmetic
/// operations.
#[derive(Debug, Clone)]
pub struct AllocatedLimb<F: PrimeField> {
    /// Limb value
    pub v: AllocatedNum<F>,
    /// Upper bound
    pub ub: BigNat<F>,
}

/// `NonNativeUintVar` represents a non-native unsigned integer (BigUint) in the
/// circuit.
/// We apply [xJsnark](https://akosba.github.io/papers/xjsnark.pdf)'s techniques
/// for efficient operations on `NonNativeUintVar`.
/// Note that `NonNativeUintVar` is different from arkworks' `NonNativeFieldVar`
/// in that the latter runs the expensive `reduce` (`align` + `modulo` in our
/// terminology) after each arithmetic operation, while the former only reduces
/// the integer when explicitly called.
#[derive(Debug, Clone)]
pub struct NonNativeUintVar<F: PrimeField>(pub Vec<AllocatedLimb<F>>);

impl<F: PrimeField> NonNativeUintVar<F> {
    /// Get the bits per limb
    pub const fn bits_per_limb() -> usize {
        assert!(F::NUM_BITS > 250);
        // For a `F` with order > 250 bits, 55 is chosen for optimizing the most
        // expensive part `Az∘Bz` when checking the R1CS relation for CycleFold.
        // Consider using `NonNativeUintVar` to represent the base field `Fq`.
        // Since 250 / 55 = 4.46, the `NonNativeUintVar` has 5 limbs.
        // Now, the multiplication of two `NonNativeUintVar`s has 9 limbs, and
        // each limb has at most 2^{55 * 2} * 5 = 112.3 bits.
        // For a 1400x1400 matrix `A`, the multiplication of `A`'s row and `z`
        // is the sum of 1400 `NonNativeUintVar`s, each with 9 limbs.
        // Thus, the maximum bit length of limbs of each element in `Az` is
        // 2^{55 * 2} * 5 * 1400 = 122.7 bits.
        // Finally, in the hadamard product of `Az` and `Bz`, every element has
        // 17 limbs, whose maximum bit length is (2^{55 * 2} * 5 * 1400)^2 * 9
        // = 248.7 bits and is less than the native field `Fr`.
        // Thus, 55 allows us to compute `Az∘Bz` without the expensive alignment
        // operation.
        //
        // TODO: either make it a global const, or compute an optimal value
        // based on the modulus size.
        55
    }
}

impl ProtocolVerifierKey for NovaCycleFoldVerifierKey {
    const PROTOCOL_NAME: &'static str = "NovaCycleFold";

    fn render_as_template(self, pragma: Option<String>) -> Vec<u8> {
        HeaderInclusion::<NovaCycleFoldDecider>::builder()
            .pragma_version(pragma.unwrap_or(PRAGMA_GROTH16_VERIFIER.to_string()))
            .template(self)
            .build()
            .render()
            .unwrap()
            .into_bytes()
    }
}

impl From<(Fr, SolidityGroth16VerifierKey, SolidityKZGVerifierKey, usize)> for NovaCycleFoldVerifierKey {
    fn from(value: (Fr, SolidityGroth16VerifierKey, SolidityKZGVerifierKey, usize)) -> Self {
        Self {
            pp_hash: DisplayFr(value.0),
            g16_vk: value.1,
            kzg_vk: value.2,
            z_len: value.3,
        }
    }
}

impl From<CompressedVK> for NovaCycleFoldVerifierKey {
    fn from(value: CompressedVK) -> Self {
        let g16_vk = SolidityGroth16VerifierKey::from(value.groth16_vk);
        let kzg_vk = SolidityKZGVerifierKey::from((value.kzg_vk, Vec::new()));
        Self {
            pp_hash: DisplayFr(value.pp_hash),
            g16_vk,
            kzg_vk,
            z_len: 1,
        }
    }
}

impl NovaCycleFoldVerifierKey {
    /// Create a new NovaCycleFoldVerifierKey
    pub fn new(
        pp_hash: Fr,
        g16_vk: groth16::VerifyingKey<Bn256EngineKZG>,
        vkey_kzg: KZGVerifierKey<Bn256>,
        crs_points: Vec<G1Affine>,
        z_len: usize,
    ) -> Self {
        Self {
            pp_hash: DisplayFr(pp_hash),
            g16_vk: SolidityGroth16VerifierKey::from(g16_vk),
            kzg_vk: SolidityKZGVerifierKey::from((vkey_kzg, crs_points)),
            z_len,
        }
    }
}
