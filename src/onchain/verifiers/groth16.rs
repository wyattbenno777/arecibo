//! Groth16 verifier
use crate::onchain::utils::encoding::{g1_to_fq_repr, g2_to_fq_repr};
use crate::onchain::utils::encoding::{G1Repr, G2Repr};
use crate::onchain::utils::HeaderInclusion;
use askama::Template;
use serde::{Deserialize, Serialize};
use super::PRAGMA_GROTH16_VERIFIER;
use super::ProtocolVerifierKey;
use super::GPL3_SDPX_IDENTIFIER;
use crate::onchain::groth16::VerifyingKey;

/// Solidity Groth16 verifier
#[derive(Template, Default)]
#[template(path = "groth16_verifier.askama.sol", ext = "sol")]
pub struct Groth16Verifier {
    /// The `alpha * G`, where `G` is the generator of `G1`.
    pub vkey_alpha_g1: G1Repr,
    /// The `alpha * H`, where `H` is the generator of `G2`.
    pub vkey_beta_g2: G2Repr,
    /// The `gamma * H`, where `H` is the generator of `G2`.
    pub vkey_gamma_g2: G2Repr,
    /// The `delta * H`, where `H` is the generator of `G2`.
    pub vkey_delta_g2: G2Repr,
    /// Length of the `gamma_abc_g1` vector.
    pub gamma_abc_len: usize,
    /// The `gamma^{-1} * (beta * a_i + alpha * b_i + c_i) * H`, where `H` is the generator of `E::G1`.
    pub gamma_abc_g1: Vec<G1Repr>,
}

impl From<Groth16VerifierKey> for Groth16Verifier {
    fn from(g16_vk: Groth16VerifierKey) -> Self {
        Self {
            vkey_alpha_g1: g1_to_fq_repr(g16_vk.0.alpha_g1),
            vkey_beta_g2: g2_to_fq_repr(g16_vk.0.beta_g2),
            vkey_gamma_g2: g2_to_fq_repr(g16_vk.0.gamma_g2),
            vkey_delta_g2: g2_to_fq_repr(g16_vk.0.delta_g2),
            gamma_abc_len: g16_vk.0.gamma_abc_g1.len(),
            gamma_abc_g1: g16_vk
                .0
                .gamma_abc_g1
                .iter()
                .copied()
                .map(g1_to_fq_repr)
                .collect(),
        }
    }
}

/// Groth16 verifier key
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Groth16VerifierKey(pub(crate) VerifyingKey);

impl From<VerifyingKey> for Groth16VerifierKey {
    fn from(value: VerifyingKey) -> Self {
        Self(value)
    }
}

impl From<Groth16VerifierKey> for VerifyingKey {
    fn from(value: Groth16VerifierKey) -> Self {
        value.0
    }
}

impl ProtocolVerifierKey for Groth16VerifierKey {
    const PROTOCOL_NAME: &'static str = "Groth16";

    fn render_as_template(self, pragma: Option<String>) -> Vec<u8> {
        HeaderInclusion::<Groth16Verifier>::builder()
            .sdpx(GPL3_SDPX_IDENTIFIER.to_string())
            .pragma_version(pragma.unwrap_or(PRAGMA_GROTH16_VERIFIER.to_string()))
            .template(self)
            .build()
            .render()
            .unwrap()
            .into_bytes()
    }
}