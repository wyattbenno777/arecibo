//! Groth16 verifier
use crate::onchain::utils::encoding::{g1_to_fq_repr, g2_to_fq_repr};
use crate::onchain::utils::encoding::{G1Repr, G2Repr};
use crate::onchain::utils::HeaderInclusion;
use crate::provider::Bn256EngineKZG;
use askama::Template;
use serde::{Deserialize, Serialize};
use super::PRAGMA_GROTH16_VERIFIER;
use super::ProtocolVerifierKey;
use super::GPL3_SDPX_IDENTIFIER;
use crate::frontend::groth16::VerifyingKey;

/// Solidity Groth16 verifier
#[derive(Template, Default, Serialize, Deserialize)]
#[template(path = "groth16_verifier.askama.sol", ext = "sol")]
pub struct SolidityGroth16Verifier {
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

impl From<SolidityGroth16VerifierKey> for SolidityGroth16Verifier {
    fn from(g16_vk: SolidityGroth16VerifierKey) -> Self {
        Self {
            vkey_alpha_g1: g1_to_fq_repr(g16_vk.vk.alpha_g1),
            vkey_beta_g2: g2_to_fq_repr(g16_vk.vk.beta_g2),
            vkey_gamma_g2: g2_to_fq_repr(g16_vk.vk.gamma_g2),
            vkey_delta_g2: g2_to_fq_repr(g16_vk.vk.delta_g2),
            gamma_abc_len: g16_vk.vk.ic.len(),
            gamma_abc_g1: g16_vk
                .vk
                .ic
                .iter()
                .copied()
                .map(g1_to_fq_repr)
                .collect(),
        }
    }
}

/// Groth16 verifier key
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct SolidityGroth16VerifierKey {
    /// Verification key
    pub vk: VerifyingKey<Bn256EngineKZG>,
}

impl From<VerifyingKey<Bn256EngineKZG>> for SolidityGroth16VerifierKey {
    fn from(data: VerifyingKey<Bn256EngineKZG>) -> Self {
        Self {
            vk: data,
        }
    }
}

impl ProtocolVerifierKey for SolidityGroth16VerifierKey {
    const PROTOCOL_NAME: &'static str = "Groth16";

    fn render_as_template(self, pragma: Option<String>) -> Vec<u8> {
        HeaderInclusion::<SolidityGroth16Verifier>::builder()
            .sdpx(GPL3_SDPX_IDENTIFIER.to_string())
            .pragma_version(pragma.unwrap_or(PRAGMA_GROTH16_VERIFIER.to_string()))
            .template(self)
            .build()
            .render()
            .unwrap()
            .into_bytes()
    }
}