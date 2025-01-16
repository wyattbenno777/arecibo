//! KZG verifier
use crate::onchain::utils::encoding::{g1_to_fq_repr, g2_to_fq_repr};
use crate::onchain::utils::encoding::{G1Repr, G2Repr};
use crate::onchain::utils::HeaderInclusion;
use crate::onchain::verifiers::ProtocolVerifierKey;
use crate::onchain::verifiers::MIT_SDPX_IDENTIFIER;
use halo2curves::bn256::{Bn256, G1Affine};
use askama::Template;
use serde::{Deserialize, Serialize};
use pairing::Engine;
use super::PRAGMA_KZG10_VERIFIER;

/// Solidity KZG10 verifier
#[derive(Template, Default)]
#[template(path = "kzg10_verifier.askama.sol", ext = "sol")]
pub struct KZG10Verifier {
    /// The generator of `G1`.
    pub(crate) g1: G1Repr,
    /// The generator of `G2`.
    pub(crate) g2: G2Repr,
    /// The verification key
    pub(crate) vk: G2Repr,
    /// Length of the trusted setup vector.
    pub(crate) g1_crs_len: usize,
    /// The trusted setup vector.
    pub(crate) g1_crs: Vec<G1Repr>,
}

/// KZG10 verifier key
#[derive(Deserialize,Serialize, Clone, PartialEq, Debug)]
pub struct VerifierKey<E: Engine> {
    /// The generator of G1.
    pub g: E::G1Affine,
    /// The generator of G1 that is used for making a commitment hiding.
    pub gamma_g: E::G1Affine,
    /// The generator of G2.
    pub h: E::G2Affine,
    /// \beta times the above generator of G2.
    pub beta_h: E::G2Affine,
    /// The generator of G2, prepared for use in pairings.
    // #[derivative(Debug = "ignore", PartialEq = "ignore")]
    pub prepared_h: E::G2,
    /// \beta times the above generator of G2, prepared for use in pairings.
    // #[derivative(Debug = "ignore", PartialEq = "ignore")]
    pub prepared_beta_h: E::G2,
}




impl From<KZG10VerifierKey> for KZG10Verifier {
    fn from(data: KZG10VerifierKey) -> Self {
        Self {
            g1: g1_to_fq_repr(data.vk.g),
            g2: g2_to_fq_repr(data.vk.h),
            vk: g2_to_fq_repr(data.vk.beta_h),
            g1_crs_len: data.g1_crs_batch_points.len(),
            g1_crs: data
                .g1_crs_batch_points
                .iter()
                .map(|g1| g1_to_fq_repr(*g1))
                .collect(),
        }
    }
}

/// KZG10 verifier key
#[derive(Deserialize,Serialize, Clone, Debug)]
pub struct KZG10VerifierKey {
    /// Verification key
    pub vk: VerifierKey<Bn256>,
    /// G1 CRS batch points
    pub g1_crs_batch_points: Vec<G1Affine>,
}

impl From<(VerifierKey<Bn256>, Vec<G1Affine>)> for KZG10VerifierKey {
    fn from(value: (VerifierKey<Bn256>, Vec<G1Affine>)) -> Self {
        Self {
            vk: value.0,
            g1_crs_batch_points: value.1,
        }
    }
}

impl ProtocolVerifierKey for KZG10VerifierKey {
    const PROTOCOL_NAME: &'static str = "KZG";

    fn render_as_template(self, pragma: Option<String>) -> Vec<u8> {
        HeaderInclusion::<KZG10Verifier>::builder()
            .sdpx(MIT_SDPX_IDENTIFIER.to_string())
            .pragma_version(pragma.unwrap_or(PRAGMA_KZG10_VERIFIER.to_string()))
            .template(self)
            .build()
            .render()
            .unwrap()
            .into_bytes()
    }
}
