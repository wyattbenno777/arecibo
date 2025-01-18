//! Utility functions for onchain verification
use crate::onchain::verifiers::{GPL3_SDPX_IDENTIFIER, PRAGMA_GROTH16_VERIFIER};
use askama::Template;
use sha3::{Digest, Keccak256};
use num_bigint::BigUint;
pub mod encoding;

/// Formats call data from a vec of bytes to a hashmap
/// Useful for debugging directly on the EVM
/// !! Should follow the contract's function signature, we assuming the order of arguments is correct
pub fn get_formatted_calldata(calldata: Vec<u8>) -> Vec<String> {
    let mut formatted_calldata = vec![];
    for i in (4..calldata.len()).step_by(32) {
        let val = BigUint::from_bytes_be(&calldata[i..i + 32]);
        formatted_calldata.push(format!("{}", val));
    }
    formatted_calldata
}

/// Computes the function selector for the nova cyclefold verifier
/// It is computed on the fly since it depends on the length of the first parameter array
pub fn get_function_selector_for_nova_cyclefold_verifier(
    first_param_array_length: usize,
) -> [u8; 4] {
    let mut hasher = Keccak256::new();
    let fn_sig = format!("verifyNovaProof(uint256[{}],uint256[4],uint256[2],uint256[3],uint256[2],uint256[2][2],uint256[2],uint256[4],uint256[2][2])", first_param_array_length);
    hasher.update(&fn_sig);
    let hash = &mut [0u8; 32];
    hasher.update(hash);
    let h = hasher.finalize();
    [h[0], h[1], h[2], h[3]]
}

/// Header inclusion template
#[derive(Template)]
#[template(path = "header_template.askama.sol", ext = "sol")]
pub struct HeaderInclusion<T: Template> {
    /// SPDX-License-Identifier
    pub sdpx: String,
    /// The `pragma` statement.
    pub pragma_version: String,
    /// The template to render alongside the header.
    pub template: T,
}

impl<T: Template + Default> HeaderInclusion<T> {
    /// Build a new header inclusion
    pub fn builder() -> HeaderInclusionBuilder<T> {
        HeaderInclusionBuilder::default()
    }
}

/// Header inclusion builder
#[derive(Debug)]
pub struct HeaderInclusionBuilder<T: Template + Default> {
    /// SPDX-License-Identifier
    sdpx: String,
    /// The `pragma` statement.
    pragma_version: String,
    /// The template to render alongside the header.
    template: T,
}

impl<T: Template + Default> Default for HeaderInclusionBuilder<T> {
    fn default() -> Self {
        Self {
            sdpx: GPL3_SDPX_IDENTIFIER.to_string(),
            pragma_version: PRAGMA_GROTH16_VERIFIER.to_string(),
            template: T::default(),
        }
    }
}

impl<T: Template + Default> HeaderInclusionBuilder<T> {
    /// Set the SPDX license identifier
    pub fn sdpx<S: Into<String>>(mut self, sdpx: S) -> Self {
        self.sdpx = sdpx.into();
        self
    }

    /// Set the pragma version
    pub fn pragma_version<S: Into<String>>(mut self, pragma_version: S) -> Self {
        self.pragma_version = pragma_version.into();
        self
    }

    /// Set the template
    pub fn template(mut self, template: impl Into<T>) -> Self {
        self.template = template.into();
        self
    }

    /// Build the header inclusion
    pub fn build(self) -> HeaderInclusion<T> {
        HeaderInclusion {
            sdpx: self.sdpx,
            pragma_version: self.pragma_version,
            template: self.template,
        }
    }
}
