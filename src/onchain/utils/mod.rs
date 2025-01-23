//! Utility functions for onchain verification
use crate::onchain::verifiers::{GPL3_SDPX_IDENTIFIER, PRAGMA_GROTH16_VERIFIER};
use askama::Template;
use sha3::{Digest, Keccak256};
use num_bigint::BigUint;
use ff::PrimeField;

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

pub fn lagrange_interpolation<F: PrimeField>(v: Vec<F>) -> Vec<F> {
    // Ensure v is not empty
    if v.is_empty() {
        return vec![];
    }

    let n = v.len();
    let mut coeffs = vec![F::ZERO; n];

    // For each point i
    for i in 0..n {
        let mut term = F::ONE;

        // Calculate the Lagrange basis polynomial for point i
        for j in 0..n {
            if i != j {
                // Calculate (x - x_j)/(x_i - x_j) where x_i = i, x_j = j
                let denom = F::from((i as u64).saturating_sub(j as u64));
                let denom_inv = denom.invert().unwrap_or(F::ZERO);
                term = term * denom_inv;
            }
        }

        // Multiply by y_i and add to coefficients
        let scaled = term * v[i];
        coeffs[0] = coeffs[0] + scaled;

        let mut prev = term;
        for j in 1..n {
            let mut curr = F::ZERO;
            for k in 0..n {
                if k != i {
                    curr = curr + (prev * F::from(k as u64));
                }
            }
            curr = curr * -F::ONE;
            coeffs[j] = coeffs[j] + (curr * v[i]);
            prev = curr;
        }
    }

    coeffs
}

pub fn evaluate_polynomial<F: PrimeField>(p: Vec<F>, point: F) -> F {
    p.iter().zip(0..p.len()).fold(F::ZERO, |acc, (coeff, i)| acc + (*coeff) * point.pow([i as u64]))
}

/// Compute the n-th root of unity for a given field
/// Returns None if the n-th root of unity doesn't exist
pub fn nth_root_of_unity<F: PrimeField>(n: usize) -> Option<F> {
    let modulus = BigUint::parse_bytes(F::MODULUS.as_bytes(), 16).unwrap(); // This assumes the modulus is in hex format
    if (&modulus - 1u32) % BigUint::from(n) != BigUint::from(0u32) {
        return None;
    }

    let cofactor = (modulus - BigUint::from(1u32)) / BigUint::from(n);

    
    // Compute generator^(modulus-1)/n
    let root = F::MULTIPLICATIVE_GENERATOR.pow(&cofactor.to_u64_digits());

    // // Verify the order is correct: root^n should = 1 and root^(n-1) should != 1
    // if root.pow(&[n as u64]) != F::ONE || root.pow(&[(n-1) as u64]) == F::ONE {
    //     return None;
    // }

    Some(root)
}
