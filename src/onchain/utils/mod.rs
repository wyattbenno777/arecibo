//! Utility functions for onchain verification
use crate::{constants::{BN_LIMB_WIDTH, BN_N_LIMBS}, gadgets::nat_to_limbs, onchain::verifiers::{GPL3_SDPX_IDENTIFIER, PRAGMA_GROTH16_VERIFIER}, provider::Bn256EngineKZG, traits::commitment::CommitmentTrait, Commitment, NovaError};
use askama::Template;
use halo2curves::bn256::Fr;
use sha3::{Digest, Keccak256};
use num_bigint::{BigInt, BigUint, Sign};
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
    let fn_sig = format!("verifyNovaProof(uint256[{}],uint256[4],uint256[2],uint256[3],uint256[2],uint256[2][2],uint256[2],uint256[4],uint256[2][2])", first_param_array_length);
    let mut hasher = Keccak256::new();
    hasher.update(fn_sig.as_bytes());
    let hash = hasher.finalize();
    [hash[0], hash[1], hash[2], hash[3]]
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

fn parse_hex_string_to_u8_vec(hex_str: &str) -> Vec<u8> {
    // Remove the "0x" prefix
    let hex_str = &hex_str[2..];

    // Convert the hex string to bytes
    hex::decode(hex_str).expect("Invalid hex string")
}
/// Compute the n-th root of unity for a given field
/// Returns None if the n-th root of unity doesn't exist
pub fn nth_root_of_unity<F: PrimeField>(n: usize) -> Option<F> {
    let bytes = parse_hex_string_to_u8_vec(F::MODULUS);

    let modulus = BigUint::from_bytes_be(&bytes);
    if (&modulus - BigUint::from(1u32)) % BigUint::from(n) != BigUint::from(0u32) {
        println!("Modulus - 1 is not divisible by n");
        return None;
    }

    let cofactor = (modulus - BigUint::from(1u32)) / BigUint::from(n);
    
    // Compute generator^(modulus-1)/n
    let root = F::MULTIPLICATIVE_GENERATOR.pow(cofactor.to_u64_digits());

    // Verify the order is correct: root^n should = 1 and root^(n-1) should != 1
    if root.pow([n as u64]) != F::ONE || root.pow([(n-1) as u64]) == F::ONE {
        return None;
    }

    Some(root)
}

/// Converts a commitment to its scalar coordinates.
pub fn to_scalar_coordinates(comm: &Commitment<Bn256EngineKZG>) -> Result<(Vec<Fr>, Vec<Fr>, Fr), NovaError> {
    let (x, y, id) = comm.to_coordinates();
    let x_bignat = BigInt::from_bytes_le(Sign::Plus, &x.to_repr());
    let x_limbs = nat_to_limbs(&x_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
    let y_bignat = BigInt::from_bytes_le(Sign::Plus, &y.to_repr());
    let y_limbs = nat_to_limbs(&y_bignat, BN_LIMB_WIDTH, BN_N_LIMBS)?;
    let id_fr = Fr::from(id);
    Ok((x_limbs, y_limbs, id_fr))
  }