//! Solidity templates for the verifier contracts.
//! We use askama for templating and define which variables are required for each template.

/// Pragma statements for Groth16 verifiers
pub const PRAGMA_GROTH16_VERIFIER: &str = "pragma solidity >=0.7.0 <0.9.0;"; // from snarkjs, avoid changing
/// Pragma statements for KZG verifiers
pub const PRAGMA_KZG10_VERIFIER: &str = "pragma solidity >=0.8.1 <=0.8.4;";

/// GPL3 SDPX License identifier
pub const GPL3_SDPX_IDENTIFIER: &str = "// SPDX-License-Identifier: GPL-3.0";
/// MIT SDPX License identifier
pub const MIT_SDPX_IDENTIFIER: &str = "// SPDX-License-Identifier: MIT";

pub mod groth16;
pub mod kzg;
pub mod nebula;

use serde::{Serialize, de::DeserializeOwned};
use std::io::{Write, Read};
use std::fmt;
use serde_json;

/// Trait for protocol verifier keys
pub trait ProtocolVerifierKey: Serialize + DeserializeOwned {
    /// Protocol name
    const PROTOCOL_NAME: &'static str;

    /// Serialize protocol name
    fn serialize_name<W>(&self, writer: &mut W) -> Result<(), fmt::Error>
    where
        W: Write
    {
        serde_json::to_writer(writer, &Self::PROTOCOL_NAME).map_err(|_| fmt::Error)
    }

    /// Serialize protocol verifier key
    fn serialize_protocol_verifier_key<W: Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), fmt::Error> {
        self.serialize_name(writer)?;
        serde_json::to_writer(writer, &self).map_err(|_| fmt::Error)
    }

    /// Deserialize protocol verifier key
    fn deserialize_protocol_verifier_key<R: Read>(
        mut reader: R,
    ) -> Result<Self, fmt::Error> {
        let name: String = serde_json::from_reader(&mut reader).map_err(|_| fmt::Error)?;
        let data = serde_json::from_reader(&mut reader).map_err(|_| fmt::Error)?;

        if name != Self::PROTOCOL_NAME {
            return Err(fmt::Error);
        }

        Ok(data)
    }

    /// Render verifier key as template
    fn render_as_template(self, pragma: Option<String>) -> Vec<u8>;
}
