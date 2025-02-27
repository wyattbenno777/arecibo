//! This module provides a trait and implementations for converting Rust types
//! to EVM calldata.
use crate::{frontend::groth16::Proof as Groth16Proof, provider::{kzg_commitment::UVKZGCommitment, pedersen::Commitment, traits::DlogGroup, Bn256EngineKZG}};
use ff::PrimeField;
use halo2curves::bn256::{Bn256, Fq, Fq2, Fr, G1Affine, G2Affine, G1};
pub mod evm;

/// Trait for converting Rust types to EVM calldata.
pub trait ToEth {
    /// Convert the type to a vector of bytes.
    fn to_eth(&self) -> Vec<u8>;
}

impl<T: ToEth> ToEth for [T] {
    fn to_eth(&self) -> Vec<u8> {
        self.iter().flat_map(ToEth::to_eth).collect()
    }
}

impl ToEth for u8 {
    fn to_eth(&self) -> Vec<u8> {
        vec![*self]
    }
}

impl ToEth for Fr {
    fn to_eth(&self) -> Vec<u8> {
        let mut repr = self.to_repr().to_vec();
        repr.reverse();
        repr
    }
}

impl ToEth for Fq {
    fn to_eth(&self) -> Vec<u8> {
        let mut repr = self.to_repr().to_vec();
        repr.reverse();
        repr
    }
}

impl ToEth for Fq2 {
    fn to_eth(&self) -> Vec<u8> {
        [self.c1.to_eth(), self.c0.to_eth()].concat()
    }
}

impl ToEth for G1Affine {
    fn to_eth(&self) -> Vec<u8> {
        // the encoding of the additive identity is [0, 0] on the EVM
        [self.x.to_eth(), self.y.to_eth()].concat()
    }
}

impl ToEth for G2Affine {
    fn to_eth(&self) -> Vec<u8> {
        // the encoding of the additive identity is [0, 0] on the EVM
        [self.x.to_eth(), self.y.to_eth()].concat()
    }
}

impl ToEth for G1 {
    fn to_eth(&self) -> Vec<u8> {
        let (x, y, _) = self.to_coordinates();
        [x.to_eth(), y.to_eth()].concat()
    }
}

impl ToEth for Groth16Proof<Bn256EngineKZG> {
    fn to_eth(&self) -> Vec<u8> {
        [self.a.to_eth(), self.b.to_eth(), self.c.to_eth()].concat()
    }
}

impl ToEth for UVKZGCommitment<Bn256> {
    fn to_eth(&self) -> Vec<u8> {
        self.0.to_eth()
    }
}

impl ToEth for UVKZGCommitment<Bn256EngineKZG> {
    fn to_eth(&self) -> Vec<u8> {
        self.0.to_eth()
    }
}

impl ToEth for Commitment<Bn256EngineKZG> {
    fn to_eth(&self) -> Vec<u8> {
        self.comm.to_eth()
    }
}