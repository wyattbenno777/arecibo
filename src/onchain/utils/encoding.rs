//! Encoding utilities for onchain verification
use halo2curves::bn256::{Fq, G1Affine, G2Affine};
use std::fmt::{self, Display};
use serde::{Deserialize, Serialize};
/// Fq wrapper
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FqWrapper(pub Fq);

impl Display for FqWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// G1 representation
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct G1Repr(pub [FqWrapper; 2]);

impl Display for G1Repr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#?}", self.0)
    }
}

/// Converts a G1 element to a representation that can be used in Solidity templates.
pub fn g1_to_fq_repr(g1: G1Affine) -> G1Repr {
    G1Repr([FqWrapper(g1.x), FqWrapper(g1.y)])
}

/// G2 representation
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct G2Repr(pub [[FqWrapper; 2]; 2]);

impl Display for G2Repr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#?}", self.0)
    }
}

/// Converts a G2 element to a representation that can be used in Solidity templates.
pub fn g2_to_fq_repr(g2: G2Affine) -> G2Repr {
    G2Repr([
        [FqWrapper(g2.x.c0), FqWrapper(g2.x.c1)],
        [FqWrapper(g2.y.c0), FqWrapper(g2.y.c1)],
    ])
}
