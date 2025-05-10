//! GPU support
mod error;

use halo2curves::bn256::{G1Affine, G2Affine, Fr as Scalar};
pub use self::error::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod locks;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::locks::*;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod multiexp;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::multiexp::CpuGpuMultiexpKernel;

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
mod nogpu;

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
pub use self::nogpu::*;

// This is a hack, so that the same traits can be used for the GPU and non-GPU code path.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use ec_gpu::GpuName;
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
/// A trait for GPU names
pub trait GpuName {}
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl GpuName for G1Affine {}
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl GpuName for G2Affine {}
#[cfg(not(any(feature = "cuda", feature = "opencl")))]
impl GpuName for Scalar {}
