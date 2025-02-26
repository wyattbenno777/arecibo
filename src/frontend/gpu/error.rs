use crate::frontend::SynthesisError;
use ec_gpu_gen::EcError;

#[derive(thiserror::Error, Debug)]
/// An error from the EC GPU
pub enum GpuError {
    /// A simple error
    #[error("GPUError: {0}")]
    Simple(&'static str),
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    #[error("No kernel is initialized!")]
    KernelUninitialized,
    /// An error from the EC GPU
    #[error("EC GPU error: {0}")]
    EcGpu(#[from] EcError),
    /// GPU accelerator is disabled
    #[error("GPU accelerator is disabled!")]
    GpuDisabled,
}

/// A result type for GPU operations
pub type GpuResult<T> = std::result::Result<T, GpuError>;

impl From<GpuError> for SynthesisError {
    fn from(e: GpuError) -> Self {
        // inspired by the commenct on MalformedProofs
        SynthesisError::MalformedProofs(format!("Encountered a GPU Error: {}", e))
    }
}
