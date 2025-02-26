//! This module acts like a polyfill for the case when `bellperson` is compiled without GPU
//! support.

use super::error::{GpuError, GpuResult};
use ec_gpu_gen::threadpool::Worker;
use ff::{Field, PrimeField};
use group::prime::PrimeCurveAffine;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::frontend::gpu::GpuName;

/// A kernel for multiexp
pub struct MultiexpKernel<G>(PhantomData<G>)
where
    G: PrimeCurveAffine;

impl<G> MultiexpKernel<G>
where
    G: PrimeCurveAffine,
{
    /// Create a new kernel
    pub fn create(_: bool) -> GpuResult<Self> {
        Err(GpuError::GpuDisabled)
    }

    /// Run a kernel
    pub fn multiexp(
        &mut self,
        _: &Worker,
        _: Arc<Vec<G>>,
        _: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        _: usize,
        _: usize,
    ) -> GpuResult<G::Curve>
    where
        G: PrimeCurveAffine,
    {
        Err(GpuError::GpuDisabled)
    }
}

macro_rules! locked_kernel {
    (pub struct $class:ident<$generic:ident>
        where $(
            $bound:ty: $boundvalue:tt $(+ $morebounds:tt )*,
        )+
    ) => {
        /// A kernel for multiexp
        pub struct $class<$generic>(PhantomData<$generic>);

        impl<$generic> $class<$generic>
        where $(
            $bound: $boundvalue $(+ $morebounds)*,
        )+
        {
            /// Create a new kernel
            pub fn new(_: bool) -> Self {
                Self(PhantomData)
            }

            /// Run a kernel
            pub fn with<Fun, R, K>(&mut self, _: Fun) -> GpuResult<R>
            where
                Fun: FnMut(&mut K) -> GpuResult<R>,
            {
                return Err(GpuError::GpuDisabled);
            }
        }
    };
}

locked_kernel!(pub struct LockedFftKernel<F> where F: Field + GpuName,);
locked_kernel!(
    pub struct LockedMultiexpKernel<G>
    where
        G: PrimeCurveAffine,
);
