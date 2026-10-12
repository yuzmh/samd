#![no_std]
//! Public facade for automatically dispatched SAMD SIMD kernels.

extern crate self as samd;

pub use samd_core::impl_simd_ops;
pub use samd_core::{
    Chunk, Element, F32, F64, I8, I16, I32, I64, MaskSelect, Scalar, ScalarMask, Simd, SimdAbs,
    SimdMask, SimdSqrt, SimdVector, U8, U16, U32, U64,
};
pub use samd_macros::simd;

/// Types used by generated `#[samd::simd]` kernels.
#[doc(hidden)]
pub mod __private {
    pub use samd_core::Scalar;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub use samd_wasm::simd128::WasmSimd128;
    #[cfg(target_arch = "x86_64")]
    pub use samd_x86::X86Backend;
    #[cfg(target_arch = "x86_64")]
    pub use samd_x86::avx2::Avx2;
    #[cfg(target_arch = "x86_64")]
    pub use samd_x86::best_x86_backend;
    #[cfg(target_arch = "x86_64")]
    pub use samd_x86::sse2::Sse2;
}
