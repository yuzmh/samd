#![no_std]
//! Architecture-independent traits and scalar implementation for SAMD.
mod chunk;
mod element;
mod macros;
mod scalar;
mod simd;
mod vector;
pub use chunk::Chunk;
pub use element::Element;
pub use scalar::{F32, F64, I8, I16, I32, I64, Scalar, ScalarMask, U8, U16, U32, U64};
pub use simd::Simd;
pub use vector::{MaskSelect, SimdAbs, SimdMask, SimdSqrt, SimdVector};
