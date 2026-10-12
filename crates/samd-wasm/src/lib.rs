#![no_std]
//! Native WebAssembly SIMD128 backend for SAMD.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod simd128;
