//! Native WebAssembly SIMD128 instruction mappings.

use core::arch::wasm32;
use core::marker::PhantomData;

use samd_core::impl_simd_ops;

use super::{
    F32x4, F64x2, I8x16, I16x8, I32x4, I64x2, Lane, U8x16, U16x8, U32x4, U64x2, WasmSimd128,
    WasmVector,
};

macro_rules! binary {
    ($name:ident, $ty:ty, $lanes:expr, $intrinsic:path) => {
        #[inline]
        fn $name(
            a: WasmVector<$ty, $lanes>,
            b: WasmVector<$ty, $lanes>,
        ) -> WasmVector<$ty, $lanes> {
            WasmVector($intrinsic(a.0, b.0), PhantomData)
        }
    };
}
macro_rules! unary {
    ($name:ident, $ty:ty, $lanes:expr, $intrinsic:path) => {
        #[inline]
        fn $name(a: WasmVector<$ty, $lanes>) -> WasmVector<$ty, $lanes> {
            WasmVector($intrinsic(a.0), PhantomData)
        }
    };
}
macro_rules! shift {
    ($name:ident, $ty:ty, $lanes:expr, $intrinsic:path) => {
        #[inline]
        fn $name(a: WasmVector<$ty, $lanes>, n: u32) -> WasmVector<$ty, $lanes> {
            WasmVector($intrinsic(a.0, n), PhantomData)
        }
    };
}

binary!(add_i8, i8, 16, wasm32::i8x16_add);
binary!(sub_i8, i8, 16, wasm32::i8x16_sub);
binary!(add_u8, u8, 16, wasm32::i8x16_add);
binary!(sub_u8, u8, 16, wasm32::i8x16_sub);
binary!(add_i16, i16, 8, wasm32::i16x8_add);
binary!(sub_i16, i16, 8, wasm32::i16x8_sub);
binary!(mul_i16, i16, 8, wasm32::i16x8_mul);
binary!(add_u16, u16, 8, wasm32::i16x8_add);
binary!(sub_u16, u16, 8, wasm32::i16x8_sub);
binary!(mul_u16, u16, 8, wasm32::i16x8_mul);
binary!(add_i32, i32, 4, wasm32::i32x4_add);
binary!(sub_i32, i32, 4, wasm32::i32x4_sub);
binary!(mul_i32, i32, 4, wasm32::i32x4_mul);
binary!(add_u32, u32, 4, wasm32::i32x4_add);
binary!(sub_u32, u32, 4, wasm32::i32x4_sub);
binary!(mul_u32, u32, 4, wasm32::i32x4_mul);
binary!(add_i64, i64, 2, wasm32::i64x2_add);
binary!(sub_i64, i64, 2, wasm32::i64x2_sub);
binary!(add_u64, u64, 2, wasm32::i64x2_add);
binary!(sub_u64, u64, 2, wasm32::i64x2_sub);
binary!(add_f32, f32, 4, wasm32::f32x4_add);
binary!(sub_f32, f32, 4, wasm32::f32x4_sub);
binary!(mul_f32, f32, 4, wasm32::f32x4_mul);
binary!(div_f32, f32, 4, wasm32::f32x4_div);
binary!(add_f64, f64, 2, wasm32::f64x2_add);
binary!(sub_f64, f64, 2, wasm32::f64x2_sub);
binary!(mul_f64, f64, 2, wasm32::f64x2_mul);
binary!(div_f64, f64, 2, wasm32::f64x2_div);
unary!(neg_f32, f32, 4, wasm32::f32x4_neg);
unary!(neg_f64, f64, 2, wasm32::f64x2_neg);
unary!(neg_i8, i8, 16, wasm32::i8x16_neg);
unary!(neg_i16, i16, 8, wasm32::i16x8_neg);
unary!(neg_i32, i32, 4, wasm32::i32x4_neg);
unary!(neg_i64, i64, 2, wasm32::i64x2_neg);
unary!(abs_i8, i8, 16, wasm32::i8x16_abs);
unary!(abs_i16, i16, 8, wasm32::i16x8_abs);
unary!(abs_i32, i32, 4, wasm32::i32x4_abs);
unary!(abs_i64, i64, 2, wasm32::i64x2_abs);
unary!(abs_f32, f32, 4, wasm32::f32x4_abs);
unary!(sqrt_f32, f32, 4, wasm32::f32x4_sqrt);
unary!(abs_f64, f64, 2, wasm32::f64x2_abs);
unary!(sqrt_f64, f64, 2, wasm32::f64x2_sqrt);

#[inline]
fn bitand<T: Lane, const LANES: usize>(
    lhs: WasmVector<T, LANES>,
    rhs: WasmVector<T, LANES>,
) -> WasmVector<T, LANES> {
    WasmVector(wasm32::v128_and(lhs.0, rhs.0), PhantomData)
}

#[inline]
fn bitor<T: Lane, const LANES: usize>(
    lhs: WasmVector<T, LANES>,
    rhs: WasmVector<T, LANES>,
) -> WasmVector<T, LANES> {
    WasmVector(wasm32::v128_or(lhs.0, rhs.0), PhantomData)
}

#[inline]
fn bitxor<T: Lane, const LANES: usize>(
    lhs: WasmVector<T, LANES>,
    rhs: WasmVector<T, LANES>,
) -> WasmVector<T, LANES> {
    WasmVector(wasm32::v128_xor(lhs.0, rhs.0), PhantomData)
}

#[inline]
fn bitnot<T: Lane, const LANES: usize>(value: WasmVector<T, LANES>) -> WasmVector<T, LANES> {
    WasmVector(wasm32::v128_not(value.0), PhantomData)
}

shift!(shl_i8, i8, 16, wasm32::i8x16_shl);
shift!(shr_i8, i8, 16, wasm32::i8x16_shr);
shift!(shl_u8, u8, 16, wasm32::i8x16_shl);
shift!(shr_u8, u8, 16, wasm32::u8x16_shr);
shift!(shl_i16, i16, 8, wasm32::i16x8_shl);
shift!(shr_i16, i16, 8, wasm32::i16x8_shr);
shift!(shl_u16, u16, 8, wasm32::i16x8_shl);
shift!(shr_u16, u16, 8, wasm32::u16x8_shr);
shift!(shl_i32, i32, 4, wasm32::i32x4_shl);
shift!(shr_i32, i32, 4, wasm32::i32x4_shr);
shift!(shl_u32, u32, 4, wasm32::i32x4_shl);
shift!(shr_u32, u32, 4, wasm32::u32x4_shr);
shift!(shl_i64, i64, 2, wasm32::i64x2_shl);
shift!(shr_i64, i64, 2, wasm32::i64x2_shr);
shift!(shl_u64, u64, 2, wasm32::i64x2_shl);
shift!(shr_u64, u64, 2, wasm32::u64x2_shr);

impl_simd_ops! {
    WasmSimd128;
    i8 => (I8x16, 16) { add: add_i8, sub: sub_i8, neg: neg_i8, abs: abs_i8, bitand: bitand::<i8, 16>, bitor: bitor::<i8, 16>, bitxor: bitxor::<i8, 16>, not: bitnot::<i8, 16>, shl: shl_i8, shr: shr_i8, }
    u8 => (U8x16, 16) { add: add_u8, sub: sub_u8, bitand: bitand::<u8, 16>, bitor: bitor::<u8, 16>, bitxor: bitxor::<u8, 16>, not: bitnot::<u8, 16>, shl: shl_u8, shr: shr_u8, }
    i16 => (I16x8, 8) { add: add_i16, sub: sub_i16, mul: mul_i16, neg: neg_i16, abs: abs_i16, bitand: bitand::<i16, 8>, bitor: bitor::<i16, 8>, bitxor: bitxor::<i16, 8>, not: bitnot::<i16, 8>, shl: shl_i16, shr: shr_i16, }
    u16 => (U16x8, 8) { add: add_u16, sub: sub_u16, mul: mul_u16, bitand: bitand::<u16, 8>, bitor: bitor::<u16, 8>, bitxor: bitxor::<u16, 8>, not: bitnot::<u16, 8>, shl: shl_u16, shr: shr_u16, }
    i32 => (I32x4, 4) { add: add_i32, sub: sub_i32, mul: mul_i32, neg: neg_i32, abs: abs_i32, bitand: bitand::<i32, 4>, bitor: bitor::<i32, 4>, bitxor: bitxor::<i32, 4>, not: bitnot::<i32, 4>, shl: shl_i32, shr: shr_i32, }
    u32 => (U32x4, 4) { add: add_u32, sub: sub_u32, mul: mul_u32, bitand: bitand::<u32, 4>, bitor: bitor::<u32, 4>, bitxor: bitxor::<u32, 4>, not: bitnot::<u32, 4>, shl: shl_u32, shr: shr_u32, }
    i64 => (I64x2, 2) { add: add_i64, sub: sub_i64, neg: neg_i64, abs: abs_i64, bitand: bitand::<i64, 2>, bitor: bitor::<i64, 2>, bitxor: bitxor::<i64, 2>, not: bitnot::<i64, 2>, shl: shl_i64, shr: shr_i64, }
    u64 => (U64x2, 2) { add: add_u64, sub: sub_u64, bitand: bitand::<u64, 2>, bitor: bitor::<u64, 2>, bitxor: bitxor::<u64, 2>, not: bitnot::<u64, 2>, shl: shl_u64, shr: shr_u64, }
    f32 => (F32x4, 4) { add: add_f32, sub: sub_f32, mul: mul_f32, div: div_f32, neg: neg_f32, abs: abs_f32, sqrt: sqrt_f32, }
    f64 => (F64x2, 2) { add: add_f64, sub: sub_f64, mul: mul_f64, div: div_f64, neg: neg_f64, abs: abs_f64, sqrt: sqrt_f64, }
}
