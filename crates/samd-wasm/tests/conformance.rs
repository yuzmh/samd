#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use core::cmp::PartialEq;
use samd_core::{SimdAbs, SimdMask, SimdSqrt, SimdVector};
use samd_wasm::simd128::{
    F32x4, F64x2, I8x16, I16x8, I32x4, I64x2, U8x16, U16x8, U32x4, U64x2, WasmSimd128,
};

macro_rules! check_lanes {
    (
        $name:ident,
        $scalar:ty,
        $vector:ty,
        $lanes:expr,
        $min:expr,
        $max:expr,
        $zero:expr,
        $one:expr,
        $two:expr,
        $four:expr,
        $six:expr,
        $sentinel:expr,
        $lane_count:expr $(,)?) => {
        fn $name() {
            let zero: $scalar = $zero;
            let one: $scalar = $one;
            let two: $scalar = $two;
            let four: $scalar = $four;
            let six: $scalar = $six;
            let sentinel: $scalar = $sentinel;
            let lane_count: $scalar = $lane_count;
            let a = [four; $lanes];
            let b = [two; $lanes];
            let x = <$vector as SimdVector<$scalar, WasmSimd128>>::load(WasmSimd128, &a, 0, $lanes);
            let y = <$vector as SimdVector<$scalar, WasmSimd128>>::load(WasmSimd128, &b, 0, $lanes);
            let mut out = [zero; $lanes];

            <$vector as SimdVector<$scalar, WasmSimd128>>::store(x + y, &mut out, 0, $lanes);
            assert!(out.iter().all(|&v| PartialEq::eq(&v, &six)));
            <$vector as SimdVector<$scalar, WasmSimd128>>::store(x - y, &mut out, 0, $lanes);
            assert!(out.iter().all(|&v| PartialEq::eq(&v, &two)));

            assert_eq!(x.sum(), four * lane_count);
            assert_eq!(x.min(), four);
            assert_eq!(x.max(), four);

            let mask = x.gt(&y);
            assert!(mask.all());
            assert_eq!(mask.count_true(), $lanes);
            assert_eq!(mask.active_count_true($lanes - 1), $lanes - 1);
            assert!(mask.active_all($lanes - 1));
            assert!(!mask.active_any(0));
            assert!(x.eq(&x).all());
            assert!(x.ne(&y).all());
            assert!(x.lt(&y).none());
            assert!(x.le(&x).all());
            assert!(x.ge(&x).all());
            <$vector as SimdVector<$scalar, WasmSimd128>>::store(
                mask.select(x, y),
                &mut out,
                0,
                $lanes,
            );
            assert!(out.iter().all(|&v| PartialEq::eq(&v, &four)));

            let ones = <$vector as SimdVector<$scalar, WasmSimd128>>::splat(WasmSimd128, one);
            assert_eq!(ones.prod(), one);

            // Tail accesses may touch only the active elements. Sentinel elements
            // before and after the selected range must be preserved.
            let tail =
                <$vector as SimdVector<$scalar, WasmSimd128>>::load(WasmSimd128, &a, 0, $lanes - 1);
            assert_eq!(tail.active_sum($lanes - 1), four * (lane_count - one));
            let tail_ones = <$vector as SimdVector<$scalar, WasmSimd128>>::load(
                WasmSimd128,
                &[one; $lanes],
                0,
                $lanes - 1,
            );
            assert_eq!(tail_ones.active_prod($lanes - 1), one);
            assert_eq!(tail.active_min($lanes - 1), four);
            assert_eq!(tail.active_max($lanes - 1), four);
            let mut guarded = [sentinel; $lanes + 2];
            <$vector as SimdVector<$scalar, WasmSimd128>>::store(tail, &mut guarded, 1, $lanes - 1);
            assert!(PartialEq::eq(&guarded[0], &sentinel));
            assert!(PartialEq::eq(&guarded[$lanes], &sentinel));
            assert!(guarded[1..$lanes].iter().all(|&v| PartialEq::eq(&v, &four)));

            let empty = <$vector as SimdVector<$scalar, WasmSimd128>>::load(WasmSimd128, &[], 0, 0);
            assert_eq!(empty.active_sum(0), zero);
            assert_eq!(empty.active_prod(0), one);
            assert_eq!(empty.active_min(0), $min);
            assert_eq!(empty.active_max(0), $max);
        }
    };
}

check_lanes!(
    i8_conformance,
    i8,
    I8x16,
    16,
    i8::MAX,
    i8::MIN,
    0_i8,
    1_i8,
    2_i8,
    4_i8,
    6_i8,
    99_i8,
    16_i8
);
check_lanes!(
    u8_conformance,
    u8,
    U8x16,
    16,
    u8::MAX,
    u8::MIN,
    0_u8,
    1_u8,
    2_u8,
    4_u8,
    6_u8,
    99_u8,
    16_u8
);
check_lanes!(
    i16_conformance,
    i16,
    I16x8,
    8,
    i16::MAX,
    i16::MIN,
    0_i16,
    1_i16,
    2_i16,
    4_i16,
    6_i16,
    99_i16,
    8_i16
);
check_lanes!(
    u16_conformance,
    u16,
    U16x8,
    8,
    u16::MAX,
    u16::MIN,
    0_u16,
    1_u16,
    2_u16,
    4_u16,
    6_u16,
    99_u16,
    8_u16
);
check_lanes!(
    i32_conformance,
    i32,
    I32x4,
    4,
    i32::MAX,
    i32::MIN,
    0_i32,
    1_i32,
    2_i32,
    4_i32,
    6_i32,
    99_i32,
    4_i32
);
check_lanes!(
    u32_conformance,
    u32,
    U32x4,
    4,
    u32::MAX,
    u32::MIN,
    0_u32,
    1_u32,
    2_u32,
    4_u32,
    6_u32,
    99_u32,
    4_u32
);
check_lanes!(
    i64_conformance,
    i64,
    I64x2,
    2,
    i64::MAX,
    i64::MIN,
    0_i64,
    1_i64,
    2_i64,
    4_i64,
    6_i64,
    99_i64,
    2_i64
);
check_lanes!(
    u64_conformance,
    u64,
    U64x2,
    2,
    u64::MAX,
    u64::MIN,
    0_u64,
    1_u64,
    2_u64,
    4_u64,
    6_u64,
    99_u64,
    2_u64
);
check_lanes!(
    f32_conformance,
    f32,
    F32x4,
    4,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0_f32,
    1.0_f32,
    2.0_f32,
    4.0_f32,
    6.0_f32,
    99.0_f32,
    4.0_f32
);
check_lanes!(
    f64_conformance,
    f64,
    F64x2,
    2,
    f64::INFINITY,
    f64::NEG_INFINITY,
    0.0_f64,
    1.0_f64,
    2.0_f64,
    4.0_f64,
    6.0_f64,
    99.0_f64,
    2.0_f64
);

#[test]
fn native_vectors_cover_every_supported_element_type() {
    i8_conformance();
    u8_conformance();
    i16_conformance();
    u16_conformance();
    i32_conformance();
    u32_conformance();
    i64_conformance();
    u64_conformance();
    f32_conformance();
    f64_conformance();
}

fn native_multiplication_and_float_unary_ops() {
    let a = F32x4::splat(WasmSimd128, 3.0);
    let b = F32x4::splat(WasmSimd128, 2.0);
    let p = a * b;
    assert_eq!(p.sum(), 24.0);
    assert_eq!((-a).abs().sqrt().sum(), 4.0 * 3.0_f32.sqrt());

    let left_i16 = I16x8::splat(WasmSimd128, 7);
    let right_i16 = I16x8::splat(WasmSimd128, 6);
    assert_eq!((left_i16 * right_i16).sum(), 336);
    let left_i32 = I32x4::splat(WasmSimd128, 7);
    let right_i32 = I32x4::splat(WasmSimd128, 6);
    assert_eq!((left_i32 * right_i32).sum(), 168);
}

fn i64_relational_comparisons_cover_signed_and_unsigned_high_words() {
    let signed_a =
        <I64x2 as SimdVector<i64, WasmSimd128>>::load(WasmSimd128, &[i64::MIN, i64::MAX], 0, 2);
    let signed_b = <I64x2 as SimdVector<i64, WasmSimd128>>::load(WasmSimd128, &[0, 0], 0, 2);
    assert_eq!(signed_a.lt(&signed_b).count_true(), 1);
    assert_eq!(signed_a.gt(&signed_b).count_true(), 1);

    let unsigned_a =
        <U64x2 as SimdVector<u64, WasmSimd128>>::load(WasmSimd128, &[u64::MAX, 0], 0, 2);
    let unsigned_b =
        <U64x2 as SimdVector<u64, WasmSimd128>>::load(WasmSimd128, &[0, u64::MAX], 0, 2);
    assert_eq!(unsigned_a.gt(&unsigned_b).count_true(), 1);
    assert_eq!(unsigned_a.lt(&unsigned_b).count_true(), 1);
}

fn float_nan_and_signed_zero_comparison_and_reduction() {
    let a = <F32x4 as SimdVector<f32, WasmSimd128>>::load(
        WasmSimd128,
        &[f32::NAN, 0.0, f32::INFINITY, f32::NEG_INFINITY],
        0,
        4,
    );
    let b = <F32x4 as SimdVector<f32, WasmSimd128>>::splat(WasmSimd128, 0.0);
    assert_eq!(a.ne(&b).count_true(), 3);
    assert_eq!(a.lt(&b).count_true(), 1);
    assert_eq!(a.gt(&b).count_true(), 1);

    let zeros =
        <F32x4 as SimdVector<f32, WasmSimd128>>::load(WasmSimd128, &[0.0, -0.0, 0.0, -0.0], 0, 4);
    assert!(zeros.min().is_sign_negative());
    assert!(!zeros.max().is_sign_negative());
    let nans = <F32x4 as SimdVector<f32, WasmSimd128>>::splat(WasmSimd128, f32::NAN);
    assert!(nans.min().is_nan());
    assert!(nans.max().is_nan());
}

fn generated_native_bitwise_and_shift_operators() {
    let a = I16x8::splat(WasmSimd128, 0b1010);
    let b = I16x8::splat(WasmSimd128, 0b1100);
    assert_eq!((a & b).sum(), 8 * 0b1000);
    assert_eq!((a | b).sum(), 8 * 0b1110);
    assert_eq!((a ^ b).sum(), 8 * 0b0110);
    assert_eq!((a << 2).sum(), 8 * 40);
    assert_eq!((b >> 2).sum(), 8 * 3);

    assert_eq!((a << 16).sum(), 8 * 0b1010);
    assert_eq!((I16x8::splat(WasmSimd128, -1) >> 16).sum(), -8);
    assert_eq!((I16x8::splat(WasmSimd128, 1) >> 16).sum(), 8);
    assert_eq!(
        (U16x8::splat(WasmSimd128, u16::MAX) >> 16).sum(),
        u16::MAX.wrapping_mul(8)
    );
}

#[test]
fn native_operations_match_integer_and_float_edge_semantics() {
    native_multiplication_and_float_unary_ops();
    i64_relational_comparisons_cover_signed_and_unsigned_high_words();
    float_nan_and_signed_zero_comparison_and_reduction();
    generated_native_bitwise_and_shift_operators();
}
