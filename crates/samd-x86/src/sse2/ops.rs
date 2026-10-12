//! SSE2 instruction mappings and generated operator implementations.

use core::arch::x86_64::{
    __m128i, _mm_add_epi8, _mm_add_epi16, _mm_add_epi32, _mm_add_pd, _mm_add_ps, _mm_and_pd,
    _mm_and_ps, _mm_and_si128, _mm_castpd_si128, _mm_castps_si128, _mm_castsi128_pd,
    _mm_castsi128_ps, _mm_cmpgt_epi8, _mm_cmpgt_epi16, _mm_cmpgt_epi32, _mm_cvtsi32_si128,
    _mm_div_pd, _mm_div_ps, _mm_mul_epu32, _mm_mul_pd, _mm_mul_ps, _mm_mullo_epi16, _mm_or_si128,
    _mm_set_epi32, _mm_set1_epi32, _mm_set1_pd, _mm_set1_ps, _mm_setzero_si128, _mm_shuffle_epi32,
    _mm_sll_epi16, _mm_sll_epi32, _mm_sll_epi64, _mm_sqrt_pd, _mm_sqrt_ps, _mm_sra_epi16,
    _mm_sra_epi32, _mm_srai_epi32, _mm_srl_epi16, _mm_srl_epi32, _mm_srl_epi64, _mm_srli_si128,
    _mm_sub_epi8, _mm_sub_epi16, _mm_sub_epi32, _mm_sub_pd, _mm_sub_ps, _mm_unpackhi_epi32,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_pd, _mm_xor_ps, _mm_xor_si128,
};
use core::marker::PhantomData;

use samd_core::impl_simd_ops;

use super::{
    F32x4, F64x2, I8x16, I16x8, I32x4, I64x2, Lane, Sse2, SseVector, U8x16, U16x8, U32x4, U64x2,
};

macro_rules! binary {
    ($name:ident, $ty:ty, $lanes:expr, $intrinsic:ident) => {
        #[inline]
        fn $name(a: SseVector<$ty, $lanes>, b: SseVector<$ty, $lanes>) -> SseVector<$ty, $lanes> {
            // SAFETY: this backend is entered only when SSE2 is available.
            SseVector(unsafe { $intrinsic(a.0, b.0) }, PhantomData)
        }
    };
}

macro_rules! unary {
    ($name:ident, $ty:ty, $lanes:expr, $body:expr) => {
        #[inline]
        fn $name(a: SseVector<$ty, $lanes>) -> SseVector<$ty, $lanes> {
            // SAFETY: this backend is entered only when SSE2 is available.
            SseVector(unsafe { $body(a.0) }, PhantomData)
        }
    };
}

macro_rules! binary_safe {
    ($name:ident, $ty:ty, $lanes:expr, $operation:ident) => {
        #[inline]
        fn $name(a: SseVector<$ty, $lanes>, b: SseVector<$ty, $lanes>) -> SseVector<$ty, $lanes> {
            SseVector($operation(a.0, b.0), PhantomData)
        }
    };
}

macro_rules! shift {
    ($name:ident, $ty:ty, $lanes:expr, $bits:expr, $intrinsic:ident) => {
        #[inline]
        fn $name(value: SseVector<$ty, $lanes>, count: u32) -> SseVector<$ty, $lanes> {
            // SAFETY: SSE2 is the required baseline and these intrinsics define
            // oversized shift counts as zero or sign-fill lanes.
            SseVector(
                unsafe {
                    $intrinsic(
                        value.0,
                        _mm_cvtsi32_si128((count & ($bits - 1)).cast_signed()),
                    )
                },
                PhantomData,
            )
        }
    };
}

/// Computes the low 32 bits of each product using SSE2's even-lane 32x32 multiply.
#[inline]
unsafe fn mullo_i32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: all operations are SSE2 and the intermediate products are valid
    // for every pair of 32-bit bit patterns.
    unsafe {
        let even = _mm_mul_epu32(a, b);
        let odd = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
        let even_low = _mm_shuffle_epi32::<0x88>(even);
        let odd_low = _mm_shuffle_epi32::<0x88>(odd);
        _mm_unpacklo_epi32(even_low, odd_low)
    }
}

binary!(add_i8, i8, 16, _mm_add_epi8);
binary!(sub_i8, i8, 16, _mm_sub_epi8);
binary!(add_u8, u8, 16, _mm_add_epi8);
binary!(sub_u8, u8, 16, _mm_sub_epi8);
binary!(add_i16, i16, 8, _mm_add_epi16);
binary!(sub_i16, i16, 8, _mm_sub_epi16);
binary!(mul_i16, i16, 8, _mm_mullo_epi16);
binary!(add_u16, u16, 8, _mm_add_epi16);
binary!(sub_u16, u16, 8, _mm_sub_epi16);
binary!(mul_u16, u16, 8, _mm_mullo_epi16);
binary!(add_i32, i32, 4, _mm_add_epi32);
binary!(sub_i32, i32, 4, _mm_sub_epi32);
binary!(mul_i32, i32, 4, mullo_i32);
binary!(add_u32, u32, 4, _mm_add_epi32);
binary!(sub_u32, u32, 4, _mm_sub_epi32);
binary!(mul_u32, u32, 4, mullo_i32);
binary_safe!(add_f32, f32, 4, add_f32_bits);
binary_safe!(sub_f32, f32, 4, sub_f32_bits);
binary_safe!(mul_f32, f32, 4, mul_f32_bits);
binary_safe!(div_f32, f32, 4, div_f32_bits);
binary_safe!(add_f64, f64, 2, add_f64_bits);
binary_safe!(sub_f64, f64, 2, sub_f64_bits);
binary_safe!(mul_f64, f64, 2, mul_f64_bits);
binary_safe!(div_f64, f64, 2, div_f64_bits);

// Floating-point registers share the exact 128-bit representation; these
// wrappers keep the implementation table uniform while using native float ops.
#[inline]
fn add_f32_bits(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is the required feature and the casts preserve register bits.
    unsafe { _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b))) }
}
#[inline]
fn sub_f32_bits(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is the required feature and the casts preserve register bits.
    unsafe { _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b))) }
}
#[inline]
fn mul_f32_bits(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is the required feature and the casts preserve register bits.
    unsafe { _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b))) }
}
#[inline]
fn div_f32_bits(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is the required feature and the casts preserve register bits.
    unsafe { _mm_castps_si128(_mm_div_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b))) }
}
#[inline]
fn add_f64_bits(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is the required feature and the casts preserve register bits.
    unsafe { _mm_castpd_si128(_mm_add_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b))) }
}
#[inline]
fn sub_f64_bits(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is the required feature and the casts preserve register bits.
    unsafe { _mm_castpd_si128(_mm_sub_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b))) }
}
#[inline]
fn mul_f64_bits(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is the required feature and the casts preserve register bits.
    unsafe { _mm_castpd_si128(_mm_mul_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b))) }
}
#[inline]
fn div_f64_bits(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is the required feature and the casts preserve register bits.
    unsafe { _mm_castpd_si128(_mm_div_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b))) }
}

unary!(neg_i8, i8, 16, |v| _mm_sub_epi8(_mm_setzero_si128(), v));
unary!(neg_i16, i16, 8, |v| _mm_sub_epi16(_mm_setzero_si128(), v));
unary!(neg_i32, i32, 4, |v| _mm_sub_epi32(_mm_setzero_si128(), v));
unary!(neg_i64, i64, 2, |v| sub_qword(_mm_setzero_si128(), v));
unary!(neg_f32, f32, 4, |v| _mm_castps_si128(_mm_xor_ps(
    _mm_castsi128_ps(v),
    _mm_set1_ps(-0.0),
)));
unary!(neg_f64, f64, 2, |v| _mm_castpd_si128(_mm_xor_pd(
    _mm_castsi128_pd(v),
    _mm_set1_pd(-0.0),
)));

unary!(abs_f32, f32, 4, |v| _mm_castps_si128(_mm_and_ps(
    _mm_castsi128_ps(v),
    _mm_castsi128_ps(_mm_set1_epi32(i32::MAX)),
)));
unary!(abs_f64, f64, 2, |v| _mm_castpd_si128(_mm_and_pd(
    _mm_castsi128_pd(v),
    _mm_castsi128_pd(_mm_set_epi32(i32::MAX, -1, i32::MAX, -1)),
)));
unary!(sqrt_f32, f32, 4, |v| _mm_castps_si128(_mm_sqrt_ps(
    _mm_castsi128_ps(v)
)));
unary!(sqrt_f64, f64, 2, |v| _mm_castpd_si128(_mm_sqrt_pd(
    _mm_castsi128_pd(v)
)));

unary!(abs_i8, i8, 16, |v| {
    let sign = _mm_cmpgt_epi8(_mm_setzero_si128(), v);
    _mm_sub_epi8(_mm_xor_si128(v, sign), sign)
});
unary!(abs_i16, i16, 8, |v| {
    let sign = _mm_cmpgt_epi16(_mm_setzero_si128(), v);
    _mm_sub_epi16(_mm_xor_si128(v, sign), sign)
});
unary!(abs_i32, i32, 4, |v| {
    let sign = _mm_cmpgt_epi32(_mm_setzero_si128(), v);
    _mm_sub_epi32(_mm_xor_si128(v, sign), sign)
});
unary!(abs_i64, i64, 2, |v| {
    let high = _mm_shuffle_epi32::<0xF5>(v);
    let sign = _mm_srai_epi32::<31>(high);
    sub_qword(_mm_xor_si128(v, sign), sign)
});

#[inline]
fn bitand<T: Lane, const LANES: usize>(
    lhs: SseVector<T, LANES>,
    rhs: SseVector<T, LANES>,
) -> SseVector<T, LANES> {
    // SAFETY: SSE2 is guaranteed on x86_64.
    SseVector(unsafe { _mm_and_si128(lhs.0, rhs.0) }, PhantomData)
}

#[inline]
fn bitor<T: Lane, const LANES: usize>(
    lhs: SseVector<T, LANES>,
    rhs: SseVector<T, LANES>,
) -> SseVector<T, LANES> {
    // SAFETY: SSE2 is guaranteed on x86_64.
    SseVector(unsafe { _mm_or_si128(lhs.0, rhs.0) }, PhantomData)
}

#[inline]
fn bitxor<T: Lane, const LANES: usize>(
    lhs: SseVector<T, LANES>,
    rhs: SseVector<T, LANES>,
) -> SseVector<T, LANES> {
    // SAFETY: SSE2 is guaranteed on x86_64.
    SseVector(unsafe { _mm_xor_si128(lhs.0, rhs.0) }, PhantomData)
}

#[inline]
fn bitnot<T: Lane, const LANES: usize>(value: SseVector<T, LANES>) -> SseVector<T, LANES> {
    // SAFETY: SSE2 is guaranteed on x86_64.
    SseVector(
        unsafe { _mm_xor_si128(value.0, _mm_set1_epi32(-1)) },
        PhantomData,
    )
}

shift!(shl_i16, i16, 8, 16, _mm_sll_epi16);
shift!(shr_i16, i16, 8, 16, _mm_sra_epi16);
shift!(shl_u16, u16, 8, 16, _mm_sll_epi16);
shift!(shr_u16, u16, 8, 16, _mm_srl_epi16);
shift!(shl_i32, i32, 4, 32, _mm_sll_epi32);
shift!(shr_i32, i32, 4, 32, _mm_sra_epi32);
shift!(shl_u32, u32, 4, 32, _mm_sll_epi32);
shift!(shr_u32, u32, 4, 32, _mm_srl_epi32);
shift!(shl_i64, i64, 2, 64, _mm_sll_epi64);
shift!(shl_u64, u64, 2, 64, _mm_sll_epi64);
shift!(shr_u64, u64, 2, 64, _mm_srl_epi64);

// SSE2 has no 64-bit packed add/sub. This composes the operation from 32-bit
// adds/subtracts and an unsigned carry/borrow comparison in native registers.
#[inline]
unsafe fn add_qword(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: all used operations are SSE2.
    unsafe {
        let a_low = _mm_shuffle_epi32::<0xA0>(a);
        let b_low = _mm_shuffle_epi32::<0xA0>(b);
        let a_high = _mm_shuffle_epi32::<0xF5>(a);
        let b_high = _mm_shuffle_epi32::<0xF5>(b);
        let low = _mm_add_epi32(a_low, b_low);
        let sign = _mm_set1_epi32(i32::MIN);
        let carry = _mm_cmpgt_epi32(_mm_xor_si128(a_low, sign), _mm_xor_si128(low, sign));
        let high = _mm_add_epi32(
            _mm_add_epi32(a_high, b_high),
            _mm_and_si128(carry, _mm_set1_epi32(1)),
        );
        let q0 = _mm_unpacklo_epi32(low, high);
        let q1 = _mm_unpackhi_epi32(low, high);
        _mm_unpacklo_epi64(q0, q1)
    }
}

#[inline]
unsafe fn sub_qword(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: all used operations are SSE2.
    unsafe {
        let a_low = _mm_shuffle_epi32::<0xA0>(a);
        let b_low = _mm_shuffle_epi32::<0xA0>(b);
        let a_high = _mm_shuffle_epi32::<0xF5>(a);
        let b_high = _mm_shuffle_epi32::<0xF5>(b);
        let low = _mm_sub_epi32(a_low, b_low);
        let sign = _mm_set1_epi32(i32::MIN);
        let borrow = _mm_cmpgt_epi32(_mm_xor_si128(b_low, sign), _mm_xor_si128(a_low, sign));
        let high = _mm_sub_epi32(
            _mm_sub_epi32(a_high, b_high),
            _mm_and_si128(borrow, _mm_set1_epi32(1)),
        );
        let q0 = _mm_unpacklo_epi32(low, high);
        let q1 = _mm_unpackhi_epi32(low, high);
        _mm_unpacklo_epi64(q0, q1)
    }
}

binary!(add_i64, i64, 2, add_qword);
binary!(sub_i64, i64, 2, sub_qword);
binary!(add_u64, u64, 2, add_qword);
binary!(sub_u64, u64, 2, sub_qword);

impl_simd_ops! {
    Sse2;
    i8 => (I8x16, 16) { add: add_i8, sub: sub_i8, neg: neg_i8, abs: abs_i8, bitand: bitand::<i8, 16>, bitor: bitor::<i8, 16>, bitxor: bitxor::<i8, 16>, not: bitnot::<i8, 16>, }
    u8 => (U8x16, 16) { add: add_u8, sub: sub_u8, bitand: bitand::<u8, 16>, bitor: bitor::<u8, 16>, bitxor: bitxor::<u8, 16>, not: bitnot::<u8, 16>, }
    i16 => (I16x8, 8) { add: add_i16, sub: sub_i16, mul: mul_i16, neg: neg_i16, abs: abs_i16, bitand: bitand::<i16, 8>, bitor: bitor::<i16, 8>, bitxor: bitxor::<i16, 8>, not: bitnot::<i16, 8>, shl: shl_i16, shr: shr_i16, }
    u16 => (U16x8, 8) { add: add_u16, sub: sub_u16, mul: mul_u16, bitand: bitand::<u16, 8>, bitor: bitor::<u16, 8>, bitxor: bitxor::<u16, 8>, not: bitnot::<u16, 8>, shl: shl_u16, shr: shr_u16, }
    i32 => (I32x4, 4) { add: add_i32, sub: sub_i32, mul: mul_i32, neg: neg_i32, abs: abs_i32, bitand: bitand::<i32, 4>, bitor: bitor::<i32, 4>, bitxor: bitxor::<i32, 4>, not: bitnot::<i32, 4>, shl: shl_i32, shr: shr_i32, }
    u32 => (U32x4, 4) { add: add_u32, sub: sub_u32, mul: mul_u32, bitand: bitand::<u32, 4>, bitor: bitor::<u32, 4>, bitxor: bitxor::<u32, 4>, not: bitnot::<u32, 4>, shl: shl_u32, shr: shr_u32, }
    i64 => (I64x2, 2) { add: add_i64, sub: sub_i64, neg: neg_i64, abs: abs_i64, bitand: bitand::<i64, 2>, bitor: bitor::<i64, 2>, bitxor: bitxor::<i64, 2>, not: bitnot::<i64, 2>, shl: shl_i64, }
    u64 => (U64x2, 2) { add: add_u64, sub: sub_u64, bitand: bitand::<u64, 2>, bitor: bitor::<u64, 2>, bitxor: bitxor::<u64, 2>, not: bitnot::<u64, 2>, shl: shl_u64, shr: shr_u64, }
    f32 => (F32x4, 4) { add: add_f32, sub: sub_f32, mul: mul_f32, div: div_f32, neg: neg_f32, abs: abs_f32, sqrt: sqrt_f32, }
    f64 => (F64x2, 2) { add: add_f64, sub: sub_f64, mul: mul_f64, div: div_f64, neg: neg_f64, abs: abs_f64, sqrt: sqrt_f64, }
}
