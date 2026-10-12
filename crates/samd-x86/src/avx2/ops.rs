//! AVX2 operation mappings and the single shared operator declaration.

use core::arch::x86_64;
use core::cmp::{max, min};
use core::marker::PhantomData;
use core::mem::transmute;
use samd_core::impl_simd_ops;

use super::{
    Avx2, Avx2Vector, CompareOp, F32x8, F64x4, I8x32, I16x16, I32x8, I64x4, Lane, U8x32, U16x16,
    U32x8, U64x4,
};

macro_rules! integer_lane {
    ($ty:ty, $signed:ty, $array:ty, $lanes:literal, $splat:path, $cmp:path, $zero:expr, $one:expr, $min:expr, $max:expr) => {
        // SAFETY: primitive integer lanes have all bit patterns valid and the
        // supplied array contains exactly 32 bytes.
        unsafe impl Lane for $ty {
            const LANES: usize = $lanes;
            type Array = $array;

            #[inline]
            unsafe fn splat(value: Self) -> x86_64::__m256i {
                // SAFETY: callers establish AVX2 support before constructing vectors.
                let signed = <$signed>::from_ne_bytes(value.to_ne_bytes());
                unsafe { $splat(signed) }
            }

            #[inline]
            unsafe fn compare(
                op: CompareOp,
                lhs: x86_64::__m256i,
                rhs: x86_64::__m256i,
            ) -> x86_64::__m256i {
                // SAFETY: AVX2 values are created only after runtime feature detection.
                unsafe { $cmp(op, lhs, rhs) }
            }

            #[inline]
            fn to_array(bits: x86_64::__m256i) -> Self::Array {
                // SAFETY: every bit pattern is valid for primitive integer lanes,
                // and the array and register have the same size.
                unsafe { transmute(bits) }
            }

            #[inline]
            fn from_array(values: Self::Array) -> x86_64::__m256i {
                // SAFETY: the array and register have equal sizes, and the array
                // contains only valid primitive integer lane representations.
                unsafe { transmute(values) }
            }

            #[inline]
            fn add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline]
            fn mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline]
            fn min(self, rhs: Self) -> Self {
                min(self, rhs)
            }

            #[inline]
            fn max(self, rhs: Self) -> Self {
                max(self, rhs)
            }

            #[inline]
            fn zero() -> Self {
                $zero
            }
            #[inline]
            fn one() -> Self {
                $one
            }
            #[inline]
            fn min_identity() -> Self {
                $min
            }
            #[inline]
            fn max_identity() -> Self {
                $max
            }
        }
    };
}

macro_rules! float_lane {
    ($ty:ty, $array:ty, $lanes:literal, $splat:path, $cast:ident, $cmp:path, $zero:expr, $one:expr, $min:expr, $max:expr) => {
        // SAFETY: every floating-point bit pattern is valid and the supplied
        // array contains exactly 32 bytes.
        unsafe impl Lane for $ty {
            const LANES: usize = $lanes;
            type Array = $array;

            #[inline]
            unsafe fn splat(value: Self) -> x86_64::__m256i {
                // SAFETY: callers establish AVX2 support before constructing vectors.
                unsafe { x86_64::$cast($splat(value)) }
            }

            #[inline]
            unsafe fn compare(
                op: CompareOp,
                lhs: x86_64::__m256i,
                rhs: x86_64::__m256i,
            ) -> x86_64::__m256i {
                // SAFETY: AVX2 values are created only after runtime feature detection.
                unsafe { $cmp(op, lhs, rhs) }
            }

            #[inline]
            fn to_array(bits: x86_64::__m256i) -> Self::Array {
                // SAFETY: every floating-point bit pattern is valid, and the
                // array and register have the same size.
                unsafe { transmute(bits) }
            }

            #[inline]
            fn from_array(values: Self::Array) -> x86_64::__m256i {
                // SAFETY: the array and register have equal sizes, and every
                // floating-point bit pattern is valid.
                unsafe { transmute(values) }
            }

            #[inline]
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            #[inline]
            fn min(self, rhs: Self) -> Self {
                if self == 0.0 && rhs == 0.0 && (self.is_sign_negative() || rhs.is_sign_negative())
                {
                    -0.0
                } else {
                    self.min(rhs)
                }
            }
            #[inline]
            fn max(self, rhs: Self) -> Self {
                if self == 0.0 && rhs == 0.0 && !(self.is_sign_negative() && rhs.is_sign_negative())
                {
                    0.0
                } else {
                    self.max(rhs)
                }
            }
            #[inline]
            fn zero() -> Self {
                $zero
            }
            #[inline]
            fn one() -> Self {
                $one
            }
            #[inline]
            fn min_identity() -> Self {
                $min
            }
            #[inline]
            fn max_identity() -> Self {
                $max
            }
        }
    };
}

macro_rules! integer_compare {
    ($name:ident, $eq:path, $gt:path) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(
            op: CompareOp,
            lhs: x86_64::__m256i,
            rhs: x86_64::__m256i,
        ) -> x86_64::__m256i {
            let equal = $eq(lhs, rhs);
            let all_ones = x86_64::_mm256_set1_epi32(-1);
            match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => x86_64::_mm256_xor_si256(equal, all_ones),
                CompareOp::Lt => $gt(rhs, lhs),
                CompareOp::Le => x86_64::_mm256_xor_si256($gt(lhs, rhs), all_ones),
                CompareOp::Gt => $gt(lhs, rhs),
                CompareOp::Ge => x86_64::_mm256_xor_si256($gt(rhs, lhs), all_ones),
            }
        }
    };
}

integer_compare!(
    compare_i8,
    x86_64::_mm256_cmpeq_epi8,
    x86_64::_mm256_cmpgt_epi8
);
integer_compare!(
    compare_i16,
    x86_64::_mm256_cmpeq_epi16,
    x86_64::_mm256_cmpgt_epi16
);
integer_compare!(
    compare_i32,
    x86_64::_mm256_cmpeq_epi32,
    x86_64::_mm256_cmpgt_epi32
);
integer_compare!(
    compare_i64,
    x86_64::_mm256_cmpeq_epi64,
    x86_64::_mm256_cmpgt_epi64
);

#[target_feature(enable = "avx2")]
unsafe fn compare_u8(op: CompareOp, lhs: x86_64::__m256i, rhs: x86_64::__m256i) -> x86_64::__m256i {
    let bias = x86_64::_mm256_set1_epi8(i8::MIN);
    // SAFETY: this helper and compare_i8 require AVX2.
    unsafe {
        compare_i8(
            op,
            x86_64::_mm256_xor_si256(lhs, bias),
            x86_64::_mm256_xor_si256(rhs, bias),
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn compare_u16(
    op: CompareOp,
    lhs: x86_64::__m256i,
    rhs: x86_64::__m256i,
) -> x86_64::__m256i {
    let bias = x86_64::_mm256_set1_epi16(i16::MIN);
    // SAFETY: this helper and compare_i16 require AVX2.
    unsafe {
        compare_i16(
            op,
            x86_64::_mm256_xor_si256(lhs, bias),
            x86_64::_mm256_xor_si256(rhs, bias),
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn compare_u32(
    op: CompareOp,
    lhs: x86_64::__m256i,
    rhs: x86_64::__m256i,
) -> x86_64::__m256i {
    let bias = x86_64::_mm256_set1_epi32(i32::MIN);
    // SAFETY: this helper and compare_i32 require AVX2.
    unsafe {
        compare_i32(
            op,
            x86_64::_mm256_xor_si256(lhs, bias),
            x86_64::_mm256_xor_si256(rhs, bias),
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn compare_u64(
    op: CompareOp,
    lhs: x86_64::__m256i,
    rhs: x86_64::__m256i,
) -> x86_64::__m256i {
    let bias = x86_64::_mm256_set1_epi64x(i64::MIN);
    // SAFETY: this helper and compare_i64 require AVX2.
    unsafe {
        compare_i64(
            op,
            x86_64::_mm256_xor_si256(lhs, bias),
            x86_64::_mm256_xor_si256(rhs, bias),
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn compare_f32(
    op: CompareOp,
    lhs: x86_64::__m256i,
    rhs: x86_64::__m256i,
) -> x86_64::__m256i {
    let lhs = x86_64::_mm256_castsi256_ps(lhs);
    let rhs = x86_64::_mm256_castsi256_ps(rhs);
    let mask = match op {
        CompareOp::Eq => x86_64::_mm256_cmp_ps::<{ x86_64::_CMP_EQ_OQ }>(lhs, rhs),
        CompareOp::Ne => x86_64::_mm256_cmp_ps::<{ x86_64::_CMP_NEQ_UQ }>(lhs, rhs),
        CompareOp::Lt => x86_64::_mm256_cmp_ps::<{ x86_64::_CMP_LT_OQ }>(lhs, rhs),
        CompareOp::Le => x86_64::_mm256_cmp_ps::<{ x86_64::_CMP_LE_OQ }>(lhs, rhs),
        CompareOp::Gt => x86_64::_mm256_cmp_ps::<{ x86_64::_CMP_GT_OQ }>(lhs, rhs),
        CompareOp::Ge => x86_64::_mm256_cmp_ps::<{ x86_64::_CMP_GE_OQ }>(lhs, rhs),
    };
    x86_64::_mm256_castps_si256(mask)
}

#[target_feature(enable = "avx2")]
unsafe fn compare_f64(
    op: CompareOp,
    lhs: x86_64::__m256i,
    rhs: x86_64::__m256i,
) -> x86_64::__m256i {
    let lhs = x86_64::_mm256_castsi256_pd(lhs);
    let rhs = x86_64::_mm256_castsi256_pd(rhs);
    let mask = match op {
        CompareOp::Eq => x86_64::_mm256_cmp_pd::<{ x86_64::_CMP_EQ_OQ }>(lhs, rhs),
        CompareOp::Ne => x86_64::_mm256_cmp_pd::<{ x86_64::_CMP_NEQ_UQ }>(lhs, rhs),
        CompareOp::Lt => x86_64::_mm256_cmp_pd::<{ x86_64::_CMP_LT_OQ }>(lhs, rhs),
        CompareOp::Le => x86_64::_mm256_cmp_pd::<{ x86_64::_CMP_LE_OQ }>(lhs, rhs),
        CompareOp::Gt => x86_64::_mm256_cmp_pd::<{ x86_64::_CMP_GT_OQ }>(lhs, rhs),
        CompareOp::Ge => x86_64::_mm256_cmp_pd::<{ x86_64::_CMP_GE_OQ }>(lhs, rhs),
    };
    x86_64::_mm256_castpd_si256(mask)
}

integer_lane!(
    i8,
    i8,
    [i8; 32],
    32,
    x86_64::_mm256_set1_epi8,
    compare_i8,
    0,
    1,
    i8::MAX,
    i8::MIN
);
integer_lane!(
    u8,
    i8,
    [u8; 32],
    32,
    x86_64::_mm256_set1_epi8,
    compare_u8,
    0,
    1,
    u8::MAX,
    u8::MIN
);
integer_lane!(
    i16,
    i16,
    [i16; 16],
    16,
    x86_64::_mm256_set1_epi16,
    compare_i16,
    0,
    1,
    i16::MAX,
    i16::MIN
);
integer_lane!(
    u16,
    i16,
    [u16; 16],
    16,
    x86_64::_mm256_set1_epi16,
    compare_u16,
    0,
    1,
    u16::MAX,
    u16::MIN
);
integer_lane!(
    i32,
    i32,
    [i32; 8],
    8,
    x86_64::_mm256_set1_epi32,
    compare_i32,
    0,
    1,
    i32::MAX,
    i32::MIN
);
integer_lane!(
    u32,
    i32,
    [u32; 8],
    8,
    x86_64::_mm256_set1_epi32,
    compare_u32,
    0,
    1,
    u32::MAX,
    u32::MIN
);
integer_lane!(
    i64,
    i64,
    [i64; 4],
    4,
    x86_64::_mm256_set1_epi64x,
    compare_i64,
    0,
    1,
    i64::MAX,
    i64::MIN
);
integer_lane!(
    u64,
    i64,
    [u64; 4],
    4,
    x86_64::_mm256_set1_epi64x,
    compare_u64,
    0,
    1,
    u64::MAX,
    u64::MIN
);
float_lane!(
    f32,
    [f32; 8],
    8,
    x86_64::_mm256_set1_ps,
    _mm256_castps_si256,
    compare_f32,
    0.0,
    1.0,
    f32::INFINITY,
    f32::NEG_INFINITY
);
float_lane!(
    f64,
    [f64; 4],
    4,
    x86_64::_mm256_set1_pd,
    _mm256_castpd_si256,
    compare_f64,
    0.0,
    1.0,
    f64::INFINITY,
    f64::NEG_INFINITY
);

macro_rules! binary_int {
    ($safe:ident, $name:ident, $ty:ty, $lanes:literal, $intrinsic:path) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(
            lhs: Avx2Vector<$ty, $lanes>,
            rhs: Avx2Vector<$ty, $lanes>,
        ) -> Avx2Vector<$ty, $lanes> {
            Avx2Vector {
                bits: $intrinsic(lhs.bits, rhs.bits),
                _element: PhantomData,
            }
        }
        #[inline]
        fn $safe(
            lhs: Avx2Vector<$ty, $lanes>,
            rhs: Avx2Vector<$ty, $lanes>,
        ) -> Avx2Vector<$ty, $lanes> {
            // SAFETY: the dispatch boundary checks AVX2 before creating `Avx2`.
            unsafe { $name(lhs, rhs) }
        }
    };
}

macro_rules! binary_float {
    ($safe:ident, $name:ident, $ty:ty, $lanes:literal, $cast:ident, $uncast:ident, $intrinsic:path) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(
            lhs: Avx2Vector<$ty, $lanes>,
            rhs: Avx2Vector<$ty, $lanes>,
        ) -> Avx2Vector<$ty, $lanes> {
            let a = x86_64::$cast(lhs.bits);
            let b = x86_64::$cast(rhs.bits);
            let result = $intrinsic(a, b);
            Avx2Vector {
                bits: x86_64::$uncast(result),
                _element: PhantomData,
            }
        }
        #[inline]
        fn $safe(
            lhs: Avx2Vector<$ty, $lanes>,
            rhs: Avx2Vector<$ty, $lanes>,
        ) -> Avx2Vector<$ty, $lanes> {
            // SAFETY: the dispatch boundary checks AVX2 before creating `Avx2`.
            unsafe { $name(lhs, rhs) }
        }
    };
}

macro_rules! unary_int {
    ($safe:ident, $name:ident, $ty:ty, $lanes:literal, $intrinsic:path) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(value: Avx2Vector<$ty, $lanes>) -> Avx2Vector<$ty, $lanes> {
            Avx2Vector {
                bits: $intrinsic(value.bits),
                _element: PhantomData,
            }
        }
        #[inline]
        fn $safe(value: Avx2Vector<$ty, $lanes>) -> Avx2Vector<$ty, $lanes> {
            // SAFETY: the dispatch boundary checks AVX2 before creating `Avx2`.
            unsafe { $name(value) }
        }
    };
}

macro_rules! unary_float {
    ($safe:ident, $name:ident, $ty:ty, $lanes:literal, $cast:ident, $uncast:ident, $intrinsic:path, $input:expr) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(value: Avx2Vector<$ty, $lanes>) -> Avx2Vector<$ty, $lanes> {
            let value = x86_64::$cast(value.bits);
            let result = $intrinsic($input, value);
            Avx2Vector {
                bits: x86_64::$uncast(result),
                _element: PhantomData,
            }
        }
        #[inline]
        fn $safe(value: Avx2Vector<$ty, $lanes>) -> Avx2Vector<$ty, $lanes> {
            // SAFETY: the dispatch boundary checks AVX2 before creating `Avx2`.
            unsafe { $name(value) }
        }
    };
}

macro_rules! unary_sqrt {
    ($safe:ident, $name:ident, $ty:ty, $lanes:literal, $cast:ident, $uncast:ident, $intrinsic:path) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(value: Avx2Vector<$ty, $lanes>) -> Avx2Vector<$ty, $lanes> {
            let value = x86_64::$cast(value.bits);
            let result = $intrinsic(value);
            Avx2Vector {
                bits: x86_64::$uncast(result),
                _element: PhantomData,
            }
        }
        #[inline]
        fn $safe(value: Avx2Vector<$ty, $lanes>) -> Avx2Vector<$ty, $lanes> {
            // SAFETY: the dispatch boundary checks AVX2 before creating `Avx2`.
            unsafe { $name(value) }
        }
    };
}

macro_rules! shift_int {
    ($safe:ident, $name:ident, $ty:ty, $lanes:literal, $bits:literal, $intrinsic:path) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(value: Avx2Vector<$ty, $lanes>, count: u32) -> Avx2Vector<$ty, $lanes> {
            let count = i32::try_from(count & ($bits - 1)).expect("masked shift count fits i32");
            let count = x86_64::_mm_cvtsi32_si128(count);
            Avx2Vector {
                bits: $intrinsic(value.bits, count),
                _element: PhantomData,
            }
        }
        #[inline]
        fn $safe(value: Avx2Vector<$ty, $lanes>, count: u32) -> Avx2Vector<$ty, $lanes> {
            // SAFETY: the dispatch boundary checks AVX2 before creating `Avx2`.
            unsafe { $name(value, count) }
        }
    };
}

macro_rules! binary_bitwise {
    ($safe:ident, $native:ident, $intrinsic:path) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $native<T: Lane, const LANES: usize>(
            lhs: Avx2Vector<T, LANES>,
            rhs: Avx2Vector<T, LANES>,
        ) -> Avx2Vector<T, LANES> {
            Avx2Vector {
                bits: $intrinsic(lhs.bits, rhs.bits),
                _element: PhantomData,
            }
        }

        #[inline]
        fn $safe<T: Lane, const LANES: usize>(
            lhs: Avx2Vector<T, LANES>,
            rhs: Avx2Vector<T, LANES>,
        ) -> Avx2Vector<T, LANES> {
            // SAFETY: the dispatch boundary checks AVX2 before creating `Avx2`.
            unsafe { $native(lhs, rhs) }
        }
    };
}

macro_rules! unary_bitwise {
    ($safe:ident, $native:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $native<T: Lane, const LANES: usize>(
            value: Avx2Vector<T, LANES>,
        ) -> Avx2Vector<T, LANES> {
            Avx2Vector {
                bits: x86_64::_mm256_xor_si256(value.bits, x86_64::_mm256_set1_epi32(-1)),
                _element: PhantomData,
            }
        }

        #[inline]
        fn $safe<T: Lane, const LANES: usize>(value: Avx2Vector<T, LANES>) -> Avx2Vector<T, LANES> {
            // SAFETY: the dispatch boundary checks AVX2 before creating `Avx2`.
            unsafe { $native(value) }
        }
    };
}

binary_bitwise!(bitand, bitand_avx2, x86_64::_mm256_and_si256);
binary_bitwise!(bitor, bitor_avx2, x86_64::_mm256_or_si256);
binary_bitwise!(bitxor, bitxor_avx2, x86_64::_mm256_xor_si256);
unary_bitwise!(bitnot, bitnot_avx2);

binary_int!(safe_add_i8, add_i8, i8, 32, x86_64::_mm256_add_epi8);
binary_int!(safe_sub_i8, sub_i8, i8, 32, x86_64::_mm256_sub_epi8);
binary_int!(safe_add_u8, add_u8, u8, 32, x86_64::_mm256_add_epi8);
binary_int!(safe_sub_u8, sub_u8, u8, 32, x86_64::_mm256_sub_epi8);
binary_int!(safe_add_i16, add_i16, i16, 16, x86_64::_mm256_add_epi16);
binary_int!(safe_sub_i16, sub_i16, i16, 16, x86_64::_mm256_sub_epi16);
binary_int!(safe_mul_i16, mul_i16, i16, 16, x86_64::_mm256_mullo_epi16);
binary_int!(safe_add_u16, add_u16, u16, 16, x86_64::_mm256_add_epi16);
binary_int!(safe_sub_u16, sub_u16, u16, 16, x86_64::_mm256_sub_epi16);
binary_int!(safe_mul_u16, mul_u16, u16, 16, x86_64::_mm256_mullo_epi16);
binary_int!(safe_add_i32, add_i32, i32, 8, x86_64::_mm256_add_epi32);
binary_int!(safe_sub_i32, sub_i32, i32, 8, x86_64::_mm256_sub_epi32);
binary_int!(safe_mul_i32, mul_i32, i32, 8, x86_64::_mm256_mullo_epi32);
binary_int!(safe_add_u32, add_u32, u32, 8, x86_64::_mm256_add_epi32);
binary_int!(safe_sub_u32, sub_u32, u32, 8, x86_64::_mm256_sub_epi32);
binary_int!(safe_mul_u32, mul_u32, u32, 8, x86_64::_mm256_mullo_epi32);
binary_int!(safe_add_i64, add_i64, i64, 4, x86_64::_mm256_add_epi64);
binary_int!(safe_sub_i64, sub_i64, i64, 4, x86_64::_mm256_sub_epi64);
binary_int!(safe_add_u64, add_u64, u64, 4, x86_64::_mm256_add_epi64);
binary_int!(safe_sub_u64, sub_u64, u64, 4, x86_64::_mm256_sub_epi64);
binary_float!(
    safe_add_f32,
    add_f32,
    f32,
    8,
    _mm256_castsi256_ps,
    _mm256_castps_si256,
    x86_64::_mm256_add_ps
);
binary_float!(
    safe_sub_f32,
    sub_f32,
    f32,
    8,
    _mm256_castsi256_ps,
    _mm256_castps_si256,
    x86_64::_mm256_sub_ps
);
binary_float!(
    safe_mul_f32,
    mul_f32,
    f32,
    8,
    _mm256_castsi256_ps,
    _mm256_castps_si256,
    x86_64::_mm256_mul_ps
);
binary_float!(
    safe_div_f32,
    div_f32,
    f32,
    8,
    _mm256_castsi256_ps,
    _mm256_castps_si256,
    x86_64::_mm256_div_ps
);
binary_float!(
    safe_add_f64,
    add_f64,
    f64,
    4,
    _mm256_castsi256_pd,
    _mm256_castpd_si256,
    x86_64::_mm256_add_pd
);
binary_float!(
    safe_sub_f64,
    sub_f64,
    f64,
    4,
    _mm256_castsi256_pd,
    _mm256_castpd_si256,
    x86_64::_mm256_sub_pd
);
binary_float!(
    safe_mul_f64,
    mul_f64,
    f64,
    4,
    _mm256_castsi256_pd,
    _mm256_castpd_si256,
    x86_64::_mm256_mul_pd
);
binary_float!(
    safe_div_f64,
    div_f64,
    f64,
    4,
    _mm256_castsi256_pd,
    _mm256_castpd_si256,
    x86_64::_mm256_div_pd
);
unary_int!(safe_abs_i32, abs_i32, i32, 8, x86_64::_mm256_abs_epi32);
unary_float!(
    safe_abs_f32,
    abs_f32,
    f32,
    8,
    _mm256_castsi256_ps,
    _mm256_castps_si256,
    x86_64::_mm256_andnot_ps,
    x86_64::_mm256_set1_ps(-0.0)
);
unary_sqrt!(
    safe_sqrt_f32,
    sqrt_f32,
    f32,
    8,
    _mm256_castsi256_ps,
    _mm256_castps_si256,
    x86_64::_mm256_sqrt_ps
);
unary_float!(
    safe_abs_f64,
    abs_f64,
    f64,
    4,
    _mm256_castsi256_pd,
    _mm256_castpd_si256,
    x86_64::_mm256_andnot_pd,
    x86_64::_mm256_set1_pd(-0.0)
);
unary_sqrt!(
    safe_sqrt_f64,
    sqrt_f64,
    f64,
    4,
    _mm256_castsi256_pd,
    _mm256_castpd_si256,
    x86_64::_mm256_sqrt_pd
);
shift_int!(safe_shl_i16, shl_i16, i16, 16, 16, x86_64::_mm256_sll_epi16);
shift_int!(safe_shr_i16, shr_i16, i16, 16, 16, x86_64::_mm256_sra_epi16);
shift_int!(safe_shl_u16, shl_u16, u16, 16, 16, x86_64::_mm256_sll_epi16);
shift_int!(safe_shr_u16, shr_u16, u16, 16, 16, x86_64::_mm256_srl_epi16);
shift_int!(safe_shl_i32, shl_i32, i32, 8, 32, x86_64::_mm256_sll_epi32);
shift_int!(safe_shr_i32, shr_i32, i32, 8, 32, x86_64::_mm256_sra_epi32);
shift_int!(safe_shl_u32, shl_u32, u32, 8, 32, x86_64::_mm256_sll_epi32);
shift_int!(safe_shr_u32, shr_u32, u32, 8, 32, x86_64::_mm256_srl_epi32);
shift_int!(safe_shl_i64, shl_i64, i64, 4, 64, x86_64::_mm256_sll_epi64);
shift_int!(safe_shl_u64, shl_u64, u64, 4, 64, x86_64::_mm256_sll_epi64);
shift_int!(safe_shr_u64, shr_u64, u64, 4, 64, x86_64::_mm256_srl_epi64);

impl_simd_ops! {
    Avx2;
    i8 => (I8x32, 32) { add: safe_add_i8, sub: safe_sub_i8, bitand: bitand::<i8, 32>, bitor: bitor::<i8, 32>, bitxor: bitxor::<i8, 32>, not: bitnot::<i8, 32>, }
    u8 => (U8x32, 32) { add: safe_add_u8, sub: safe_sub_u8, bitand: bitand::<u8, 32>, bitor: bitor::<u8, 32>, bitxor: bitxor::<u8, 32>, not: bitnot::<u8, 32>, }
    i16 => (I16x16, 16) { add: safe_add_i16, sub: safe_sub_i16, bitand: bitand::<i16, 16>, bitor: bitor::<i16, 16>, bitxor: bitxor::<i16, 16>, not: bitnot::<i16, 16>, mul: safe_mul_i16, shl: safe_shl_i16, shr: safe_shr_i16, }
    u16 => (U16x16, 16) { add: safe_add_u16, sub: safe_sub_u16, bitand: bitand::<u16, 16>, bitor: bitor::<u16, 16>, bitxor: bitxor::<u16, 16>, not: bitnot::<u16, 16>, mul: safe_mul_u16, shl: safe_shl_u16, shr: safe_shr_u16, }
    i32 => (I32x8, 8) { add: safe_add_i32, sub: safe_sub_i32, bitand: bitand::<i32, 8>, bitor: bitor::<i32, 8>, bitxor: bitxor::<i32, 8>, not: bitnot::<i32, 8>, mul: safe_mul_i32, abs: safe_abs_i32, shl: safe_shl_i32, shr: safe_shr_i32, }
    u32 => (U32x8, 8) { add: safe_add_u32, sub: safe_sub_u32, bitand: bitand::<u32, 8>, bitor: bitor::<u32, 8>, bitxor: bitxor::<u32, 8>, not: bitnot::<u32, 8>, mul: safe_mul_u32, shl: safe_shl_u32, shr: safe_shr_u32, }
    i64 => (I64x4, 4) { add: safe_add_i64, sub: safe_sub_i64, bitand: bitand::<i64, 4>, bitor: bitor::<i64, 4>, bitxor: bitxor::<i64, 4>, not: bitnot::<i64, 4>, shl: safe_shl_i64, }
    u64 => (U64x4, 4) { add: safe_add_u64, sub: safe_sub_u64, bitand: bitand::<u64, 4>, bitor: bitor::<u64, 4>, bitxor: bitxor::<u64, 4>, not: bitnot::<u64, 4>, shl: safe_shl_u64, shr: safe_shr_u64, }
    f32 => (F32x8, 8) { add: safe_add_f32, sub: safe_sub_f32, mul: safe_mul_f32, div: safe_div_f32, abs: safe_abs_f32, sqrt: safe_sqrt_f32, }
    f64 => (F64x4, 4) { add: safe_add_f64, sub: safe_sub_f64, mul: safe_mul_f64, div: safe_div_f64, abs: safe_abs_f64, sqrt: safe_sqrt_f64, }
}
