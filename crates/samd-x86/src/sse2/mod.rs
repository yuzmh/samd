//! Native 128-bit SSE2 backend for `x86_64`.
//!
//! Vector values wrap one native `__m128i` register. Floating-point values use
//! the same bits and are cast to native `__m128`/`__m128d` registers for their
//! operations. Full loads and stores use unaligned SSE2 accesses; partial
//! chunks use initialized stack storage and touch only active slice elements.

use core::arch::x86_64::{
    __m128i, _mm_and_si128, _mm_andnot_si128, _mm_castpd_si128, _mm_castps_si128, _mm_castsi128_pd,
    _mm_castsi128_ps, _mm_cmpeq_epi8, _mm_cmpeq_epi16, _mm_cmpeq_epi32, _mm_cmpeq_pd, _mm_cmpeq_ps,
    _mm_cmpge_pd, _mm_cmpge_ps, _mm_cmpgt_epi8, _mm_cmpgt_epi16, _mm_cmpgt_epi32, _mm_cmpgt_pd,
    _mm_cmpgt_ps, _mm_cmple_pd, _mm_cmple_ps, _mm_cmplt_pd, _mm_cmplt_ps, _mm_cmpneq_pd,
    _mm_cmpneq_ps, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8, _mm_set1_epi16,
    _mm_set1_epi32, _mm_set1_epi64x, _mm_set1_pd, _mm_set1_ps, _mm_storeu_si128, _mm_xor_si128,
};
use core::array;
use core::marker::PhantomData;

use samd_core::{MaskSelect, Simd, SimdMask, SimdVector};

mod ops;

/// Marker selecting the SSE2 backend.
///
/// SSE2 is guaranteed by the `x86_64` ABI.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sse2;

impl Simd for Sse2 {}

/// Internal scalar metadata and native comparison implementation.
///
/// # Safety
///
/// Implementors must have no invalid bit patterns and exactly fill one 128-bit
/// register across `LANES` elements. Comparison results must contain either
/// zero or all-one bits in each logical lane.
#[doc(hidden)]
pub unsafe trait Lane: Copy + Default {
    /// Logical lanes in an SSE2 register.
    const LANES: usize;

    /// Constructs a native splat.
    fn splat(value: Self) -> __m128i;

    /// Compares native vectors using the requested relation.
    fn compare(a: __m128i, b: __m128i, relation: CompareOp) -> __m128i;

    /// Scalar reduction addition, with integer wrapping semantics.
    #[must_use]
    fn add(self, rhs: Self) -> Self;

    /// Scalar reduction multiplication, with integer wrapping semantics.
    #[must_use]
    fn mul(self, rhs: Self) -> Self;

    /// Minimum-number reduction semantics for floating-point values.
    #[must_use]
    fn min(self, rhs: Self) -> Self;

    /// Maximum-number reduction semantics for floating-point values.
    #[must_use]
    fn max(self, rhs: Self) -> Self;

    /// Product reduction identity.
    fn one() -> Self;

    /// Minimum reduction identity.
    fn min_identity() -> Self;

    /// Maximum reduction identity.
    fn max_identity() -> Self;
}

/// CompareOp used by [`Lane::compare`].
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum CompareOp {
    /// Equality.
    Eq,
    /// Inequality, including unordered floating-point values.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// SSE2 vector representation backed by one native 128-bit register.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct SseVector<T: Lane, const LANES: usize>(__m128i, PhantomData<T>);

/// SSE2 predicate representation backed by one native 128-bit register.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct SseMask<T: Lane, const LANES: usize>(__m128i, PhantomData<T>);

/// Sixteen signed byte lanes.
pub type I8x16 = SseVector<i8, 16>;
/// Sixteen unsigned byte lanes.
pub type U8x16 = SseVector<u8, 16>;
/// Eight signed 16-bit lanes.
pub type I16x8 = SseVector<i16, 8>;
/// Eight unsigned 16-bit lanes.
pub type U16x8 = SseVector<u16, 8>;
/// Four signed 32-bit lanes.
pub type I32x4 = SseVector<i32, 4>;
/// Four unsigned 32-bit lanes.
pub type U32x4 = SseVector<u32, 4>;
/// Two signed 64-bit lanes.
pub type I64x2 = SseVector<i64, 2>;
/// Two unsigned 64-bit lanes.
pub type U64x2 = SseVector<u64, 2>;
/// Four `f32` lanes.
pub type F32x4 = SseVector<f32, 4>;
/// Two `f64` lanes.
pub type F64x2 = SseVector<f64, 2>;

impl<T: Lane, const LANES: usize> SimdMask for SseMask<T, LANES> {
    #[inline]
    fn any(self) -> bool {
        self.active_any(LANES)
    }

    #[inline]
    fn all(self) -> bool {
        self.active_all(LANES)
    }

    #[inline]
    fn count_true(self) -> usize {
        self.active_count_true(LANES)
    }

    #[inline]
    fn active_any(self, active: usize) -> bool {
        assert_eq!(LANES, T::LANES, "mask type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed mask width");
        lane_mask::<T, LANES>(self.0) & lane_bits(active) != 0
    }

    #[inline]
    fn active_all(self, active: usize) -> bool {
        assert_eq!(LANES, T::LANES, "mask type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed mask width");
        let active_bits = lane_bits(active);
        lane_mask::<T, LANES>(self.0) & active_bits == active_bits
    }

    #[inline]
    fn active_count_true(self, active: usize) -> usize {
        assert_eq!(LANES, T::LANES, "mask type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed mask width");
        (lane_mask::<T, LANES>(self.0) & lane_bits(active)).count_ones() as usize
    }
}

impl<T: Lane, const LANES: usize> MaskSelect<SseMask<T, LANES>> for SseVector<T, LANES> {
    #[inline]
    fn select(mask: SseMask<T, LANES>, if_true: Self, if_false: Self) -> Self {
        // SAFETY: SSE2 is the target feature required by this backend.
        Self(
            unsafe {
                _mm_or_si128(
                    _mm_and_si128(mask.0, if_true.0),
                    _mm_andnot_si128(mask.0, if_false.0),
                )
            },
            PhantomData,
        )
    }
}

#[inline]
const fn lane_bits(active: usize) -> u32 {
    if active == 16 {
        u32::MAX
    } else {
        (1_u32 << active) - 1
    }
}

#[inline]
fn lane_mask<T: Lane, const LANES: usize>(mask: __m128i) -> u32 {
    assert_eq!(LANES, T::LANES, "mask type has an invalid lane width");
    // SAFETY: `_mm_movemask_epi8` is an SSE2 operation and `mask` is a valid
    // native register. Each comparison sets every byte of a true lane.
    let bytes = unsafe { _mm_movemask_epi8(mask) }.cast_unsigned();
    let width = 16 / LANES;
    (0..LANES).fold(0, |bits, lane| {
        bits | (((bytes >> (lane * width)) & 1) << lane)
    })
}

// SAFETY: all access methods validate lane count and slice bounds. All scalar
// lane types are primitive numeric types with valid bit patterns.
unsafe impl<T: Lane, const LANES: usize> SimdVector<T, Sse2> for SseVector<T, LANES> {
    type Mask = SseMask<T, LANES>;

    #[inline]
    fn load(_: Sse2, slice: &[T], offset: usize, active: usize) -> Self {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed vector width");
        assert!(
            offset <= slice.len() && active <= slice.len() - offset,
            "chunk load out of bounds"
        );
        if active == LANES {
            // SAFETY: the complete 16-byte range was bounds-checked. SSE2
            // unaligned loads accept any scalar slice alignment.
            Self(
                unsafe { _mm_loadu_si128(slice.as_ptr().add(offset).cast()) },
                PhantomData,
            )
        } else {
            let mut lanes = [T::default(); LANES];
            lanes[..active].copy_from_slice(&slice[offset..offset + active]);
            // SAFETY: LANES elements of T are initialized and their size is 16 bytes.
            Self(
                unsafe { _mm_loadu_si128(lanes.as_ptr().cast()) },
                PhantomData,
            )
        }
    }

    #[inline]
    fn store(self, slice: &mut [T], offset: usize, active: usize) {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed vector width");
        assert!(
            offset <= slice.len() && active <= slice.len() - offset,
            "chunk store out of bounds"
        );
        if active == LANES {
            // SAFETY: the complete 16-byte destination was bounds-checked;
            // the unaligned store imposes no alignment requirement.
            unsafe { _mm_storeu_si128(slice.as_mut_ptr().add(offset).cast(), self.0) };
        } else if active != 0 {
            let mut lanes = [T::default(); LANES];
            // SAFETY: the initialized stack array is exactly 16 bytes.
            unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), self.0) };
            slice[offset..offset + active].copy_from_slice(&lanes[..active]);
        }
    }

    #[inline]
    fn splat(_: Sse2, value: T) -> Self {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        Self(T::splat(value), PhantomData)
    }

    fn sum(self) -> T {
        self.active_sum(LANES)
    }
    fn prod(self) -> T {
        self.active_prod(LANES)
    }
    fn min(self) -> T {
        self.active_min(LANES)
    }
    fn max(self) -> T {
        self.active_max(LANES)
    }

    fn active_sum(self, active: usize) -> T {
        self.reduce(active, T::default(), Lane::add)
    }

    fn active_prod(self, active: usize) -> T {
        self.reduce_nonempty(active, T::one(), Lane::mul)
    }

    fn active_min(self, active: usize) -> T {
        self.reduce_nonempty(active, T::min_identity(), Lane::min)
    }

    fn active_max(self, active: usize) -> T {
        self.reduce_nonempty(active, T::max_identity(), Lane::max)
    }

    #[inline]
    fn eq(&self, rhs: &Self) -> Self::Mask {
        self.compare(rhs, CompareOp::Eq)
    }
    #[inline]
    fn ne(&self, rhs: &Self) -> Self::Mask {
        self.compare(rhs, CompareOp::Ne)
    }
    #[inline]
    fn lt(&self, rhs: &Self) -> Self::Mask {
        self.compare(rhs, CompareOp::Lt)
    }
    #[inline]
    fn le(&self, rhs: &Self) -> Self::Mask {
        self.compare(rhs, CompareOp::Le)
    }
    #[inline]
    fn gt(&self, rhs: &Self) -> Self::Mask {
        self.compare(rhs, CompareOp::Gt)
    }
    #[inline]
    fn ge(&self, rhs: &Self) -> Self::Mask {
        self.compare(rhs, CompareOp::Ge)
    }
}

impl<T: Lane, const LANES: usize> SseVector<T, LANES> {
    #[inline]
    fn compare(&self, rhs: &Self, relation: CompareOp) -> SseMask<T, LANES> {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        SseMask(T::compare(self.0, rhs.0, relation), PhantomData)
    }

    #[inline]
    fn reduce(self, active: usize, identity: T, combine: impl Fn(T, T) -> T) -> T {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed vector width");
        let mut values = [T::default(); LANES];
        // SAFETY: values is initialized and exactly 16 bytes.
        unsafe { _mm_storeu_si128(values.as_mut_ptr().cast(), self.0) };
        values[..active].iter().copied().fold(identity, combine)
    }

    #[inline]
    fn reduce_nonempty(self, active: usize, identity: T, combine: impl Fn(T, T) -> T) -> T {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed vector width");
        let mut values = [T::default(); LANES];
        // SAFETY: values is initialized and exactly 16 bytes.
        unsafe { _mm_storeu_si128(values.as_mut_ptr().cast(), self.0) };
        match values[..active].split_first() {
            Some((&first, rest)) => rest.iter().copied().fold(first, combine),
            None => identity,
        }
    }
}

macro_rules! integer_lane {
    ($ty:ty, $signed:ty, $lanes:expr, $splat:ident, $eq:ident, $gt:ident, $set_sign:ident, $sign:expr) => {
        // SAFETY: primitive integers have valid bit patterns; the lane count
        // and element size fill exactly one 128-bit register.
        unsafe impl Lane for $ty {
            const LANES: usize = $lanes;
            #[inline]
            fn splat(value: Self) -> __m128i {
                // SAFETY: this SSE2 splat intrinsic only requires the baseline feature.
                let signed = <$signed>::from_ne_bytes(value.to_ne_bytes());
                unsafe { $splat(signed) }
            }
            #[inline]
            fn compare(a: __m128i, b: __m128i, relation: CompareOp) -> __m128i {
                // SAFETY: every intrinsic below is part of SSE2.
                unsafe {
                    let sign = $set_sign($sign);
                    let (left, right) = if $sign == 0 {
                        (a, b)
                    } else {
                        (_mm_xor_si128(a, sign), _mm_xor_si128(b, sign))
                    };
                    let eq = $eq(left, right);
                    let greater = $gt(left, right);
                    let less = $gt(right, left);
                    match relation {
                        CompareOp::Eq => eq,
                        CompareOp::Ne => _mm_xor_si128(eq, _mm_set1_epi32(-1)),
                        CompareOp::Lt => less,
                        CompareOp::Le => _mm_xor_si128(greater, _mm_set1_epi32(-1)),
                        CompareOp::Gt => greater,
                        CompareOp::Ge => _mm_xor_si128(less, _mm_set1_epi32(-1)),
                    }
                }
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
                Ord::min(self, rhs)
            }
            #[inline]
            fn max(self, rhs: Self) -> Self {
                Ord::max(self, rhs)
            }
            #[inline]
            fn one() -> Self {
                1
            }
            #[inline]
            fn min_identity() -> Self {
                Self::MAX
            }
            #[inline]
            fn max_identity() -> Self {
                Self::MIN
            }
        }
    };
}

integer_lane!(
    i8,
    i8,
    16,
    _mm_set1_epi8,
    _mm_cmpeq_epi8,
    _mm_cmpgt_epi8,
    _mm_set1_epi8,
    0_i8
);
integer_lane!(
    u8,
    i8,
    16,
    _mm_set1_epi8,
    _mm_cmpeq_epi8,
    _mm_cmpgt_epi8,
    _mm_set1_epi8,
    i8::MIN
);
integer_lane!(
    i16,
    i16,
    8,
    _mm_set1_epi16,
    _mm_cmpeq_epi16,
    _mm_cmpgt_epi16,
    _mm_set1_epi16,
    0_i16
);
integer_lane!(
    u16,
    i16,
    8,
    _mm_set1_epi16,
    _mm_cmpeq_epi16,
    _mm_cmpgt_epi16,
    _mm_set1_epi16,
    i16::MIN
);
integer_lane!(
    i32,
    i32,
    4,
    _mm_set1_epi32,
    _mm_cmpeq_epi32,
    _mm_cmpgt_epi32,
    _mm_set1_epi32,
    0_i32
);
integer_lane!(
    u32,
    i32,
    4,
    _mm_set1_epi32,
    _mm_cmpeq_epi32,
    _mm_cmpgt_epi32,
    _mm_set1_epi32,
    i32::MIN
);

// SSE2 has no packed 64-bit integer comparisons. These comparisons inspect
// two lanes scalarly and materialize native all-zero/all-one predicate lanes.
// The arithmetic and memory operations for these vectors remain register based.
// SAFETY: i64/u64 have valid bit patterns and two lanes fill one register.
unsafe impl Lane for i64 {
    const LANES: usize = 2;
    fn splat(value: Self) -> __m128i {
        // SAFETY: the intrinsic is part of SSE2.
        unsafe { _mm_set1_epi64x(value) }
    }
    fn compare(a: __m128i, b: __m128i, relation: CompareOp) -> __m128i {
        let mut left = [0_i64; 2];
        let mut right = [0_i64; 2];
        // SAFETY: both initialized arrays occupy exactly 16 bytes.
        unsafe {
            _mm_storeu_si128(left.as_mut_ptr().cast(), a);
            _mm_storeu_si128(right.as_mut_ptr().cast(), b);
        }
        let result = array::from_fn::<_, 2, _>(|i| {
            let matches = match relation {
                CompareOp::Eq => left[i] == right[i],
                CompareOp::Ne => left[i] != right[i],
                CompareOp::Lt => left[i] < right[i],
                CompareOp::Le => left[i] <= right[i],
                CompareOp::Gt => left[i] > right[i],
                CompareOp::Ge => left[i] >= right[i],
            };
            if matches { u64::MAX } else { 0 }
        });
        // SAFETY: result is an initialized 16-byte array.
        unsafe { _mm_loadu_si128(result.as_ptr().cast()) }
    }
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
    fn min(self, rhs: Self) -> Self {
        Ord::min(self, rhs)
    }
    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }
    fn one() -> Self {
        1
    }
    fn min_identity() -> Self {
        Self::MAX
    }
    fn max_identity() -> Self {
        Self::MIN
    }
}

// SAFETY: u64 has valid bit patterns and two lanes fill one register.
unsafe impl Lane for u64 {
    const LANES: usize = 2;
    fn splat(value: Self) -> __m128i {
        // SAFETY: the intrinsic preserves the input's bit pattern.
        unsafe { _mm_set1_epi64x(value.cast_signed()) }
    }
    fn compare(a: __m128i, b: __m128i, relation: CompareOp) -> __m128i {
        let mut left = [0_u64; 2];
        let mut right = [0_u64; 2];
        // SAFETY: both initialized arrays occupy exactly 16 bytes.
        unsafe {
            _mm_storeu_si128(left.as_mut_ptr().cast(), a);
            _mm_storeu_si128(right.as_mut_ptr().cast(), b);
        }
        let result = array::from_fn::<_, 2, _>(|i| {
            let matches = match relation {
                CompareOp::Eq => left[i] == right[i],
                CompareOp::Ne => left[i] != right[i],
                CompareOp::Lt => left[i] < right[i],
                CompareOp::Le => left[i] <= right[i],
                CompareOp::Gt => left[i] > right[i],
                CompareOp::Ge => left[i] >= right[i],
            };
            if matches { Self::MAX } else { 0 }
        });
        // SAFETY: result is an initialized 16-byte array.
        unsafe { _mm_loadu_si128(result.as_ptr().cast()) }
    }
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
    fn min(self, rhs: Self) -> Self {
        Ord::min(self, rhs)
    }
    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }
    fn one() -> Self {
        1
    }
    fn min_identity() -> Self {
        Self::MAX
    }
    fn max_identity() -> Self {
        0
    }
}

macro_rules! float_lane {
    ($ty:ty, $lanes:expr, $splat:ident, $eq:ident, $ne:ident, $lt:ident, $le:ident, $gt:ident, $ge:ident, $cast_from:ident, $cast_to:ident) => {
        // SAFETY: f32/f64 have no invalid bit patterns and fill one register.
        unsafe impl Lane for $ty {
            const LANES: usize = $lanes;
            #[inline]
            fn splat(value: Self) -> __m128i {
                // SAFETY: the float splat and cast are SSE2 operations.
                unsafe { $cast_from($splat(value)) }
            }
            #[inline]
            fn compare(a: __m128i, b: __m128i, relation: CompareOp) -> __m128i {
                // SAFETY: each mapped comparison is an SSE2 intrinsic.
                unsafe {
                    let a = $cast_to(a);
                    let b = $cast_to(b);
                    match relation {
                        CompareOp::Eq => $cast_from($eq(a, b)),
                        CompareOp::Ne => $cast_from($ne(a, b)),
                        CompareOp::Lt => $cast_from($lt(a, b)),
                        CompareOp::Le => $cast_from($le(a, b)),
                        CompareOp::Gt => $cast_from($gt(a, b)),
                        CompareOp::Ge => $cast_from($ge(a, b)),
                    }
                }
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
            fn one() -> Self {
                1.0
            }
            #[inline]
            fn min_identity() -> Self {
                Self::INFINITY
            }
            #[inline]
            fn max_identity() -> Self {
                Self::NEG_INFINITY
            }
        }
    };
}

float_lane!(
    f32,
    4,
    _mm_set1_ps,
    _mm_cmpeq_ps,
    _mm_cmpneq_ps,
    _mm_cmplt_ps,
    _mm_cmple_ps,
    _mm_cmpgt_ps,
    _mm_cmpge_ps,
    _mm_castps_si128,
    _mm_castsi128_ps
);
float_lane!(
    f64,
    2,
    _mm_set1_pd,
    _mm_cmpeq_pd,
    _mm_cmpneq_pd,
    _mm_cmplt_pd,
    _mm_cmple_pd,
    _mm_cmpgt_pd,
    _mm_cmpge_pd,
    _mm_castpd_si128,
    _mm_castsi128_pd
);
