//! Native WebAssembly SIMD128 backend.
//!
//! Every public vector is a transparent wrapper over one native `v128`. Partial
//! memory operations are scalarized into a small stack array so they never touch
//! inactive slice elements.

use core::arch::wasm32::{self, v128};
use core::marker::PhantomData;
use samd_core::{MaskSelect, Simd, SimdMask, SimdVector};

/// Marker for WebAssembly SIMD128 operations.
///
/// This type is available only for `wasm32` builds compiled with
/// `-C target-feature=+simd128`.
#[derive(Clone, Copy, Debug, Default)]
pub struct WasmSimd128;

impl Simd for WasmSimd128 {}

/// Scalar lane operations used only by the shared reduction and safe tail paths.
/// Vector arithmetic and comparisons use native SIMD instructions.
///
/// # Safety
///
/// Implementors must be a primitive numeric scalar with no invalid bit patterns,
/// `size_of::<Self>() == WIDTH`, and `LANES * WIDTH == 16`. Every splat and
/// comparison must return a valid native SIMD value with all-ones true masks.
#[doc(hidden)]
pub unsafe trait Lane: Copy + Default {
    /// Number of scalar lanes in a `v128` register.
    const LANES: usize;

    /// Byte width of one scalar lane.
    const WIDTH: usize;

    /// Broadcasts one lane into a native vector.
    fn splat(value: Self) -> v128;

    /// Compares two native vectors, producing all-ones true lanes.
    fn cmp(op: CompareOp, a: v128, b: v128) -> v128;

    /// Adds two scalar lanes using the backend's arithmetic semantics.
    #[must_use]
    fn add(self, rhs: Self) -> Self;

    /// Multiplies two scalar lanes using the backend's arithmetic semantics.
    #[must_use]
    fn mul(self, rhs: Self) -> Self;

    /// Returns the minimum of two scalar lanes.
    #[must_use]
    fn min(self, rhs: Self) -> Self;

    /// Returns the maximum of two scalar lanes.
    #[must_use]
    fn max(self, rhs: Self) -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Identity used for the minimum of an empty lane range.
    fn min_identity() -> Self;

    /// Identity used for the maximum of an empty lane range.
    fn max_identity() -> Self;
}

/// Comparison relation used by native lane comparison implementations.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub enum CompareOp {
    /// Equal values.
    Eq,
    /// Unequal values, including unordered floating-point comparisons.
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

/// Native vector wrapper with a `v128` register and a zero-sized lane marker.
///
/// `LANES` must match the physical lane count for `T`; the provided aliases encode
/// the supported combinations. Public operations panic on a mismatched width.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct WasmVector<T: Lane, const LANES: usize>(v128, PhantomData<T>);

/// Native comparison predicate. `T` distinguishes logical lane widths.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct WasmMask<T: Lane, const LANES: usize>(v128, PhantomData<T>);

/// Sixteen signed 8-bit integer lanes in one `v128` register.
pub type I8x16 = WasmVector<i8, 16>;
/// Sixteen unsigned 8-bit integer lanes in one `v128` register.
pub type U8x16 = WasmVector<u8, 16>;
/// Eight signed 16-bit integer lanes in one `v128` register.
pub type I16x8 = WasmVector<i16, 8>;
/// Eight unsigned 16-bit integer lanes in one `v128` register.
pub type U16x8 = WasmVector<u16, 8>;
/// Four signed 32-bit integer lanes in one `v128` register.
pub type I32x4 = WasmVector<i32, 4>;
/// Four unsigned 32-bit integer lanes in one `v128` register.
pub type U32x4 = WasmVector<u32, 4>;
/// Two signed 64-bit integer lanes in one `v128` register.
pub type I64x2 = WasmVector<i64, 2>;
/// Two unsigned 64-bit integer lanes in one `v128` register.
pub type U64x2 = WasmVector<u64, 2>;
/// Four 32-bit floating-point lanes in one `v128` register.
pub type F32x4 = WasmVector<f32, 4>;
/// Two 64-bit floating-point lanes in one `v128` register.
pub type F64x2 = WasmVector<f64, 2>;

/// Predicate for sixteen signed 8-bit lanes.
pub type MaskI8x16 = WasmMask<i8, 16>;
/// Predicate for sixteen unsigned 8-bit lanes.
pub type MaskU8x16 = WasmMask<u8, 16>;
/// Predicate for eight signed 16-bit lanes.
pub type MaskI16x8 = WasmMask<i16, 8>;
/// Predicate for eight unsigned 16-bit lanes.
pub type MaskU16x8 = WasmMask<u16, 8>;
/// Predicate for four signed 32-bit lanes.
pub type MaskI32x4 = WasmMask<i32, 4>;
/// Predicate for four unsigned 32-bit lanes.
pub type MaskU32x4 = WasmMask<u32, 4>;
/// Predicate for two signed 64-bit lanes.
pub type MaskI64x2 = WasmMask<i64, 2>;
/// Predicate for two unsigned 64-bit lanes.
pub type MaskU64x2 = WasmMask<u64, 2>;
/// Predicate for four 32-bit floating-point lanes.
pub type MaskF32x4 = WasmMask<f32, 4>;
/// Predicate for two 64-bit floating-point lanes.
pub type MaskF64x2 = WasmMask<f64, 2>;

impl<T: Lane, const LANES: usize> SimdMask for WasmMask<T, LANES> {
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
        mask_bits::<T, LANES>(self.0) & active_lane_bits(active) != 0
    }
    #[inline]
    fn active_all(self, active: usize) -> bool {
        assert_eq!(LANES, T::LANES, "mask type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed mask width");
        let active_bits = active_lane_bits(active);
        mask_bits::<T, LANES>(self.0) & active_bits == active_bits
    }
    #[inline]
    fn active_count_true(self, active: usize) -> usize {
        assert_eq!(LANES, T::LANES, "mask type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed mask width");
        (mask_bits::<T, LANES>(self.0) & active_lane_bits(active)).count_ones() as usize
    }
}

impl<T: Lane, const LANES: usize> MaskSelect<WasmMask<T, LANES>> for WasmVector<T, LANES> {
    #[inline]
    fn select(mask: WasmMask<T, LANES>, if_true: Self, if_false: Self) -> Self {
        Self(
            wasm32::v128_bitselect(if_true.0, if_false.0, mask.0),
            PhantomData,
        )
    }
}

fn mask_bits<T: Lane, const LANES: usize>(mask: v128) -> u16 {
    assert_eq!(LANES, T::LANES, "mask type has an invalid lane width");
    let bits = wasm32::i8x16_bitmask(mask) as u16;
    (0..LANES).fold(0, |lanes, lane| {
        lanes | (((bits >> (lane * T::WIDTH)) & 1) << lane)
    })
}

#[inline]
const fn active_lane_bits(active: usize) -> u16 {
    if active == 16 {
        u16::MAX
    } else {
        (1_u16 << active) - 1
    }
}

// SAFETY: every memory method validates the active range and the lane-count
// invariant before using v128 loads/stores. Lane's unsafe contract guarantees
// each supported primitive lane layout is exactly 128 bits per register.
unsafe impl<T: Lane, const LANES: usize> SimdVector<T, WasmSimd128> for WasmVector<T, LANES> {
    type Mask = WasmMask<T, LANES>;

    #[inline]
    fn load(_: WasmSimd128, slice: &[T], offset: usize, active: usize) -> Self {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed vector width");
        assert!(
            offset <= slice.len() && active <= slice.len() - offset,
            "chunk load out of bounds"
        );
        if active == LANES {
            // SAFETY: range is checked above, and wasm SIMD loads allow unaligned addresses.
            Self(
                unsafe { wasm32::v128_load(slice.as_ptr().add(offset).cast()) },
                PhantomData,
            )
        } else {
            let mut lanes = [T::default(); LANES];
            lanes[..active].copy_from_slice(&slice[offset..offset + active]);
            // SAFETY: the stack array contains 16 initialized bytes.
            Self(
                unsafe { wasm32::v128_load(lanes.as_ptr().cast()) },
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
            // SAFETY: range is checked above, and wasm SIMD stores allow unaligned addresses.
            unsafe { wasm32::v128_store(slice.as_mut_ptr().add(offset).cast(), self.0) }
        } else if active != 0 {
            let mut lanes = [T::default(); LANES];
            // SAFETY: the stack array is a valid initialized 16-byte destination.
            unsafe { wasm32::v128_store(lanes.as_mut_ptr().cast(), self.0) };
            slice[offset..offset + active].copy_from_slice(&lanes[..active]);
        }
    }

    #[inline]
    fn splat(_: WasmSimd128, value: T) -> Self {
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
    fn eq(&self, r: &Self) -> Self::Mask {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        WasmMask(T::cmp(CompareOp::Eq, self.0, r.0), PhantomData)
    }
    #[inline]
    fn ne(&self, r: &Self) -> Self::Mask {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        WasmMask(T::cmp(CompareOp::Ne, self.0, r.0), PhantomData)
    }
    #[inline]
    fn lt(&self, r: &Self) -> Self::Mask {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        WasmMask(T::cmp(CompareOp::Lt, self.0, r.0), PhantomData)
    }
    #[inline]
    fn le(&self, r: &Self) -> Self::Mask {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        WasmMask(T::cmp(CompareOp::Le, self.0, r.0), PhantomData)
    }
    #[inline]
    fn gt(&self, r: &Self) -> Self::Mask {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        WasmMask(T::cmp(CompareOp::Gt, self.0, r.0), PhantomData)
    }
    #[inline]
    fn ge(&self, r: &Self) -> Self::Mask {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        WasmMask(T::cmp(CompareOp::Ge, self.0, r.0), PhantomData)
    }
}

impl<T: Lane, const LANES: usize> WasmVector<T, LANES> {
    #[inline]
    fn reduce(self, active: usize, identity: T, combine: impl Fn(T, T) -> T) -> T {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed vector width");
        self.to_array()[..active]
            .iter()
            .copied()
            .fold(identity, combine)
    }

    #[inline]
    fn reduce_nonempty(self, active: usize, identity: T, combine: impl Fn(T, T) -> T) -> T {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed vector width");
        let lanes = self.to_array();
        let active_lanes = &lanes[..active];
        match active_lanes.split_first() {
            Some((&first, rest)) => rest.iter().copied().fold(first, combine),
            None => identity,
        }
    }

    #[inline]
    fn to_array(self) -> [T; LANES] {
        let mut lanes = [T::default(); LANES];
        // SAFETY: the array is exactly 16 bytes and writable.
        unsafe { wasm32::v128_store(lanes.as_mut_ptr().cast(), self.0) };
        lanes
    }
}

mod ops;

macro_rules! integer_lane {
    ($t:ty, $width:expr, $lanes:expr, $splat:path,
     $eq:path, $ne:path, $lt:path, $le:path, $gt:path, $ge:path, $min:path, $max:path) => {
        // SAFETY: these primitive lanes have valid bit patterns, and the
        // supplied width and lane constants fill exactly one v128 register.
        unsafe impl Lane for $t {
            const LANES: usize = $lanes;
            const WIDTH: usize = $width;
            #[inline]
            fn splat(v: Self) -> v128 {
                $splat(v as _)
            }
            #[inline]
            fn cmp(op: CompareOp, a: v128, b: v128) -> v128 {
                match op {
                    CompareOp::Eq => $eq(a, b),
                    CompareOp::Ne => $ne(a, b),
                    CompareOp::Lt => $lt(a, b),
                    CompareOp::Le => $le(a, b),
                    CompareOp::Gt => $gt(a, b),
                    CompareOp::Ge => $ge(a, b),
                }
            }
            #[inline]
            fn add(self, r: Self) -> Self {
                self.wrapping_add(r)
            }
            #[inline]
            fn mul(self, r: Self) -> Self {
                self.wrapping_mul(r)
            }
            #[inline]
            fn min(self, r: Self) -> Self {
                Ord::min(self, r)
            }
            #[inline]
            fn max(self, r: Self) -> Self {
                Ord::max(self, r)
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

macro_rules! float_lane {
    ($t:ty, $width:expr, $lanes:expr, $splat:path,
     $eq:path, $ne:path, $lt:path, $le:path, $gt:path, $ge:path) => {
        // SAFETY: these primitive floats have valid bit patterns, and the
        // supplied width and lane constants fill exactly one v128 register.
        unsafe impl Lane for $t {
            const LANES: usize = $lanes;
            const WIDTH: usize = $width;
            #[inline]
            fn splat(v: Self) -> v128 {
                $splat(v)
            }
            #[inline]
            fn cmp(op: CompareOp, a: v128, b: v128) -> v128 {
                match op {
                    CompareOp::Eq => $eq(a, b),
                    CompareOp::Ne => $ne(a, b),
                    CompareOp::Lt => $lt(a, b),
                    CompareOp::Le => $le(a, b),
                    CompareOp::Gt => $gt(a, b),
                    CompareOp::Ge => $ge(a, b),
                }
            }
            #[inline]
            fn add(self, r: Self) -> Self {
                self + r
            }
            #[inline]
            fn mul(self, r: Self) -> Self {
                self * r
            }
            #[inline]
            fn min(self, r: Self) -> Self {
                if self == 0.0 && r == 0.0 && (self.is_sign_negative() || r.is_sign_negative()) {
                    -0.0
                } else {
                    self.min(r)
                }
            }
            #[inline]
            fn max(self, r: Self) -> Self {
                if self == 0.0 && r == 0.0 && !(self.is_sign_negative() && r.is_sign_negative()) {
                    0.0
                } else {
                    self.max(r)
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

integer_lane!(
    i8,
    1,
    16,
    wasm32::i8x16_splat,
    wasm32::i8x16_eq,
    wasm32::i8x16_ne,
    wasm32::i8x16_lt,
    wasm32::i8x16_le,
    wasm32::i8x16_gt,
    wasm32::i8x16_ge,
    wasm32::i8x16_min,
    wasm32::i8x16_max
);
integer_lane!(
    u8,
    1,
    16,
    wasm32::u8x16_splat,
    wasm32::i8x16_eq,
    wasm32::i8x16_ne,
    wasm32::u8x16_lt,
    wasm32::u8x16_le,
    wasm32::u8x16_gt,
    wasm32::u8x16_ge,
    wasm32::u8x16_min,
    wasm32::u8x16_max
);
integer_lane!(
    i16,
    2,
    8,
    wasm32::i16x8_splat,
    wasm32::i16x8_eq,
    wasm32::i16x8_ne,
    wasm32::i16x8_lt,
    wasm32::i16x8_le,
    wasm32::i16x8_gt,
    wasm32::i16x8_ge,
    wasm32::i16x8_min,
    wasm32::i16x8_max
);
integer_lane!(
    u16,
    2,
    8,
    wasm32::u16x8_splat,
    wasm32::i16x8_eq,
    wasm32::i16x8_ne,
    wasm32::u16x8_lt,
    wasm32::u16x8_le,
    wasm32::u16x8_gt,
    wasm32::u16x8_ge,
    wasm32::u16x8_min,
    wasm32::u16x8_max
);
integer_lane!(
    i32,
    4,
    4,
    wasm32::i32x4_splat,
    wasm32::i32x4_eq,
    wasm32::i32x4_ne,
    wasm32::i32x4_lt,
    wasm32::i32x4_le,
    wasm32::i32x4_gt,
    wasm32::i32x4_ge,
    wasm32::i32x4_min,
    wasm32::i32x4_max
);
integer_lane!(
    u32,
    4,
    4,
    wasm32::u32x4_splat,
    wasm32::i32x4_eq,
    wasm32::i32x4_ne,
    wasm32::u32x4_lt,
    wasm32::u32x4_le,
    wasm32::u32x4_gt,
    wasm32::u32x4_ge,
    wasm32::u32x4_min,
    wasm32::u32x4_max
);
float_lane!(
    f32,
    4,
    4,
    wasm32::f32x4_splat,
    wasm32::f32x4_eq,
    wasm32::f32x4_ne,
    wasm32::f32x4_lt,
    wasm32::f32x4_le,
    wasm32::f32x4_gt,
    wasm32::f32x4_ge
);
float_lane!(
    f64,
    8,
    2,
    wasm32::f64x2_splat,
    wasm32::f64x2_eq,
    wasm32::f64x2_ne,
    wasm32::f64x2_lt,
    wasm32::f64x2_le,
    wasm32::f64x2_gt,
    wasm32::f64x2_ge
);

macro_rules! wide_cmp {
    ($t:ty, $n:expr, $eq:path, $splat:path, $high_lt:path) => {
        // SAFETY: these primitive 64-bit lanes have valid bit patterns and the
        // two supplied lanes exactly fill one v128 register.
        unsafe impl Lane for $t {
            const LANES: usize = $n;
            const WIDTH: usize = 8;
            #[inline]
            fn splat(v: Self) -> v128 {
                $splat(v as _)
            }
            fn cmp(op: CompareOp, a: v128, b: v128) -> v128 {
                let eq = $eq(a, b);
                let high_a = wasm32::i32x4_shuffle::<1, 1, 3, 3>(a, a);
                let high_b = wasm32::i32x4_shuffle::<1, 1, 3, 3>(b, b);
                let low_a = wasm32::i32x4_shuffle::<0, 0, 2, 2>(a, a);
                let low_b = wasm32::i32x4_shuffle::<0, 0, 2, 2>(b, b);
                let high_eq = wasm32::i32x4_eq(high_a, high_b);
                let less = wasm32::v128_or(
                    $high_lt(high_a, high_b),
                    wasm32::v128_and(high_eq, wasm32::u32x4_lt(low_a, low_b)),
                );
                let greater = wasm32::v128_or(
                    $high_lt(high_b, high_a),
                    wasm32::v128_and(high_eq, wasm32::u32x4_lt(low_b, low_a)),
                );
                match op {
                    CompareOp::Eq => eq,
                    CompareOp::Ne => wasm32::v128_not(eq),
                    CompareOp::Lt => less,
                    CompareOp::Le => wasm32::v128_or(less, eq),
                    CompareOp::Gt => greater,
                    CompareOp::Ge => wasm32::v128_or(greater, eq),
                }
            }
            #[inline]
            fn add(self, r: Self) -> Self {
                self.wrapping_add(r)
            }
            #[inline]
            fn mul(self, r: Self) -> Self {
                self.wrapping_mul(r)
            }
            #[inline]
            fn min(self, r: Self) -> Self {
                Ord::min(self, r)
            }
            #[inline]
            fn max(self, r: Self) -> Self {
                Ord::max(self, r)
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
wide_cmp!(
    i64,
    2,
    wasm32::i64x2_eq,
    wasm32::i64x2_splat,
    wasm32::i32x4_lt
);
wide_cmp!(
    u64,
    2,
    wasm32::i64x2_eq,
    wasm32::u64x2_splat,
    wasm32::u32x4_lt
);
