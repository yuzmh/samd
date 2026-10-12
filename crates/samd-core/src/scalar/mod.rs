//! One-lane scalar vector representations and their safe backend contract.

use core::cmp::Ordering;

use crate::{MaskSelect, Simd, SimdMask, SimdVector};

mod ops;

/// Scalar backend marker. Each chunk processes one logical element.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scalar;

impl Simd for Scalar {}

/// One-lane scalar vector containing an `f32` value.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct F32(pub f32);
/// One-lane scalar vector containing an `f64` value.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct F64(pub f64);
/// One-lane scalar vector containing an `i8` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct I8(pub i8);
/// One-lane scalar vector containing a `u8` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct U8(pub u8);
/// One-lane scalar vector containing an `i16` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct I16(pub i16);
/// One-lane scalar vector containing a `u16` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct U16(pub u16);
/// One-lane scalar vector containing an `i32` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct I32(pub i32);
/// One-lane scalar vector containing a `u32` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct U32(pub u32);
/// One-lane scalar vector containing an `i64` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct I64(pub i64);
/// One-lane scalar vector containing a `u64` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct U64(pub u64);

/// One-lane predicate returned by scalar vector comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarMask(pub bool);

impl SimdMask for ScalarMask {
    #[inline]
    fn any(self) -> bool {
        self.0
    }
    #[inline]
    fn all(self) -> bool {
        self.0
    }
    #[inline]
    fn count_true(self) -> usize {
        usize::from(self.0)
    }
    #[inline]
    fn active_any(self, active: usize) -> bool {
        assert!(active <= 1, "active lanes exceed scalar mask width");
        active != 0 && self.0
    }
    #[inline]
    fn active_all(self, active: usize) -> bool {
        assert!(active <= 1, "active lanes exceed scalar mask width");
        active == 0 || self.0
    }
    #[inline]
    fn active_count_true(self, active: usize) -> usize {
        assert!(active <= 1, "active lanes exceed scalar mask width");
        usize::from(active != 0 && self.0)
    }
}

macro_rules! impl_scalar_vector {
    ($vec:ident, $t:ty, $zero:expr, $one:expr, $min_identity:expr, $max_identity:expr) => {
        impl $vec {
            /// Compares this scalar vector with `rhs`, returning a one-lane mask.
            #[must_use]
            #[inline]
            pub fn eq(&self, rhs: &Self) -> ScalarMask {
                <Self as SimdVector<$t, Scalar>>::eq(self, rhs)
            }
        }

        // SAFETY: the wrapper contains one valid scalar value, and each memory
        // method checks both its active width and slice bounds before access.
        unsafe impl SimdVector<$t, Scalar> for $vec {
            type Mask = ScalarMask;
            #[inline]
            fn load(_: Scalar, slice: &[$t], offset: usize, active: usize) -> Self {
                assert!(active <= 1, "active lanes exceed scalar vector width");
                assert!(
                    offset <= slice.len() && active <= slice.len() - offset,
                    "chunk load out of bounds"
                );
                if active == 0 {
                    Self($zero)
                } else {
                    Self(slice[offset])
                }
            }
            #[inline]
            fn store(self, slice: &mut [$t], offset: usize, active: usize) {
                assert!(active <= 1, "active lanes exceed scalar vector width");
                assert!(
                    offset <= slice.len() && active <= slice.len() - offset,
                    "chunk store out of bounds"
                );
                if active != 0 {
                    slice[offset] = self.0;
                }
            }
            #[inline]
            fn splat(_: Scalar, value: $t) -> Self {
                Self(value)
            }
            #[inline]
            fn sum(self) -> $t {
                self.0
            }
            #[inline]
            fn prod(self) -> $t {
                self.0
            }
            #[inline]
            fn min(self) -> $t {
                self.0
            }
            #[inline]
            fn max(self) -> $t {
                self.0
            }
            #[inline]
            fn active_sum(self, active: usize) -> $t {
                assert!(active <= 1);
                if active == 0 { $zero } else { self.0 }
            }
            #[inline]
            fn active_prod(self, active: usize) -> $t {
                assert!(active <= 1);
                if active == 0 { $one } else { self.0 }
            }
            #[inline]
            fn active_min(self, active: usize) -> $t {
                assert!(active <= 1);
                if active == 0 { $min_identity } else { self.0 }
            }
            #[inline]
            fn active_max(self, active: usize) -> $t {
                assert!(active <= 1);
                if active == 0 { $max_identity } else { self.0 }
            }
            #[inline]
            fn eq(&self, rhs: &Self) -> Self::Mask {
                ScalarMask(self.0.partial_cmp(&rhs.0) == Some(Ordering::Equal))
            }
            #[inline]
            fn ne(&self, rhs: &Self) -> Self::Mask {
                ScalarMask(self.0.partial_cmp(&rhs.0) != Some(Ordering::Equal))
            }
            #[inline]
            fn lt(&self, rhs: &Self) -> Self::Mask {
                ScalarMask(self.0.partial_cmp(&rhs.0) == Some(Ordering::Less))
            }
            #[inline]
            fn le(&self, rhs: &Self) -> Self::Mask {
                ScalarMask(matches!(
                    self.0.partial_cmp(&rhs.0),
                    Some(Ordering::Less | Ordering::Equal)
                ))
            }
            #[inline]
            fn gt(&self, rhs: &Self) -> Self::Mask {
                ScalarMask(self.0.partial_cmp(&rhs.0) == Some(Ordering::Greater))
            }
            #[inline]
            fn ge(&self, rhs: &Self) -> Self::Mask {
                ScalarMask(matches!(
                    self.0.partial_cmp(&rhs.0),
                    Some(Ordering::Greater | Ordering::Equal)
                ))
            }
        }
        impl MaskSelect<ScalarMask> for $vec {
            #[inline]
            fn select(mask: ScalarMask, if_true: Self, if_false: Self) -> Self {
                if mask.0 { if_true } else { if_false }
            }
        }
    };
}

impl_scalar_vector!(F32, f32, 0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY);
impl_scalar_vector!(F64, f64, 0.0, 1.0, f64::INFINITY, f64::NEG_INFINITY);
impl_scalar_vector!(I8, i8, 0, 1, i8::MAX, i8::MIN);
impl_scalar_vector!(U8, u8, 0, 1, u8::MAX, u8::MIN);
impl_scalar_vector!(I16, i16, 0, 1, i16::MAX, i16::MIN);
impl_scalar_vector!(U16, u16, 0, 1, u16::MAX, u16::MIN);
impl_scalar_vector!(I32, i32, 0, 1, i32::MAX, i32::MIN);
impl_scalar_vector!(U32, u32, 0, 1, u32::MAX, u32::MIN);
impl_scalar_vector!(I64, i64, 0, 1, i64::MAX, i64::MIN);
impl_scalar_vector!(U64, u64, 0, 1, u64::MAX, u64::MIN);
