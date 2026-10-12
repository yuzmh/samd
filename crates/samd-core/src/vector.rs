//! Backend-independent vector and mask capability contracts.

use crate::Simd;

/// Predicate values associated with a vector representation.
///
/// Full-mask methods inspect every physical lane. The `active_*` methods
/// inspect only the first `active` lanes; an empty active set has the usual
/// reduction identities (`any` is false, `all`/`none` are true, count is 0).
pub trait SimdMask: Sized {
    /// Returns whether at least one physical lane is true.
    fn any(self) -> bool;

    /// Returns whether every physical lane is true.
    fn all(self) -> bool;

    /// Returns whether no physical lane is true.
    fn none(self) -> bool {
        !self.any()
    }

    /// Counts the true physical lanes.
    fn count_true(self) -> usize;

    /// Returns whether any of the first `active` lanes is true.
    ///
    /// # Panics
    ///
    /// Panics if `active` exceeds the mask's physical lane count.
    fn active_any(self, active: usize) -> bool;

    /// Returns whether all of the first `active` lanes are true.
    ///
    /// # Panics
    ///
    /// Panics if `active` exceeds the mask's physical lane count.
    fn active_all(self, active: usize) -> bool;

    /// Counts the true values among the first `active` lanes.
    ///
    /// # Panics
    ///
    /// Panics if `active` exceeds the mask's physical lane count.
    fn active_count_true(self, active: usize) -> usize;

    /// Selects `if_true` where this mask is true and `if_false` elsewhere.
    fn select<V: MaskSelect<Self>>(self, if_true: V, if_false: V) -> V {
        V::select(self, if_true, if_false)
    }

    /// Returns whether no active lane is true.
    #[inline]
    fn active_none(self, active: usize) -> bool {
        !self.active_any(active)
    }
}

/// Associates a mask representation with the vector it selects.
pub trait MaskSelect<M: SimdMask>: Sized {
    /// Selects lanes from two vectors according to `mask`.
    fn select(mask: M, if_true: Self, if_false: Self) -> Self;
}

/// Backend vector representation for scalar element `T`.
///
/// Implementations of the memory methods must check that `offset..offset +
/// active` is within the slice before accessing memory. They must access no
/// other elements, including when a vector's physical width exceeds `active`.
/// Integer arithmetic and reductions wrap modulo the lane width. Floating
/// reductions may use backend-dependent trees and therefore can differ in the
/// final rounded bits. Floating comparisons follow IEEE ordered comparisons:
/// `ne` is true for NaN operands and the other comparisons are false.
/// Floating min/max use minimum-number semantics: one NaN is ignored, all-NaN
/// reductions return NaN, and minimum/maximum distinguish signed zero.
///
/// The backend marker `S` is required to construct vectors through [`Self::load`]
/// and [`Self::splat`]. Backends that need optional CPU features must make
/// their marker impossible to construct safely until dispatch verifies those
/// features.
///
/// # Safety
///
/// `load` and `store` must always remain within the provided slice, create no
/// invalid scalar values, and honor exactly the active lane count. Native
/// implementations must uphold their representation invariants for every safe
/// call to this trait.
pub unsafe trait SimdVector<T: Copy, S: Simd> {
    /// The backend-specific predicate representation for this vector.
    type Mask: SimdMask;

    /// Loads `active` lanes starting at `offset`.
    ///
    /// # Panics
    ///
    /// Panics if the active range is out of bounds or exceeds the vector width.
    fn load(simd: S, slice: &[T], offset: usize, active: usize) -> Self;

    /// Stores `active` lanes starting at `offset`.
    ///
    /// # Panics
    ///
    /// Panics if the active range is out of bounds or exceeds the vector width.
    fn store(self, slice: &mut [T], offset: usize, active: usize);

    /// Replicates `value` across all physical lanes.
    fn splat(simd: S, value: T) -> Self;

    /// Reduce all physical lanes. These are distinct from the chunk-aware
    /// reductions below: a vector value does not remember its originating
    /// chunk's active length.
    #[must_use]
    fn sum(self) -> T;

    /// Multiplies all physical lanes together.
    #[must_use]
    fn prod(self) -> T;

    /// Returns the minimum across all physical lanes.
    #[must_use]
    fn min(self) -> T;

    /// Returns the maximum across all physical lanes.
    #[must_use]
    fn max(self) -> T;

    /// Reduce only the first `active` logical lanes.
    ///
    /// # Panics
    ///
    /// Panics if `active` exceeds the vector's physical lane count.
    #[must_use]
    fn active_sum(self, active: usize) -> T;

    /// Multiplies only the first `active` logical lanes.
    ///
    /// # Panics
    ///
    /// Panics if `active` exceeds the vector's physical lane count.
    #[must_use]
    fn active_prod(self, active: usize) -> T;

    /// Returns the minimum of the first `active` logical lanes.
    ///
    /// # Panics
    ///
    /// Panics if `active` exceeds the vector's physical lane count.
    #[must_use]
    fn active_min(self, active: usize) -> T;

    /// Returns the maximum of the first `active` logical lanes.
    ///
    /// # Panics
    ///
    /// Panics if `active` exceeds the vector's physical lane count.
    #[must_use]
    fn active_max(self, active: usize) -> T;

    /// Compares corresponding lanes for equality.
    fn eq(&self, rhs: &Self) -> Self::Mask;

    /// Compares corresponding lanes for inequality.
    fn ne(&self, rhs: &Self) -> Self::Mask;

    /// Compares corresponding lanes using the less-than relation.
    fn lt(&self, rhs: &Self) -> Self::Mask;

    /// Compares corresponding lanes using the less-than-or-equal relation.
    fn le(&self, rhs: &Self) -> Self::Mask;

    /// Compares corresponding lanes using the greater-than relation.
    fn gt(&self, rhs: &Self) -> Self::Mask;

    /// Compares corresponding lanes using the greater-than-or-equal relation.
    fn ge(&self, rhs: &Self) -> Self::Mask;
}

/// Floating-point vector operations that are meaningful for `f32` and `f64`.
/// Generic kernels should use this capability bound when invoking `abs` or
/// `sqrt`; integer vectors do not implement it.
pub trait SimdAbs: Sized {
    /// Returns the absolute value of each lane.
    #[must_use]
    fn abs(self) -> Self;
}

/// Square root capability, implemented only by floating-point vectors.
pub trait SimdSqrt: Sized {
    /// Returns the square root of each lane.
    #[must_use]
    fn sqrt(self) -> Self;
}
