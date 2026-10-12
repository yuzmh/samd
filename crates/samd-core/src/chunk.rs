//! Active-lane chunk construction, memory access, and reductions.

use core::marker::PhantomData;

use crate::{Element, Simd, SimdMask, SimdVector};

/// The active portion of one backend vector operation.
///
/// A chunk never contains more lanes than the requested remaining element
/// count or the backend's physical lane count. A zero-length chunk represents
/// an empty remainder and does not advance a processing loop.
#[derive(Clone, Copy, Debug)]
pub struct Chunk<S, T> {
    len: usize,
    simd: S,
    _element: PhantomData<T>,
}

impl<S: Simd, T> Chunk<S, T> {
    #[inline]
    pub(crate) fn new(simd: S, remaining: usize, lanes: usize) -> Self {
        assert!(lanes > 0, "a SIMD backend must provide at least one lane");
        Self {
            len: remaining.min(lanes),
            simd,
            _element: PhantomData,
        }
    }

    /// Returns the number of active lanes in this chunk.
    ///
    /// # Returns
    ///
    /// A value no greater than the requested remainder or physical lane count.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when this chunk has no active lanes.
    ///
    /// # Returns
    ///
    /// `true` exactly when [`Self::len`] is zero.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<S: Simd, T: Element<S>> Chunk<S, T> {
    /// Broadcasts `value` to every physical lane of this chunk's vector type.
    ///
    /// # Parameters
    ///
    /// - `value`: scalar value copied into each physical lane.
    ///
    /// # Returns
    ///
    /// A vector with `value` in every lane. Chunk reductions still use only
    /// this chunk's active lanes.
    #[must_use]
    #[inline]
    pub fn splat(&self, value: T) -> T::Vector {
        <T::Vector as SimdVector<T, S>>::splat(self.simd, value)
    }

    /// Loads the active elements beginning at `offset`.
    ///
    /// # Panics
    ///
    /// Panics if the active range is outside `slice`.
    ///
    /// # Parameters
    ///
    /// - `slice`: source elements.
    /// - `offset`: first element to load.
    ///
    /// # Returns
    ///
    /// A vector containing the active elements. Inactive physical lanes are
    /// unspecified and must be ignored by chunk-aware operations.
    #[must_use]
    #[inline]
    pub fn load(&self, slice: &[T], offset: usize) -> T::Vector {
        <T::Vector as SimdVector<T, S>>::load(self.simd, slice, offset, self.len)
    }

    /// Stores only the active lanes of `value` beginning at `offset`.
    ///
    /// # Panics
    ///
    /// Panics if the active range is outside `slice`.
    ///
    /// # Parameters
    ///
    /// - `slice`: destination elements.
    /// - `offset`: first element to write.
    /// - `value`: vector whose active lanes are stored.
    #[inline]
    pub fn store(&self, slice: &mut [T], offset: usize, value: T::Vector) {
        <T::Vector as SimdVector<T, S>>::store(value, slice, offset, self.len);
    }

    /// Adds the active lanes of `value` and returns their sum.
    ///
    /// An empty chunk returns the additive identity. Floating-point reduction
    /// order and rounding follow the backend implementation.
    #[must_use]
    #[inline]
    pub fn sum(&self, value: T::Vector) -> T {
        <T::Vector as SimdVector<T, S>>::active_sum(value, self.len)
    }

    /// Multiplies the active lanes of `value` and returns their product.
    ///
    /// An empty chunk returns the multiplicative identity.
    #[must_use]
    #[inline]
    pub fn prod(&self, value: T::Vector) -> T {
        <T::Vector as SimdVector<T, S>>::active_prod(value, self.len)
    }

    /// Returns the minimum active lane, or the reduction identity if empty.
    #[must_use]
    #[inline]
    pub fn min(&self, value: T::Vector) -> T {
        <T::Vector as SimdVector<T, S>>::active_min(value, self.len)
    }

    /// Returns the maximum active lane, or the reduction identity if empty.
    #[must_use]
    #[inline]
    pub fn max(&self, value: T::Vector) -> T {
        <T::Vector as SimdVector<T, S>>::active_max(value, self.len)
    }

    /// Returns whether any active mask lane is true.
    #[must_use]
    #[inline]
    pub fn any(&self, mask: <T::Vector as SimdVector<T, S>>::Mask) -> bool {
        mask.active_any(self.len)
    }

    /// Returns whether every active mask lane is true.
    #[must_use]
    #[inline]
    pub fn all(&self, mask: <T::Vector as SimdVector<T, S>>::Mask) -> bool {
        mask.active_all(self.len)
    }

    /// Returns whether no active mask lane is true.
    #[must_use]
    #[inline]
    pub fn none(&self, mask: <T::Vector as SimdVector<T, S>>::Mask) -> bool {
        mask.active_none(self.len)
    }

    /// Counts the true active lanes in `mask`.
    #[must_use]
    #[inline]
    pub fn count_true(&self, mask: <T::Vector as SimdVector<T, S>>::Mask) -> usize {
        mask.active_count_true(self.len)
    }
}
