//! Compile-time backend marker and chunk construction.

use crate::{Chunk, Element};

/// A zero-cost marker that selects a concrete SIMD implementation.
pub trait Simd: Copy {
    /// Builds a chunk for the remaining element count.
    ///
    /// The chunk has `min(remaining, lanes)` active lanes, or zero lanes when
    /// `remaining` is zero. Implementations expose their lane count through
    /// [`Element::lanes`].
    ///
    /// # Panics
    ///
    /// Panics if the element mapping reports zero physical lanes.
    ///
    /// # Parameters
    ///
    /// - `remaining`: number of elements left to process.
    ///
    /// # Returns
    ///
    /// A chunk containing at most the backend's physical lane count. A zero
    /// remainder produces an empty chunk.
    #[must_use]
    #[inline]
    fn chunk<T: Element<Self>>(self, remaining: usize) -> Chunk<Self, T>
    where
        Self: Sized,
    {
        Chunk::new(self, remaining, T::lanes(self))
    }
}
