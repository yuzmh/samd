//! Scalar-element to backend-vector type mappings.

use core::ops::{Add, Sub};

use crate::{Simd, SimdVector};

/// Maps a scalar element type to its vector representation for backend `S`.
pub trait Element<S: Simd>: Copy {
    /// The vector type associated with this element and backend.
    ///
    /// Every mapped vector supports addition, subtraction, safe chunk memory
    /// access, reductions, and comparisons through [`SimdVector`]. Additional
    /// operations are exposed only when that backend implements them.
    type Vector: SimdVector<Self, S>
        + Add<Output = Self::Vector>
        + Sub<Output = Self::Vector>
        + Copy;

    /// Returns the physical lane count for this element and backend.
    ///
    /// Scalable backends may return a runtime vector length. The result must be
    /// greater than zero because a non-empty chunk must always make progress.
    ///
    /// # Parameters
    ///
    /// - `simd`: the zero-sized backend marker to query.
    ///
    /// # Returns
    ///
    /// The number of elements represented by one full physical vector.
    #[must_use]
    fn lanes(simd: S) -> usize;
}
