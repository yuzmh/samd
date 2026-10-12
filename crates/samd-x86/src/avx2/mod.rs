//! Native x86-64 AVX2 backend.
//!
//! All vector values use one native 256-bit register. The marker has no safe
//! constructor: callers must establish AVX2 support before creating one.

use core::arch::x86_64::{self, __m256i};
use core::marker::PhantomData;
use samd_core::{MaskSelect, Simd, SimdMask, SimdVector};

mod ops;

/// Marker for x86-64 AVX2 operations.
///
/// Construct this only after confirming that the current processor supports
/// AVX2, normally through `is_x86_feature_detected!("avx2")` at the dispatch
/// boundary. It intentionally has no `Default` or safe constructor.
#[derive(Clone, Copy, Debug)]
pub struct Avx2 {
    _private: (),
}

impl Avx2 {
    /// Creates the AVX2 marker after the caller has checked processor support.
    ///
    /// # Safety
    ///
    /// The current processor and operating system must support AVX2 for every
    /// operation performed while this marker or vectors created from it are used.
    #[must_use]
    pub const unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

impl Simd for Avx2 {}

/// Predicate bits returned by AVX2 comparisons.
#[derive(Clone, Copy)]
pub struct Avx2Mask<T, const LANES: usize> {
    bits: __m256i,
    _element: PhantomData<T>,
}

impl<T, const LANES: usize> SimdMask for Avx2Mask<T, LANES> {
    #[inline]
    fn any(self) -> bool {
        // SAFETY: AVX2Mask can only be produced by the AVX2 backend.
        unsafe { x86_64::_mm256_movemask_epi8(self.bits) != 0 }
    }

    #[inline]
    fn all(self) -> bool {
        self.count_true() == LANES
    }

    #[inline]
    fn count_true(self) -> usize {
        // Every true comparison lane is all ones, so each lane contributes
        // exactly `size_of::<T>() * 8` set bits.
        // SAFETY: AVX2Mask can only be produced by the AVX2 backend.
        let bits =
            u32::from_ne_bytes(unsafe { x86_64::_mm256_movemask_epi8(self.bits).to_ne_bytes() });
        bits.count_ones() as usize / (32 / LANES)
    }

    #[inline]
    fn active_any(self, active: usize) -> bool {
        assert!(active <= LANES, "active lanes exceed AVX2 mask width");
        self.active_count_true(active) != 0
    }

    #[inline]
    fn active_all(self, active: usize) -> bool {
        assert!(active <= LANES, "active lanes exceed AVX2 mask width");
        self.active_count_true(active) == active
    }

    #[inline]
    fn active_count_true(self, active: usize) -> usize {
        assert!(active <= LANES, "active lanes exceed AVX2 mask width");
        if active == 0 {
            return 0;
        }
        let bytes_per_lane = 32 / LANES;
        let bytes = active * bytes_per_lane;
        let byte_mask = if bytes == 32 {
            u32::MAX
        } else {
            (1_u32 << bytes) - 1
        };
        // SAFETY: AVX2Mask can only be produced by the AVX2 backend.
        let bits =
            u32::from_ne_bytes(unsafe { x86_64::_mm256_movemask_epi8(self.bits).to_ne_bytes() });
        (bits & byte_mask).count_ones() as usize / bytes_per_lane
    }
}

/// A 256-bit AVX2 vector whose lane type determines its interpretation.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Avx2Vector<T, const LANES: usize> {
    bits: __m256i,
    _element: PhantomData<T>,
}

macro_rules! vector_types {
    ($(($name:ident, $element:ty, $lanes:literal)),+ $(,)?) => {
        $(
            #[doc = concat!("Native AVX2 vector with ", stringify!($lanes), " ", stringify!($element), " lanes.")]
            pub type $name = Avx2Vector<$element, $lanes>;
        )+
    };
}

vector_types!(
    (I8x32, i8, 32),
    (U8x32, u8, 32),
    (I16x16, i16, 16),
    (U16x16, u16, 16),
    (I32x8, i32, 8),
    (U32x8, u32, 8),
    (I64x4, i64, 4),
    (U64x4, u64, 4),
    (F32x8, f32, 8),
    (F64x4, f64, 4),
);

/// Intrinsic and scalar-lane operations for a 256-bit element layout.
#[doc(hidden)]
/// # Safety
///
/// Implementations must describe a primitive lane type and provide an array
/// with exactly 32 bytes. Comparisons must return one all-zero or all-one mask
/// per lane, and all array conversions must preserve the lane bit patterns.
pub unsafe trait Lane: Copy + Default {
    /// Number of scalar lanes in the native register.
    const LANES: usize;

    /// Fixed-size lane storage used for safe scalar tails and reductions.
    type Array: AsRef<[Self]> + AsMut<[Self]> + Copy + Default;

    /// Returns the comparison mask for `op`.
    ///
    /// # Safety
    ///
    /// The current CPU and operating system must support AVX2.
    unsafe fn compare(op: CompareOp, lhs: __m256i, rhs: __m256i) -> __m256i;

    /// Extracts all initialized lanes from a native register.
    fn to_array(bits: __m256i) -> Self::Array;

    /// Packs lanes into the native representation.
    fn from_array(values: Self::Array) -> __m256i;

    /// Broadcasts `value` into a native register.
    ///
    /// # Safety
    ///
    /// The current CPU and operating system must support AVX2.
    unsafe fn splat(value: Self) -> __m256i;

    /// Adds scalar lanes with backend arithmetic semantics.
    #[must_use]
    fn add(self, rhs: Self) -> Self;

    /// Multiplies scalar lanes with backend arithmetic semantics.
    #[must_use]
    fn mul(self, rhs: Self) -> Self;

    /// Returns floating minimum-number or integer minimum semantics.
    #[must_use]
    fn min(self, rhs: Self) -> Self;

    /// Returns floating maximum-number or integer maximum semantics.
    #[must_use]
    fn max(self, rhs: Self) -> Self;

    /// Additive identity.
    fn zero() -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Empty minimum identity.
    fn min_identity() -> Self;

    /// Empty maximum identity.
    fn max_identity() -> Self;
}

/// Comparison predicate used by [`Lane`] implementations.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub enum CompareOp {
    /// Equal.
    Eq,
    /// Not equal, including unordered floating-point values.
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

impl<T: Lane, const LANES: usize> Avx2Vector<T, LANES> {
    fn array(self) -> T::Array {
        T::to_array(self.bits)
    }

    #[inline]
    fn reduce(self, active: usize, identity: T, combine: impl Fn(T, T) -> T) -> T {
        assert!(active <= LANES, "active lanes exceed AVX2 vector width");
        self.array().as_ref()[..active]
            .iter()
            .copied()
            .fold(identity, combine)
    }

    #[inline]
    fn reduce_nonempty(self, active: usize, identity: T, combine: impl Fn(T, T) -> T) -> T {
        assert!(active <= LANES, "active lanes exceed AVX2 vector width");
        let lanes = self.array();
        let values = &lanes.as_ref()[..active];
        match values.split_first() {
            Some((&first, rest)) => rest.iter().copied().fold(first, combine),
            None => identity,
        }
    }
}

// SAFETY: the vector stores exactly one initialized native register. Every
// public memory operation checks the active range and slice bounds first.
unsafe impl<T: Lane, const LANES: usize> SimdVector<T, Avx2> for Avx2Vector<T, LANES> {
    type Mask = Avx2Mask<T, LANES>;

    #[inline]
    fn load(simd: Avx2, slice: &[T], offset: usize, active: usize) -> Self {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed AVX2 vector width");
        assert!(
            offset <= slice.len() && active <= slice.len() - offset,
            "chunk load out of bounds"
        );
        let bits = if active == LANES {
            // SAFETY: `simd` can only be created after checking AVX2 support;
            // the validated active range covers one complete 256-bit vector.
            unsafe { x86_64::_mm256_loadu_si256(slice.as_ptr().add(offset).cast()) }
        } else {
            let mut lanes = T::Array::default();
            lanes.as_mut()[..active].copy_from_slice(&slice[offset..offset + active]);
            T::from_array(lanes)
        };
        let _ = simd;
        Self {
            bits,
            _element: PhantomData,
        }
    }

    #[inline]
    fn store(self, slice: &mut [T], offset: usize, active: usize) {
        assert_eq!(LANES, T::LANES, "vector type has an invalid lane width");
        assert!(active <= LANES, "active lanes exceed AVX2 vector width");
        assert!(
            offset <= slice.len() && active <= slice.len() - offset,
            "chunk store out of bounds"
        );
        if active == LANES {
            // SAFETY: the vector was obtained under an AVX2 marker, and the
            // validated range covers one complete register.
            unsafe {
                x86_64::_mm256_storeu_si256(slice.as_mut_ptr().add(offset).cast(), self.bits);
            }
        } else {
            let lanes = self.array();
            slice[offset..offset + active].copy_from_slice(&lanes.as_ref()[..active]);
        }
    }

    #[inline]
    fn splat(_simd: Avx2, value: T) -> Self {
        // SAFETY: constructing `Avx2` requires the dispatcher to verify AVX2.
        let bits = unsafe { T::splat(value) };
        Self {
            bits,
            _element: PhantomData,
        }
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
        self.reduce(active, T::zero(), Lane::add)
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
        // SAFETY: these vectors can only be created through an AVX2 marker.
        Avx2Mask {
            bits: unsafe { T::compare(CompareOp::Eq, self.bits, rhs.bits) },
            _element: PhantomData,
        }
    }

    #[inline]
    fn ne(&self, rhs: &Self) -> Self::Mask {
        // SAFETY: these vectors can only be created through an AVX2 marker.
        Avx2Mask {
            bits: unsafe { T::compare(CompareOp::Ne, self.bits, rhs.bits) },
            _element: PhantomData,
        }
    }

    #[inline]
    fn lt(&self, rhs: &Self) -> Self::Mask {
        // SAFETY: these vectors can only be created through an AVX2 marker.
        Avx2Mask {
            bits: unsafe { T::compare(CompareOp::Lt, self.bits, rhs.bits) },
            _element: PhantomData,
        }
    }

    #[inline]
    fn le(&self, rhs: &Self) -> Self::Mask {
        // SAFETY: these vectors can only be created through an AVX2 marker.
        Avx2Mask {
            bits: unsafe { T::compare(CompareOp::Le, self.bits, rhs.bits) },
            _element: PhantomData,
        }
    }

    #[inline]
    fn gt(&self, rhs: &Self) -> Self::Mask {
        // SAFETY: these vectors can only be created through an AVX2 marker.
        Avx2Mask {
            bits: unsafe { T::compare(CompareOp::Gt, self.bits, rhs.bits) },
            _element: PhantomData,
        }
    }

    #[inline]
    fn ge(&self, rhs: &Self) -> Self::Mask {
        // SAFETY: these vectors can only be created through an AVX2 marker.
        Avx2Mask {
            bits: unsafe { T::compare(CompareOp::Ge, self.bits, rhs.bits) },
            _element: PhantomData,
        }
    }
}

impl<T: Lane, const LANES: usize> MaskSelect<Avx2Mask<T, LANES>> for Avx2Vector<T, LANES> {
    #[inline]
    fn select(mask: Avx2Mask<T, LANES>, if_true: Self, if_false: Self) -> Self {
        // SAFETY: the mask and vectors are native AVX2 values created by this backend.
        let selected = unsafe {
            x86_64::_mm256_or_si256(
                x86_64::_mm256_and_si256(mask.bits, if_true.bits),
                x86_64::_mm256_andnot_si256(mask.bits, if_false.bits),
            )
        };
        Self {
            bits: selected,
            _element: PhantomData,
        }
    }
}
