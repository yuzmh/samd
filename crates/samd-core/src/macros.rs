//! Declarative generators for backend-specific vector implementations.

/// Generate element mappings and the selected native operator implementations.
///
/// Each element row names its vector type and lane-count expression, then lists
/// only the operators that backend implements. The `simd` marker is in scope
/// inside the lane expression, so scalable backends may query a runtime VL.
/// Omitting an operator leaves that Rust operator unavailable for the vector.
#[macro_export]
macro_rules! impl_simd_ops {
    (
        $backend:ty;
        $(
            $ty:ty => ($vec:ty, $lanes:expr) {
                add: $add:expr,
                sub: $sub:expr
                $(, $op:ident : $body:expr)* $(,)?
            }
        )+
    ) => {
        $(
            impl $crate::Element<$backend> for $ty {
                type Vector = $vec;
                #[inline]
                fn lanes(simd: $backend) -> usize { let _ = simd; $lanes }
            }
            impl ::core::ops::Add for $vec {
                type Output = Self;
                #[inline]
                fn add(self, rhs: Self) -> Self { ($add)(self, rhs) }
            }
            impl ::core::ops::Sub for $vec {
                type Output = Self;
                #[inline]
                fn sub(self, rhs: Self) -> Self { ($sub)(self, rhs) }
            }
            $( $crate::__impl_simd_op!($vec, $op, $body); )*
        )+
    };
}

/// Implementation details used by [`impl_simd_ops!`].
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_simd_op {
    ($vec:ty, add, $body:expr) => {
        impl ::core::ops::Add for $vec {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, sub, $body:expr) => {
        impl ::core::ops::Sub for $vec {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, mul, $body:expr) => {
        impl ::core::ops::Mul for $vec {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, div, $body:expr) => {
        impl ::core::ops::Div for $vec {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, rem, $body:expr) => {
        impl ::core::ops::Rem for $vec {
            type Output = Self;
            #[inline]
            fn rem(self, rhs: Self) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, neg, $body:expr) => {
        impl ::core::ops::Neg for $vec {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                ($body)(self)
            }
        }
    };
    ($vec:ty, bitand, $body:expr) => {
        impl ::core::ops::BitAnd for $vec {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, bitor, $body:expr) => {
        impl ::core::ops::BitOr for $vec {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, bitxor, $body:expr) => {
        impl ::core::ops::BitXor for $vec {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, not, $body:expr) => {
        impl ::core::ops::Not for $vec {
            type Output = Self;
            #[inline]
            fn not(self) -> Self {
                ($body)(self)
            }
        }
    };
    ($vec:ty, shl, $body:expr) => {
        impl ::core::ops::Shl<u32> for $vec {
            type Output = Self;
            #[inline]
            fn shl(self, rhs: u32) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, shr, $body:expr) => {
        impl ::core::ops::Shr<u32> for $vec {
            type Output = Self;
            #[inline]
            fn shr(self, rhs: u32) -> Self {
                ($body)(self, rhs)
            }
        }
    };
    ($vec:ty, abs, $body:expr) => {
        impl $crate::SimdAbs for $vec {
            #[inline]
            fn abs(self) -> Self {
                ($body)(self)
            }
        }
    };
    ($vec:ty, sqrt, $body:expr) => {
        impl $crate::SimdSqrt for $vec {
            #[inline]
            fn sqrt(self) -> Self {
                ($body)(self)
            }
        }
    };
}
