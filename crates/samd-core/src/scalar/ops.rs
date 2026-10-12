//! Generated scalar arithmetic and element mappings.

use libm::{fabs, fabsf, sqrt, sqrtf};

use super::{F32, F64, I8, I16, I32, I64, Scalar, U8, U16, U32, U64};
use crate::impl_simd_ops;

impl_simd_ops! {
    Scalar;
    f32 => (F32, 1) {
        add: |a: F32, b: F32| F32(a.0 + b.0),
        sub: |a: F32, b: F32| F32(a.0 - b.0),
        mul: |a: F32, b: F32| F32(a.0 * b.0),
        div: |a: F32, b: F32| F32(a.0 / b.0),
        rem: |a: F32, b: F32| F32(a.0 % b.0),
        neg: |a: F32| F32(-a.0),
        abs: |a: F32| F32(fabsf(a.0)),
        sqrt: |a: F32| F32(sqrtf(a.0)),
    }
    f64 => (F64, 1) {
        add: |a: F64, b: F64| F64(a.0 + b.0),
        sub: |a: F64, b: F64| F64(a.0 - b.0),
        mul: |a: F64, b: F64| F64(a.0 * b.0),
        div: |a: F64, b: F64| F64(a.0 / b.0),
        rem: |a: F64, b: F64| F64(a.0 % b.0),
        neg: |a: F64| F64(-a.0),
        abs: |a: F64| F64(fabs(a.0)),
        sqrt: |a: F64| F64(sqrt(a.0)),
    }
    i8 => (I8, 1) {
        add: |a: I8, b: I8| I8(a.0.wrapping_add(b.0)),
        sub: |a: I8, b: I8| I8(a.0.wrapping_sub(b.0)),
        mul: |a: I8, b: I8| I8(a.0.wrapping_mul(b.0)),
        neg: |a: I8| I8(a.0.wrapping_neg()),
        abs: |a: I8| I8(a.0.wrapping_abs()),
        bitand: |a: I8, b: I8| I8(a.0 & b.0),
        bitor: |a: I8, b: I8| I8(a.0 | b.0),
        bitxor: |a: I8, b: I8| I8(a.0 ^ b.0),
        not: |a: I8| I8(!a.0),
        shl: |a: I8, n: u32| I8(a.0.wrapping_shl(n)),
        shr: |a: I8, n: u32| I8(a.0.wrapping_shr(n)),
    }
    u8 => (U8, 1) {
        add: |a: U8, b: U8| U8(a.0.wrapping_add(b.0)),
        sub: |a: U8, b: U8| U8(a.0.wrapping_sub(b.0)),
        mul: |a: U8, b: U8| U8(a.0.wrapping_mul(b.0)),
        bitand: |a: U8, b: U8| U8(a.0 & b.0),
        bitor: |a: U8, b: U8| U8(a.0 | b.0),
        bitxor: |a: U8, b: U8| U8(a.0 ^ b.0),
        not: |a: U8| U8(!a.0),
        shl: |a: U8, n: u32| U8(a.0.wrapping_shl(n)),
        shr: |a: U8, n: u32| U8(a.0.wrapping_shr(n)),
    }
    i16 => (I16, 1) {
        add: |a: I16, b: I16| I16(a.0.wrapping_add(b.0)),
        sub: |a: I16, b: I16| I16(a.0.wrapping_sub(b.0)),
        mul: |a: I16, b: I16| I16(a.0.wrapping_mul(b.0)),
        neg: |a: I16| I16(a.0.wrapping_neg()),
        abs: |a: I16| I16(a.0.wrapping_abs()),
        bitand: |a: I16, b: I16| I16(a.0 & b.0),
        bitor: |a: I16, b: I16| I16(a.0 | b.0),
        bitxor: |a: I16, b: I16| I16(a.0 ^ b.0),
        not: |a: I16| I16(!a.0),
        shl: |a: I16, n: u32| I16(a.0.wrapping_shl(n)),
        shr: |a: I16, n: u32| I16(a.0.wrapping_shr(n)),
    }
    u16 => (U16, 1) {
        add: |a: U16, b: U16| U16(a.0.wrapping_add(b.0)),
        sub: |a: U16, b: U16| U16(a.0.wrapping_sub(b.0)),
        mul: |a: U16, b: U16| U16(a.0.wrapping_mul(b.0)),
        bitand: |a: U16, b: U16| U16(a.0 & b.0),
        bitor: |a: U16, b: U16| U16(a.0 | b.0),
        bitxor: |a: U16, b: U16| U16(a.0 ^ b.0),
        not: |a: U16| U16(!a.0),
        shl: |a: U16, n: u32| U16(a.0.wrapping_shl(n)),
        shr: |a: U16, n: u32| U16(a.0.wrapping_shr(n)),
    }
    i32 => (I32, 1) {
        add: |a: I32, b: I32| I32(a.0.wrapping_add(b.0)),
        sub: |a: I32, b: I32| I32(a.0.wrapping_sub(b.0)),
        mul: |a: I32, b: I32| I32(a.0.wrapping_mul(b.0)),
        neg: |a: I32| I32(a.0.wrapping_neg()),
        abs: |a: I32| I32(a.0.wrapping_abs()),
        bitand: |a: I32, b: I32| I32(a.0 & b.0),
        bitor: |a: I32, b: I32| I32(a.0 | b.0),
        bitxor: |a: I32, b: I32| I32(a.0 ^ b.0),
        not: |a: I32| I32(!a.0),
        shl: |a: I32, n: u32| I32(a.0.wrapping_shl(n)),
        shr: |a: I32, n: u32| I32(a.0.wrapping_shr(n)),
    }
    u32 => (U32, 1) {
        add: |a: U32, b: U32| U32(a.0.wrapping_add(b.0)),
        sub: |a: U32, b: U32| U32(a.0.wrapping_sub(b.0)),
        mul: |a: U32, b: U32| U32(a.0.wrapping_mul(b.0)),
        bitand: |a: U32, b: U32| U32(a.0 & b.0),
        bitor: |a: U32, b: U32| U32(a.0 | b.0),
        bitxor: |a: U32, b: U32| U32(a.0 ^ b.0),
        not: |a: U32| U32(!a.0),
        shl: |a: U32, n: u32| U32(a.0.wrapping_shl(n)),
        shr: |a: U32, n: u32| U32(a.0.wrapping_shr(n)),
    }
    i64 => (I64, 1) {
        add: |a: I64, b: I64| I64(a.0.wrapping_add(b.0)),
        sub: |a: I64, b: I64| I64(a.0.wrapping_sub(b.0)),
        mul: |a: I64, b: I64| I64(a.0.wrapping_mul(b.0)),
        neg: |a: I64| I64(a.0.wrapping_neg()),
        abs: |a: I64| I64(a.0.wrapping_abs()),
        bitand: |a: I64, b: I64| I64(a.0 & b.0),
        bitor: |a: I64, b: I64| I64(a.0 | b.0),
        bitxor: |a: I64, b: I64| I64(a.0 ^ b.0),
        not: |a: I64| I64(!a.0),
        shl: |a: I64, n: u32| I64(a.0.wrapping_shl(n)),
        shr: |a: I64, n: u32| I64(a.0.wrapping_shr(n)),
    }
    u64 => (U64, 1) {
        add: |a: U64, b: U64| U64(a.0.wrapping_add(b.0)),
        sub: |a: U64, b: U64| U64(a.0.wrapping_sub(b.0)),
        mul: |a: U64, b: U64| U64(a.0.wrapping_mul(b.0)),
        bitand: |a: U64, b: U64| U64(a.0 & b.0),
        bitor: |a: U64, b: U64| U64(a.0 | b.0),
        bitxor: |a: U64, b: U64| U64(a.0 ^ b.0),
        not: |a: U64| U64(!a.0),
        shl: |a: U64, n: u32| U64(a.0.wrapping_shl(n)),
        shr: |a: U64, n: u32| U64(a.0.wrapping_shr(n)),
    }
}
