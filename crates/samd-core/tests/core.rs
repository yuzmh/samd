use core::fmt::Debug;
use core::ops::Add;
use samd_core::{
    Element, F32, F64, I8, I16, Scalar, Simd, SimdAbs, SimdMask, SimdSqrt, SimdVector, U8, U32, U64,
};
#[cfg(not(target_arch = "wasm32"))]
use std::panic::{AssertUnwindSafe, catch_unwind};

fn add<T>(lhs: &[T], rhs: &[T], output: &mut [T])
where
    T: Element<Scalar>,
    T::Vector: Add<Output = T::Vector>,
{
    let mut i = 0;
    while i < lhs.len() {
        let chunk = Scalar.chunk::<T>(lhs.len() - i);
        chunk.store(output, i, chunk.load(lhs, i) + chunk.load(rhs, i));
        i += chunk.len();
    }
}

fn assert_adds<T, const N: usize>(lhs: [T; N], rhs: [T; N], expected: [T; N])
where
    T: Copy + Element<Scalar> + PartialEq + Debug,
    T::Vector: Add<Output = T::Vector>,
{
    let mut output = expected;
    add(&lhs, &rhs, &mut output);
    assert_eq!(output, expected);
}

#[test]
fn scalar_arithmetic_covers_all_elements_and_wrapping_edges() {
    assert_adds([i8::MAX], [1], [i8::MIN]);
    assert_adds([u8::MAX], [1], [0]);
    assert_adds([2_i16], [3], [5]);
    assert_adds([2_u16], [3], [5]);
    assert_adds([2_i32], [3], [5]);
    assert_adds([2_u32], [3], [5]);
    assert_adds([2_i64], [3], [5]);
    assert_adds([2_u64], [3], [5]);
    assert_adds([2.0_f32], [3.0], [5.0]);
    assert_adds([2.0_f64], [3.0], [5.0]);

    assert_eq!((I16(i16::MIN) - I16(1)).0, i16::MAX);
    assert_eq!((U32(u32::MAX) + U32(1)).0, 0);
    assert_eq!((U8(0xff) * U8(2)).0, 0xfe);
    assert_eq!(I8(i8::MIN).abs().0, i8::MIN);
    assert_eq!(I8(-3) >> 1, I8(-2));
    assert_eq!(U8(0x80) >> 1, U8(0x40));
    assert_eq!((I8(1) << 8).0, 1);
    assert_eq!((U64(1) << 64).0, 1);
}

#[test]
fn scalar_chunks_masks_reductions_and_float_edges_are_defined() {
    let chunk = Scalar.chunk::<f32>(1);
    let vector = chunk.load(&[2.5], 0);
    assert_eq!(chunk.len(), 1);
    assert_eq!(chunk.sum(vector), 2.5);
    assert_eq!(chunk.prod(vector), 2.5);
    assert_eq!(chunk.min(vector), 2.5);
    assert_eq!(chunk.max(vector), 2.5);

    let empty = Scalar.chunk::<f32>(0);
    let empty_vector = empty.load(&[], 0);
    assert_eq!(empty.sum(empty_vector), 0.0);
    assert_eq!(empty.prod(empty_vector), 1.0);
    assert_eq!(empty.min(empty_vector), f32::INFINITY);
    assert_eq!(empty.max(empty_vector), f32::NEG_INFINITY);
    assert!(!empty.any(F32(1.0).gt(&F32(0.0))));
    assert!(empty.all(F32(1.0).gt(&F32(0.0))));
    assert_eq!(empty.count_true(F32(1.0).gt(&F32(0.0))), 0);

    let greater = F32(3.0).gt(&F32(2.0));
    assert!(greater.all());
    assert_eq!(greater.count_true(), 1);
    assert_eq!(greater.select(F32(7.0), F32(9.0)), F32(7.0));

    let unordered = F32(f32::NAN).gt(&F32(0.0));
    assert!(unordered.none());
    assert_eq!(unordered.select(F32(7.0), F32(9.0)), F32(9.0));
    assert!(F32(0.0).eq(&F32(-0.0)).all());

    assert_eq!(F32(-4.0).abs(), F32(4.0));
    assert_eq!(F32(9.0).sqrt(), F32(3.0));
    assert_eq!(F64(16.0).sqrt(), F64(4.0));
    assert_eq!(F32(-0.0).abs().0.to_bits(), 0.0_f32.to_bits());
    assert_eq!(F32(-0.0).sqrt().0.to_bits(), (-0.0_f32).to_bits());
    assert_eq!(F32(f32::NEG_INFINITY).abs(), F32(f32::INFINITY));
    assert!(F32(f32::NAN).abs().0.is_nan());
    assert!(F32(-1.0).sqrt().0.is_nan());
    assert_eq!(F32(f32::INFINITY).sqrt(), F32(f32::INFINITY));
    assert!(F64(f64::NAN).sqrt().0.is_nan());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn scalar_memory_rejects_out_of_bounds_access() {
    let load = catch_unwind(|| <F32 as SimdVector<f32, Scalar>>::load(Scalar, &[1.0], 1, 1));
    assert!(load.is_err());

    let mut output = [1.0_f32];
    let store = catch_unwind(AssertUnwindSafe(|| {
        <F32 as SimdVector<f32, Scalar>>::store(F32(2.0), &mut output, 1, 1);
    }));
    assert!(store.is_err());
    assert_eq!(output, [1.0]);
}
