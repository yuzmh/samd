use samd::{Scalar, Simd};

#[samd::simd(s)]
fn active_lanes(remaining: usize) -> usize {
    s.chunk::<f32>(remaining).len()
}

#[samd::simd(s)]
fn add(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), output.len());

    let mut i = 0;
    while i < lhs.len() {
        let chunk = s.chunk::<f32>(lhs.len() - i);
        chunk.store(output, i, chunk.load(lhs, i) + chunk.load(rhs, i));
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn multiply(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), output.len());

    let mut i = 0;
    while i < lhs.len() {
        let chunk = s.chunk::<f32>(lhs.len() - i);
        chunk.store(output, i, chunk.load(lhs, i) * chunk.load(rhs, i));
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn sum(values: &[f32]) -> f32 {
    let mut result = 0.0;
    let mut i = 0;
    while i < values.len() {
        let chunk = s.chunk::<f32>(values.len() - i);
        result += chunk.sum(chunk.load(values, i));
        i += chunk.len();
    }
    result
}

#[samd::simd(s)]
fn product(values: &[f32]) -> f32 {
    let mut result = 1.0;
    let mut i = 0;
    while i < values.len() {
        let chunk = s.chunk::<f32>(values.len() - i);
        result *= chunk.prod(chunk.load(values, i));
        i += chunk.len();
    }
    result
}

#[samd::simd(s)]
fn minmax(values: &[f32]) -> (f32, f32) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut i = 0;
    while i < values.len() {
        let chunk = s.chunk::<f32>(values.len() - i);
        minimum = minimum.min(chunk.min(chunk.load(values, i)));
        maximum = maximum.max(chunk.max(chunk.load(values, i)));
        i += chunk.len();
    }
    (minimum, maximum)
}

#[samd::simd(s)]
fn compare_select(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), output.len());

    let mut i = 0;
    while i < lhs.len() {
        let chunk = s.chunk::<f32>(lhs.len() - i);
        let x = chunk.load(lhs, i);
        let y = chunk.load(rhs, i);
        chunk.store(output, i, x.gt(&y).select(x, y));
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn mask_summary(lhs: &[f32], rhs: &[f32]) -> (bool, bool, usize) {
    assert_eq!(lhs.len(), rhs.len());

    let mut any = false;
    let mut all = true;
    let mut count = 0;
    let mut i = 0;
    while i < lhs.len() {
        let chunk = s.chunk::<f32>(lhs.len() - i);
        any |= chunk.any(chunk.load(lhs, i).gt(&chunk.load(rhs, i)));
        all &= chunk.all(chunk.load(lhs, i).gt(&chunk.load(rhs, i)));
        count += chunk.count_true(chunk.load(lhs, i).gt(&chunk.load(rhs, i)));
        i += chunk.len();
    }
    (any, all, count)
}

#[samd::simd(s)]
fn integer_edges(input: &[i32], output: &mut [i32]) {
    assert_eq!(input.len(), output.len());

    let mut i = 0;
    while i < input.len() {
        let chunk = s.chunk::<i32>(input.len() - i);
        let absolute = chunk.load(input, i).abs();
        let shifted = chunk.load(input, i) << 2;
        chunk.store(output, i, absolute ^ shifted);
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn bitwise<T: Copy>(lhs: &[T], rhs: &[T], output: &mut [T]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), output.len());

    let mut i = 0;
    while i < lhs.len() {
        let chunk = s.chunk::<T>(lhs.len() - i);
        let first = !chunk.load(lhs, i) & chunk.load(rhs, i);
        let second = chunk.load(lhs, i) ^ chunk.load(rhs, i);
        chunk.store(output, i, first | second);
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn shift<T: Copy>(input: &[T], left: &mut [T], right: &mut [T], count: u32) {
    assert_eq!(input.len(), left.len());
    assert_eq!(input.len(), right.len());

    let mut i = 0;
    while i < input.len() {
        let chunk = s.chunk::<T>(input.len() - i);
        chunk.store(left, i, chunk.load(input, i) << count);
        chunk.store(right, i, chunk.load(input, i) >> count);
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn fill<T: Copy>(value: T, output: &mut [T]) {
    let mut i = 0;
    while i < output.len() {
        let chunk = s.chunk::<T>(output.len() - i);
        chunk.store(output, i, chunk.splat(value));
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn float_unary(input: &[f64], output: &mut [f64]) {
    assert_eq!(input.len(), output.len());

    let mut i = 0;
    while i < input.len() {
        let chunk = s.chunk::<f64>(input.len() - i);
        chunk.store(output, i, chunk.load(input, i).abs().sqrt());
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn generic_multiply<T: Copy>(lhs: &[T], rhs: &[T], output: &mut [T]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), output.len());

    let mut i = 0;
    while i < lhs.len() {
        let chunk = s.chunk::<T>(lhs.len() - i);
        chunk.store(output, i, chunk.load(lhs, i) * chunk.load(rhs, i));
        i += chunk.len();
    }
}

#[samd::simd(s)]
fn identity<'a, T: Copy>(value: &'a T) -> &'a T {
    value
}

fn sample(index: usize, seed: usize) -> f32 {
    let value = u16::try_from((index * 37 + seed * 97) % 2048).expect("sample fits u16");
    (f32::from(value) - 1024.0) / 256.0
}

fn close64(actual: f64, expected: f64) -> bool {
    actual.to_bits() == expected.to_bits()
        || (actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= 1.0e-12 * actual.abs().max(expected.abs()).max(1.0))
}

#[test]
fn kernels_match_reference_for_tails_and_unaligned_slices() {
    for length in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 4096, 65_539] {
        for input_offset in [0, 1, 3] {
            let lhs_storage: Vec<_> = (0..length + input_offset).map(|i| sample(i, 1)).collect();
            let rhs_storage: Vec<_> = (0..length + input_offset).map(|i| sample(i, 2)).collect();
            let lhs = &lhs_storage[input_offset..input_offset + length];
            let rhs = &rhs_storage[input_offset..input_offset + length];
            let output_offset = input_offset + 1;
            let mut output = vec![f32::NAN; length + output_offset + 1];

            add(lhs, rhs, &mut output[output_offset..output_offset + length]);
            for i in 0..length {
                assert_eq!(output[output_offset + i], lhs[i] + rhs[i]);
            }

            multiply(lhs, rhs, &mut output[output_offset..output_offset + length]);
            for i in 0..length {
                assert_eq!(output[output_offset + i], lhs[i] * rhs[i]);
            }

            if length <= 257 {
                let expected_sum = lhs.iter().copied().sum::<f32>();
                assert!((sum(lhs) - expected_sum).abs() <= 0.01);
            }

            compare_select(lhs, rhs, &mut output[output_offset..output_offset + length]);
            for i in 0..length {
                let expected = if lhs[i] > rhs[i] { lhs[i] } else { rhs[i] };
                assert_eq!(output[output_offset + i], expected);
            }

            assert!(output[..output_offset].iter().all(|value| value.is_nan()));
            assert!(
                output[output_offset + length..]
                    .iter()
                    .all(|value| value.is_nan())
            );
        }
    }
}

#[test]
fn reductions_and_masks_cover_empty_partial_nan_and_zero_cases() {
    assert_eq!(sum(&[]), 0.0);
    assert_eq!(product(&[]), 1.0);
    assert_eq!(minmax(&[]), (f32::INFINITY, f32::NEG_INFINITY));
    assert!((product(&[0.5, 2.0, -1.5, 4.0, 0.25]) - -1.5).abs() < f32::EPSILON);

    let values = [-3.0, 4.0, f32::NAN, 10.0];
    let (minimum, maximum) = minmax(&values);
    assert_eq!(minimum, -3.0);
    assert_eq!(maximum, 10.0);

    let (minimum, maximum) = minmax(&[0.0, -0.0]);
    // Cross-chunk scalar accumulation only guarantees numeric zero here.
    assert_eq!(minimum, 0.0);
    assert_eq!(maximum, 0.0);

    assert_eq!(mask_summary(&[], &[]), (false, true, 0));
    assert_eq!(
        mask_summary(&[3.0, 4.0, 5.0], &[1.0, 2.0, 3.0]),
        (true, true, 3)
    );
    assert_eq!(
        mask_summary(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]),
        (true, false, 1)
    );
    assert_eq!(mask_summary(&[f32::NAN], &[0.0]), (false, false, 0));
}

#[test]
fn integer_and_float_kernels_match_scalar_semantics() {
    let input = [i32::MIN, -9, -1, 2, 7];
    let mut output = [0; 5];
    integer_edges(&input, &mut output);
    for (actual, value) in output.into_iter().zip(input) {
        assert_eq!(actual, value.wrapping_abs() ^ value.wrapping_shl(2));
    }

    let input = [-4.0_f64, -0.0, 9.0, 16.0, f64::INFINITY];
    let mut output = [0.0; 5];
    float_unary(&input, &mut output);
    for (actual, value) in output.into_iter().zip(input) {
        let expected = value.abs().sqrt();
        assert!(close64(actual, expected));
    }
    assert!(!output[1].is_sign_negative());
}

#[test]
fn integer_bitwise_and_shift_kernels_match_scalar_semantics() {
    macro_rules! check_bitwise {
        ($ty:ty) => {{
            let lhs: [$ty; 5] = [0, 1, 0x55, 0x6a, <$ty>::MAX];
            let rhs: [$ty; 5] = [<$ty>::MAX, 0, 0x33, 0x4c, 0x5a];
            let mut actual = [0; 5];
            bitwise(&lhs, &rhs, &mut actual);
            for ((&a, &b), result) in lhs.iter().zip(&rhs).zip(actual) {
                assert_eq!(result, (!a & b) | (a ^ b));
            }
            let mut broadcast = [0; 5];
            fill(<$ty>::MAX, &mut broadcast);
            assert!(broadcast.into_iter().all(|value| value == <$ty>::MAX));
        }};
    }

    check_bitwise!(i8);
    check_bitwise!(u8);
    check_bitwise!(i16);
    check_bitwise!(u16);
    check_bitwise!(i32);
    check_bitwise!(u32);
    check_bitwise!(i64);
    check_bitwise!(u64);

    let signed = [i16::MIN, -17, -1, 3, i16::MAX];
    let mut left = [0; 5];
    let mut right = [0; 5];
    shift(&signed, &mut left, &mut right, 19);
    for (index, value) in signed.into_iter().enumerate() {
        assert_eq!(left[index], value.wrapping_shl(3));
        assert_eq!(right[index], value.wrapping_shr(3));
    }

    let unsigned = [0_u64, 1, 0x8000_0000_0000_0000, u64::MAX, 37];
    let mut left = [0; 5];
    let mut right = [0; 5];
    shift(&unsigned, &mut left, &mut right, 65);
    for (index, value) in unsigned.into_iter().enumerate() {
        assert_eq!(left[index], value.wrapping_shl(1));
        assert_eq!(right[index], value.wrapping_shr(1));
    }
}

#[test]
fn macro_wrapper_preserves_generics_lifetimes_and_single_evaluation() {
    assert_eq!(active_lanes(0), 0);
    assert_eq!(active_lanes(1), 1);
    #[cfg(target_arch = "wasm32")]
    let expected_lanes = if cfg!(target_feature = "simd128") {
        4
    } else {
        1
    };
    #[cfg(target_arch = "x86_64")]
    let expected_lanes = match samd::__private::best_x86_backend() {
        samd::__private::X86Backend::Avx2 => 7,
        samd::__private::X86Backend::Sse2 => 4,
    };
    #[cfg(not(any(target_arch = "wasm32", target_arch = "x86_64")))]
    let expected_lanes = 1;
    assert_eq!(active_lanes(7), expected_lanes);

    let mut evaluations = 0;
    active_lanes({
        evaluations += 1;
        2
    });
    assert_eq!(evaluations, 1);

    let value = 37_u32;
    assert_eq!(*identity(&value), value);
    let mut output = [0; 5];
    generic_multiply(&[1, 2, 3, 4, 5], &[5, 4, 3, 2, 1], &mut output);
    assert_eq!(output, [5, 8, 9, 8, 5]);

    assert_eq!(Scalar.chunk::<f32>(3).len(), 1);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[should_panic(expected = "chunk load out of bounds")]
fn chunk_load_rejects_out_of_bounds_range() {
    let _ = Scalar.chunk::<f32>(1).load(&[], 0);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[should_panic(expected = "chunk store out of bounds")]
fn chunk_store_rejects_out_of_bounds_range() {
    let chunk = Scalar.chunk::<f32>(1);
    chunk.store(&mut [], 0, chunk.load(&[1.0], 0));
}
