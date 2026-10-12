//! Compare representative compute-heavy `f32` kernels with SAMD and Pulp.
//!
//! Run with `cargo bench -p samd --bench bench`.

#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;
use std::env::consts::{ARCH, OS};
use std::hint::black_box;
use std::process::Command;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pulp::{Arch as PulpArch, Simd as PulpSimd, WithSimd};

const LENGTHS: [usize; 8] = [1, 3, 8, 31, 1_024, 65_539, 1_048_579, 16_777_219];
const ALIGNMENT_OFFSETS: [usize; 2] = [0, 1];

#[samd::simd(s)]
fn samd_dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());

    let block = s.chunk::<f32>(usize::MAX);
    if lhs.len() < block.len() {
        let mut total = 0.0;
        for (&lhs, &rhs) in lhs.iter().zip(rhs) {
            let product = lhs * rhs;
            total += product;
        }
        return total;
    }
    if lhs.len() < block.len() * 4 {
        let mut total = block.splat(0.0);
        let mut offset = 0;
        while lhs.len() - offset >= block.len() {
            total = total + block.load(lhs, offset) * block.load(rhs, offset);
            offset += block.len();
        }
        let mut scalar_total = 0.0;
        while offset < lhs.len() {
            let product = lhs[offset] * rhs[offset];
            scalar_total += product;
            offset += 1;
        }
        return block.sum(total) + scalar_total;
    }

    let mut totals0 = block.splat(0.0);
    let mut totals1 = block.splat(0.0);
    let mut totals2 = block.splat(0.0);
    let mut totals3 = block.splat(0.0);
    let mut offset = 0;
    while lhs.len() - offset >= block.len() * 4 {
        totals0 = totals0 + block.load(lhs, offset) * block.load(rhs, offset);
        totals1 =
            totals1 + block.load(lhs, offset + block.len()) * block.load(rhs, offset + block.len());
        totals2 = totals2
            + block.load(lhs, offset + block.len() * 2) * block.load(rhs, offset + block.len() * 2);
        totals3 = totals3
            + block.load(lhs, offset + block.len() * 3) * block.load(rhs, offset + block.len() * 3);
        offset += block.len() * 4;
    }
    while lhs.len() - offset >= block.len() {
        totals0 = totals0 + block.load(lhs, offset) * block.load(rhs, offset);
        offset += block.len();
    }
    let mut scalar_total = 0.0;
    while offset < lhs.len() {
        let product = lhs[offset] * rhs[offset];
        scalar_total += product;
        offset += 1;
    }

    block.sum((totals0 + totals1) + (totals2 + totals3)) + scalar_total
}

#[samd::simd(s)]
fn samd_euclidean_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());

    let block = s.chunk::<f32>(usize::MAX);
    if lhs.len() < block.len() {
        let mut total = 0.0;
        for (&lhs, &rhs) in lhs.iter().zip(rhs) {
            let difference = lhs - rhs;
            let square = difference * difference;
            total += square;
        }
        return total.sqrt();
    }
    if lhs.len() < block.len() * 4 {
        let mut total = block.splat(0.0);
        let mut offset = 0;
        while lhs.len() - offset >= block.len() {
            let difference = block.load(lhs, offset) - block.load(rhs, offset);
            total = total + difference * difference;
            offset += block.len();
        }
        let mut scalar_total = 0.0;
        while offset < lhs.len() {
            let difference = lhs[offset] - rhs[offset];
            let square = difference * difference;
            scalar_total += square;
            offset += 1;
        }
        return (block.sum(total) + scalar_total).sqrt();
    }

    let mut totals0 = block.splat(0.0);
    let mut totals1 = block.splat(0.0);
    let mut totals2 = block.splat(0.0);
    let mut totals3 = block.splat(0.0);
    let mut offset = 0;
    while lhs.len() - offset >= block.len() * 4 {
        let difference0 = block.load(lhs, offset) - block.load(rhs, offset);
        totals0 = totals0 + difference0 * difference0;
        let difference1 =
            block.load(lhs, offset + block.len()) - block.load(rhs, offset + block.len());
        totals1 = totals1 + difference1 * difference1;
        let difference2 =
            block.load(lhs, offset + block.len() * 2) - block.load(rhs, offset + block.len() * 2);
        totals2 = totals2 + difference2 * difference2;
        let difference3 =
            block.load(lhs, offset + block.len() * 3) - block.load(rhs, offset + block.len() * 3);
        totals3 = totals3 + difference3 * difference3;
        offset += block.len() * 4;
    }
    while lhs.len() - offset >= block.len() {
        let difference = block.load(lhs, offset) - block.load(rhs, offset);
        totals0 = totals0 + difference * difference;
        offset += block.len();
    }
    let mut scalar_total = 0.0;
    while offset < lhs.len() {
        let difference = lhs[offset] - rhs[offset];
        let square = difference * difference;
        scalar_total += square;
        offset += 1;
    }

    (block.sum((totals0 + totals1) + (totals2 + totals3)) + scalar_total).sqrt()
}

#[samd::simd(s)]
fn samd_sqrt(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    let block = s.chunk::<f32>(usize::MAX);
    let mut offset = 0;
    while input.len() - offset >= block.len() {
        block.store(output, offset, block.load(input, offset).sqrt());
        offset += block.len();
    }
    while offset < input.len() {
        output[offset] = input[offset].sqrt();
        offset += 1;
    }
}

#[samd::simd(s)]
fn samd_normalize(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    let block = s.chunk::<f32>(usize::MAX);
    if input.len() < block.len() {
        let norm = input.iter().map(|&value| value * value).sum::<f32>().sqrt();
        for (&value, output) in input.iter().zip(output) {
            *output = value / norm;
        }
        return;
    }
    if input.len() < block.len() * 4 {
        let mut total = block.splat(0.0);
        let mut offset = 0;
        while input.len() - offset >= block.len() {
            total = total + block.load(input, offset) * block.load(input, offset);
            offset += block.len();
        }
        let mut scalar_total = 0.0;
        while offset < input.len() {
            let square = input[offset] * input[offset];
            scalar_total += square;
            offset += 1;
        }
        let norm = (block.sum(total) + scalar_total).sqrt();
        offset = 0;
        while input.len() - offset >= block.len() {
            let values = block.load(input, offset);
            block.store(output, offset, values / block.splat(norm));
            offset += block.len();
        }
        while offset < input.len() {
            output[offset] = input[offset] / norm;
            offset += 1;
        }
        return;
    }

    let mut totals0 = block.splat(0.0);
    let mut totals1 = block.splat(0.0);
    let mut totals2 = block.splat(0.0);
    let mut totals3 = block.splat(0.0);
    let mut offset = 0;
    while input.len() - offset >= block.len() * 4 {
        totals0 = totals0 + block.load(input, offset) * block.load(input, offset);
        totals1 = totals1
            + block.load(input, offset + block.len()) * block.load(input, offset + block.len());
        totals2 = totals2
            + block.load(input, offset + block.len() * 2)
                * block.load(input, offset + block.len() * 2);
        totals3 = totals3
            + block.load(input, offset + block.len() * 3)
                * block.load(input, offset + block.len() * 3);
        offset += block.len() * 4;
    }
    while input.len() - offset >= block.len() {
        totals0 = totals0 + block.load(input, offset) * block.load(input, offset);
        offset += block.len();
    }
    let mut scalar_total = 0.0;
    while offset < input.len() {
        let square = input[offset] * input[offset];
        scalar_total += square;
        offset += 1;
    }

    let norm = (block.sum((totals0 + totals1) + (totals2 + totals3)) + scalar_total).sqrt();
    offset = 0;
    while input.len() - offset >= block.len() {
        let values = block.load(input, offset);
        block.store(output, offset, values / block.splat(norm));
        offset += block.len();
    }
    while offset < input.len() {
        output[offset] = input[offset] / norm;
        offset += 1;
    }
}

#[samd::simd(s)]
fn samd_dispatch_probe(input: &[f32]) -> f32 {
    let block = s.chunk::<f32>(usize::MAX);
    let values = block.load(input, 0);
    block.sum(values + values)
}

struct PulpDispatchProbe<'a>(&'a [f32]);

impl WithSimd for PulpDispatchProbe<'_> {
    type Output = f32;

    // Keep the probe equivalent to the optimized kernel call and avoid
    // measuring outlined AVX2 intrinsic wrappers in Pulp.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn with_simd<S: PulpSimd>(self, simd: S) -> Self::Output {
        let (vectors, tail) = S::as_simd_f32s(self.0);
        let mut total = 0.0;
        for &vector in vectors {
            let doubled = simd.add_f32s(vector, vector);
            total += simd.reduce_sum_f32s(doubled);
        }
        for &value in tail {
            total += value + value;
        }
        total
    }
}

fn benchmark_dispatch(criterion: &mut Criterion) {
    let input = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
    let mut group = criterion.benchmark_group("dispatch_overhead");
    configure_group(&mut group);
    group.bench_function("samd plus one vector", |bencher| {
        bencher.iter(|| black_box(samd_dispatch_probe(black_box(&input))));
    });
    group.bench_function("pulp plus one vector", |bencher| {
        bencher.iter(|| black_box(PulpArch::new().dispatch(PulpDispatchProbe(black_box(&input)))));
    });
    group.finish();
}

#[inline]
fn pulp_term<S: PulpSimd, const DISTANCE: bool>(simd: S, lhs: S::f32s, rhs: S::f32s) -> S::f32s {
    if DISTANCE {
        let difference = simd.sub_f32s(lhs, rhs);
        simd.mul_f32s(difference, difference)
    } else {
        simd.mul_f32s(lhs, rhs)
    }
}

// Keep this helper in the caller so Pulp and SAMD both measure a fused kernel.
#[allow(clippy::inline_always)]
#[inline(always)]
fn pulp_sum_products<S: PulpSimd, const DISTANCE: bool>(simd: S, lhs: &[f32], rhs: &[f32]) -> f32 {
    let (lhs_vectors, lhs_tail) = S::as_simd_f32s(lhs);
    let (rhs_vectors, rhs_tail) = S::as_simd_f32s(rhs);
    assert_eq!(lhs_vectors.len(), rhs_vectors.len());
    assert_eq!(lhs_tail.len(), rhs_tail.len());

    let zero = simd.splat_f32s(0.0);
    let mut accumulators = [zero; 4];
    let (lhs_blocks, lhs_extra) = pulp::as_arrays::<4, _>(lhs_vectors);
    let (rhs_blocks, rhs_extra) = pulp::as_arrays::<4, _>(rhs_vectors);

    for (lhs, rhs) in lhs_blocks.iter().zip(rhs_blocks) {
        for ((accumulator, &lhs), &rhs) in accumulators.iter_mut().zip(lhs).zip(rhs) {
            *accumulator = simd.add_f32s(*accumulator, pulp_term::<S, DISTANCE>(simd, lhs, rhs));
        }
    }

    for (&lhs, &rhs) in lhs_extra.iter().zip(rhs_extra) {
        accumulators[0] = simd.add_f32s(accumulators[0], pulp_term::<S, DISTANCE>(simd, lhs, rhs));
    }

    let combined = simd.add_f32s(
        simd.add_f32s(accumulators[0], accumulators[1]),
        simd.add_f32s(accumulators[2], accumulators[3]),
    );
    let mut total = simd.reduce_sum_f32s(combined);
    for (&lhs, &rhs) in lhs_tail.iter().zip(rhs_tail) {
        let term = if DISTANCE {
            let difference = lhs - rhs;
            difference * difference
        } else {
            lhs * rhs
        };
        total += term;
    }

    total
}

struct PulpDotProduct<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl WithSimd for PulpDotProduct<'_> {
    type Output = f32;

    #[inline]
    fn with_simd<S: PulpSimd>(self, simd: S) -> Self::Output {
        pulp_sum_products::<S, false>(simd, self.lhs, self.rhs)
    }
}

struct PulpDistance<'a> {
    lhs: &'a [f32],
    rhs: &'a [f32],
}

impl WithSimd for PulpDistance<'_> {
    type Output = f32;

    #[inline]
    fn with_simd<S: PulpSimd>(self, simd: S) -> Self::Output {
        pulp_sum_products::<S, true>(simd, self.lhs, self.rhs).sqrt()
    }
}

struct PulpSqrt<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for PulpSqrt<'_> {
    type Output = ();

    #[inline]
    fn with_simd<S: PulpSimd>(self, simd: S) {
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
        assert_eq!(input_vectors.len(), output_vectors.len());
        assert_eq!(input_tail.len(), output_tail.len());

        for (&input, output) in input_vectors.iter().zip(output_vectors) {
            *output = simd.sqrt_f32s(input);
        }
        for (&input, output) in input_tail.iter().zip(output_tail) {
            *output = input.sqrt();
        }
    }
}

struct PulpNormalize<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for PulpNormalize<'_> {
    type Output = ();

    #[inline]
    fn with_simd<S: PulpSimd>(self, simd: S) {
        let norm = pulp_sum_products::<S, false>(simd, self.input, self.input).sqrt();
        let scale = simd.splat_f32s(norm);
        let (input_vectors, input_tail) = S::as_simd_f32s(self.input);
        let (output_vectors, output_tail) = S::as_mut_simd_f32s(self.output);
        assert_eq!(input_vectors.len(), output_vectors.len());
        assert_eq!(input_tail.len(), output_tail.len());

        for (&input, output) in input_vectors.iter().zip(output_vectors) {
            *output = simd.div_f32s(input, scale);
        }
        for (&input, output) in input_tail.iter().zip(output_tail) {
            *output = input / norm;
        }
    }
}

fn pulp_dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    PulpArch::new().dispatch(PulpDotProduct { lhs, rhs })
}

fn pulp_euclidean_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    PulpArch::new().dispatch(PulpDistance { lhs, rhs })
}

fn pulp_sqrt(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    PulpArch::new().dispatch(PulpSqrt { input, output });
}

fn pulp_normalize(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    PulpArch::new().dispatch(PulpNormalize { input, output });
}

fn input(length: usize, offset: usize, multiplier: usize) -> Vec<f32> {
    (0..length + offset)
        .map(|index| {
            let Ok(lane) = u8::try_from(index * multiplier % 31) else {
                unreachable!("a remainder modulo 31 always fits in u8");
            };
            (f32::from(lane) + 1.0) * 0.125
        })
        .collect()
}

fn close(actual: f32, expected: f64) -> bool {
    // `f32` reduction order differs; permit up to 0.2% variation on long inputs.
    (f64::from(actual) - expected).abs() <= expected.abs() * 2.0e-3
}

fn benchmark_binary_reduction<S, P>(
    criterion: &mut Criterion,
    name: &str,
    samd_kernel: S,
    pulp_kernel: P,
    expected: fn(&[f32], &[f32]) -> f64,
) where
    S: Fn(&[f32], &[f32]) -> f32,
    P: Fn(&[f32], &[f32]) -> f32,
{
    let mut group = criterion.benchmark_group(name);
    configure_group(&mut group);

    for length in LENGTHS {
        set_throughput(&mut group, length);
        for offset in ALIGNMENT_OFFSETS {
            let lhs_storage = input(length, offset, 7);
            let rhs_storage = input(length, offset, 13);
            let lhs = &lhs_storage[offset..];
            let rhs = &rhs_storage[offset..];
            let expected_result = expected(lhs, rhs);
            let samd_result = samd_kernel(lhs, rhs);
            let pulp_result = pulp_kernel(lhs, rhs);
            assert!(
                close(samd_result, expected_result),
                "{name}: SAMD mismatch at len={length}, offset={offset}: got {samd_result}, expected {expected_result}"
            );
            assert!(
                close(pulp_result, expected_result),
                "{name}: Pulp mismatch at len={length}, offset={offset}: got {pulp_result}, expected {expected_result}"
            );

            let input_description = format!("len={length}, offset={offset}");
            group.bench_function(BenchmarkId::new("samd", &input_description), |bencher| {
                bencher.iter(|| black_box(samd_kernel(black_box(lhs), black_box(rhs))));
            });
            group.bench_function(BenchmarkId::new("pulp", &input_description), |bencher| {
                bencher.iter(|| black_box(pulp_kernel(black_box(lhs), black_box(rhs))));
            });
        }
    }

    group.finish();
}

fn benchmark_unary_output<S, P>(
    criterion: &mut Criterion,
    name: &str,
    samd_kernel: S,
    pulp_kernel: P,
    expected: fn(f32) -> f32,
) where
    S: Fn(&[f32], &mut [f32]),
    P: Fn(&[f32], &mut [f32]),
{
    let mut group = criterion.benchmark_group(name);
    configure_group(&mut group);

    for length in LENGTHS {
        set_throughput(&mut group, length);
        for offset in ALIGNMENT_OFFSETS {
            let input_storage = input(length, offset, 7);
            let input = &input_storage[offset..];
            let mut samd_output = vec![f32::NAN; length];
            let mut pulp_output = vec![f32::NAN; length];
            samd_kernel(input, &mut samd_output);
            pulp_kernel(input, &mut pulp_output);

            for (&value, &actual) in input.iter().zip(&samd_output) {
                assert!(
                    close(actual, f64::from(expected(value))),
                    "{name}: SAMD mismatch at len={length}, offset={offset}"
                );
            }
            for (&value, &actual) in input.iter().zip(&pulp_output) {
                assert!(
                    close(actual, f64::from(expected(value))),
                    "{name}: Pulp mismatch at len={length}, offset={offset}"
                );
            }

            let input_description = format!("len={length}, offset={offset}");
            group.bench_function(BenchmarkId::new("samd", &input_description), |bencher| {
                bencher.iter(|| {
                    samd_kernel(black_box(input), black_box(&mut samd_output));
                    black_box(&samd_output);
                });
            });
            group.bench_function(BenchmarkId::new("pulp", &input_description), |bencher| {
                bencher.iter(|| {
                    pulp_kernel(black_box(input), black_box(&mut pulp_output));
                    black_box(&pulp_output);
                });
            });
        }
    }

    group.finish();
}

fn benchmark_normalize<S, P>(criterion: &mut Criterion, samd_kernel: S, pulp_kernel: P)
where
    S: Fn(&[f32], &mut [f32]),
    P: Fn(&[f32], &mut [f32]),
{
    let mut group = criterion.benchmark_group("normalize");
    configure_group(&mut group);

    for length in LENGTHS {
        set_throughput(&mut group, length);
        for offset in ALIGNMENT_OFFSETS {
            let input_storage = input(length, offset, 7);
            let input = &input_storage[offset..];
            let mut samd_output = vec![f32::NAN; length];
            let mut pulp_output = vec![f32::NAN; length];
            let expected_norm = scalar_dot_product(input, input).sqrt();
            samd_kernel(input, &mut samd_output);
            pulp_kernel(input, &mut pulp_output);

            for ((&value, &samd_value), &pulp_value) in
                input.iter().zip(&samd_output).zip(&pulp_output)
            {
                let expected = f64::from(value) / expected_norm;
                assert!(
                    close(samd_value, expected),
                    "SAMD normalize mismatch at len={length}, offset={offset}"
                );
                assert!(
                    close(pulp_value, expected),
                    "Pulp normalize mismatch at len={length}, offset={offset}"
                );
            }

            let input_description = format!("len={length}, offset={offset}");
            group.bench_function(BenchmarkId::new("samd", &input_description), |bencher| {
                bencher.iter(|| {
                    samd_kernel(black_box(input), black_box(&mut samd_output));
                    black_box(&samd_output);
                });
            });
            group.bench_function(BenchmarkId::new("pulp", &input_description), |bencher| {
                bencher.iter(|| {
                    pulp_kernel(black_box(input), black_box(&mut pulp_output));
                    black_box(&pulp_output);
                });
            });
        }
    }

    group.finish();
}

fn scalar_dot_product(lhs: &[f32], rhs: &[f32]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| f64::from(lhs) * f64::from(rhs))
        .sum::<f64>()
}

fn scalar_euclidean_distance(lhs: &[f32], rhs: &[f32]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| {
            let difference = f64::from(lhs) - f64::from(rhs);
            difference * difference
        })
        .sum::<f64>()
        .sqrt()
}

fn scalar_sqrt(value: f32) -> f32 {
    value.sqrt()
}

fn benchmark_simd_kernels(criterion: &mut Criterion) {
    print_metadata();
    benchmark_dispatch(criterion);
    benchmark_binary_reduction(
        criterion,
        "dot_product",
        samd_dot_product,
        pulp_dot_product,
        scalar_dot_product,
    );
    benchmark_binary_reduction(
        criterion,
        "euclidean_distance",
        samd_euclidean_distance,
        pulp_euclidean_distance,
        scalar_euclidean_distance,
    );
    benchmark_unary_output(criterion, "sqrt", samd_sqrt, pulp_sqrt, scalar_sqrt);
    benchmark_normalize(criterion, samd_normalize, pulp_normalize);
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(250));
}

fn set_throughput(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    length: usize,
) {
    if length != 0 {
        group.throughput(Throughput::Elements(length as u64));
    }
}

fn print_metadata() {
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || "unknown".to_owned(),
            |output| String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        );

    #[cfg(target_arch = "x86_64")]
    let samd_backend = if cfg!(target_feature = "avx2") {
        "AVX2 (compile-time)"
    } else if is_x86_feature_detected!("avx2") {
        "AVX2 (runtime)"
    } else {
        "SSE2"
    };

    #[cfg(not(target_arch = "x86_64"))]
    let samd_backend = "target default";

    eprintln!("benchmark suite: SAMD vs Pulp");
    eprintln!("kernels: dot_product, euclidean_distance, sqrt, normalize");
    eprintln!("element type: f32");
    eprintln!("lengths: {LENGTHS:?}");
    eprintln!("alignment offsets (elements): {ALIGNMENT_OFFSETS:?}");
    eprintln!("target: {ARCH}-{OS}");
    eprintln!("SAMD backend: {samd_backend}");
    eprintln!("Pulp backend: {:?}", PulpArch::new());
    eprintln!("rustc: {rustc}");
    eprintln!("profile: release");
    eprintln!("Criterion configuration: 100 ms warm-up, 250 ms measurement, 10 samples");
    eprintln!(
        "throughput: elements per second (Criterion); command-line options may override timing"
    );
}

criterion_group!(benches, benchmark_simd_kernels);
criterion_main!(benches);
