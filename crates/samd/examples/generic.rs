#[samd::simd(s)]
fn add(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), output.len());
    let mut offset = 0;
    while offset < lhs.len() {
        let chunk = s.chunk::<f32>(lhs.len() - offset);
        chunk.store(
            output,
            offset,
            chunk.load(lhs, offset) + chunk.load(rhs, offset),
        );
        offset += chunk.len();
    }
}

#[samd::simd(backend)]
fn sum(data: &[f32]) -> f32 {
    let mut total = 0.0;
    let mut offset = 0;
    while offset < data.len() {
        let chunk = backend.chunk::<f32>(data.len() - offset);
        total += chunk.sum(chunk.load(data, offset));
        offset += chunk.len();
    }
    total
}

#[samd::simd(simd)]
fn compute(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), output.len());
    let mut offset = 0;
    while offset < lhs.len() {
        let chunk = simd.chunk::<f32>(lhs.len() - offset);
        let rhs_vector = chunk.load(rhs, offset);
        let product = chunk.load(lhs, offset) * rhs_vector;
        chunk.store(output, offset, product + chunk.load(lhs, offset));
        offset += chunk.len();
    }
}

fn main() {
    let lhs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let rhs = [5.0, 4.0, 3.0, 2.0, 1.0];
    let mut output = [0.0; 5];
    add(&lhs, &rhs, &mut output);
    assert_eq!(output, [6.0; 5]);
    assert_eq!(sum(&output), 30.0);
    compute(&lhs, &rhs, &mut output);
    assert_eq!(output, [6.0, 10.0, 12.0, 12.0, 10.0]);
}
