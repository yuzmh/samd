# SAMD

SAMD is a Rust library for writing SIMD kernels with one portable API. Add
`#[samd::simd(s)]` to a function and use `s` to create and process vector
chunks. Call the function normally; SAMD selects its backend when the function
is called.

On x86_64, SAMD checks CPU and operating-system support once per kernel call
and selects AVX2 or the x86_64 baseline SSE2 implementation. A build whose
target guarantees AVX2 calls that implementation directly. On WebAssembly, selection is
compile-time: a build with `simd128` uses the SIMD128 backend, and a build
without it uses Scalar. Other targets currently use Scalar. Kernel code stays
the same across these backends.

Add SAMD to your application's `Cargo.toml`:

```toml
[dependencies]
samd = "0.1"
```

## Add two slices

```rust
#[samd::simd(s)]
fn add(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());

    let mut i = 0;
    while i < a.len() {
        let chunk = s.chunk::<f32>(a.len() - i);
        let x = chunk.load(a, i);
        let y = chunk.load(b, i);
        chunk.store(output, i, x + y);
        i += chunk.len();
    }
}

fn main() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [5.0, 4.0, 3.0, 2.0, 1.0];
    let mut output = [0.0; 5];

    add(&a, &b, &mut output);
    assert_eq!(output, [6.0; 5]);
}
```

`chunk` chooses how many elements to process next. An `f32` chunk can contain
up to eight elements with AVX2, four with SSE2 or WebAssembly SIMD128, and one
with Scalar; the last chunk may be smaller. `load` and `store` access only the
active elements, so the same loop safely handles empty slices and tails without
reading past a slice.

## Reduce a slice

Use the chunk's reduction methods to include only active elements. This matters
for a partial final vector, whose unused physical lanes are not part of the
input.

```rust
#[samd::simd(s)]
fn sum(values: &[f32]) -> f32 {
    let mut total = 0.0;
    let mut i = 0;

    while i < values.len() {
        let chunk = s.chunk::<f32>(values.len() - i);
        total += chunk.sum(chunk.load(values, i));
        i += chunk.len();
    }

    total
}

assert_eq!(sum(&[1.0, 2.0, 3.0, 4.0, 5.0]), 15.0);
assert_eq!(sum(&[]), 0.0);
```

The corresponding chunk methods are `sum`, `prod`, `min`, and `max`. Empty
reductions return their operation's identity: `0`, `1`, positive infinity,
and negative infinity, respectively. Floating-point reduction order can vary
between backends, so the final rounded bits are not guaranteed to match.

## Compare and select

Vector comparisons return a mask. Use that mask to select values or to count
matches in the active portion of a chunk.

```rust
#[samd::simd(s)]
fn maximum(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());

    let mut i = 0;
    while i < a.len() {
        let chunk = s.chunk::<f32>(a.len() - i);
        let x = chunk.load(a, i);
        let y = chunk.load(b, i);
        let greater = x.gt(&y);
        chunk.store(output, i, greater.select(x, y));
        i += chunk.len();
    }
}
```

Chunk masks also provide `any`, `all`, `none`, and `count_true`; each ignores
inactive tail lanes.

## Build for WebAssembly SIMD128

```sh
RUSTFLAGS="-C target-feature=+simd128" \
    cargo build --release --target wasm32-unknown-unknown
```

The runtime that loads the resulting module must support WebAssembly SIMD128.
WASM feature selection is compile-time; SAMD does not probe SIMD support inside
the module. Build without `+simd128` to use the scalar implementation instead.
