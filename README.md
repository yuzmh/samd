# SAMD

SAMD is a Rust library for writing portable SIMD kernels. It provides one
kernel API across supported hardware backends and a scalar fallback.

Add `samd = "0.1"` to your `Cargo.toml`, annotate a function with
`#[samd::simd(s)]`, and call it like an ordinary Rust function. SAMD selects
the implementation at the function boundary, so backend selection does not
happen inside the kernel's vector operations.

## Backends

| Architecture | Supported backends | Selection |
| --- | --- | --- |
| x86_64 | `AVX2`, `SSE2` | AVX2 when supported by the CPU and operating system; otherwise SSE2 |
| wasm32 | `SIMD128`, `Scalar` | SIMD128 when enabled at compile time; otherwise Scalar |
| Other targets | `Scalar` | Scalar |

The same kernel handles full vectors and partial tails. Chunk loads and stores
access only the active slice elements. The core and backend crates support
`no_std` use.

## Example

```rust
#[samd::simd(s)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut total = 0.0;
    let mut i = 0;
    while i < a.len() {
        let chunk = s.chunk::<f32>(a.len() - i);
        let x = chunk.load(a, i);
        let y = chunk.load(b, i);
        total += chunk.sum(x * y);
        i += chunk.len();
    }

    total
}

fn main() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [5.0, 4.0, 3.0, 2.0, 1.0];

    assert_eq!(dot(&a, &b), 35.0);
}
```

## Project

- License: [Apache-2.0](LICENSE)
- Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md)
