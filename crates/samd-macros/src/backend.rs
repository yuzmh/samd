//! Central metadata for backends and their selection requirements.

use std::cmp::Reverse;
use std::collections::BTreeSet;

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::quote;

struct Backend {
    marker: &'static str,
    target_arch: Option<&'static str>,
    required_features: &'static [&'static str],
    target_features: &'static [&'static str],
    priority: u8,
    unsafe_constructor: bool,
    selection: SelectionKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SelectionKind {
    Fallback,
    CompileTime,
    Runtime(&'static str),
}

const BACKENDS: &[Backend] = &[
    Backend {
        marker: "Scalar",
        target_arch: None,
        required_features: &[],
        target_features: &[],
        priority: 0,
        unsafe_constructor: false,
        selection: SelectionKind::Fallback,
    },
    Backend {
        marker: "WasmSimd128",
        target_arch: Some("wasm32"),
        required_features: &["simd128"],
        target_features: &["simd128"],
        priority: 1,
        unsafe_constructor: false,
        selection: SelectionKind::CompileTime,
    },
    Backend {
        marker: "Sse2",
        target_arch: Some("x86_64"),
        required_features: &["sse2"],
        target_features: &["sse2"],
        priority: 2,
        unsafe_constructor: false,
        selection: SelectionKind::Runtime("Sse2"),
    },
    Backend {
        marker: "Avx2",
        target_arch: Some("x86_64"),
        required_features: &["sse2", "avx", "avx2", "osxsave", "xcr0_ymm"],
        target_features: &["avx2"],
        priority: 3,
        unsafe_constructor: true,
        selection: SelectionKind::Runtime("Avx2"),
    },
];

/// A direct call selected entirely from compile-time target configuration.
pub struct StaticSelection {
    pub cfg: TokenStream,
    pub marker: TokenStream,
    pub avx2_constructor: bool,
}

/// One arm of a runtime backend match.
pub struct RuntimeArm {
    pub marker: TokenStream,
    pub variant: Ident,
    pub target_feature: Option<&'static str>,
}

/// Runtime selection metadata for one target architecture.
pub struct RuntimeSelection {
    pub cfg: TokenStream,
    pub detector: TokenStream,
    pub enum_path: TokenStream,
    pub arms: Vec<RuntimeArm>,
}

/// All backend selection paths derived from the central metadata table.
pub struct Selections {
    pub static_calls: Vec<StaticSelection>,
    pub runtime: Vec<RuntimeSelection>,
}

/// Builds statically pruned calls and best-first runtime matches.
pub fn selections(samd_path: &TokenStream) -> Selections {
    let scalar = backend("Scalar");
    let wasm = backend("WasmSimd128");
    let sse2 = backend("Sse2");
    let avx2 = backend("Avx2");

    let wasm_cfg = target_cfg(wasm);
    let wasm_dominates_scalar = dominates(wasm, scalar);
    let scalar_cfg = if wasm_dominates_scalar {
        quote!(not(#wasm_cfg))
    } else {
        quote!(all())
    };

    let mut static_calls = vec![StaticSelection {
        cfg: quote!(all(not(target_arch = "x86_64"), #scalar_cfg)),
        marker: marker_path(samd_path, scalar.marker),
        avx2_constructor: false,
    }];
    static_calls.push(StaticSelection {
        cfg: wasm_cfg,
        marker: marker_path(samd_path, wasm.marker),
        avx2_constructor: false,
    });
    static_calls.push(StaticSelection {
        cfg: quote!(all(target_arch = "x86_64", target_feature = "avx2")),
        marker: marker_path(samd_path, avx2.marker),
        avx2_constructor: avx2.unsafe_constructor,
    });
    // Runtime backends are sorted by priority so the match tries the strongest
    // implementation first. SSE2 is guaranteed by the x86_64 target baseline.
    let mut runtime_backends = [avx2, sse2];
    runtime_backends.sort_by_key(|item| Reverse(item.priority));
    let arms = runtime_backends
        .into_iter()
        .map(|item| RuntimeArm {
            marker: marker_path(samd_path, item.marker),
            variant: Ident::new(
                match item.selection {
                    SelectionKind::Runtime(variant) => variant,
                    SelectionKind::Fallback | SelectionKind::CompileTime => unreachable!(),
                },
                Span::call_site(),
            ),
            target_feature: item.target_features.first().copied(),
        })
        .collect();

    let runtime = vec![RuntimeSelection {
        cfg: quote!(all(target_arch = "x86_64", not(target_feature = "avx2"))),
        detector: quote!(#samd_path::__private::best_x86_backend),
        enum_path: quote!(#samd_path::__private::X86Backend),
        arms,
    }];

    Selections {
        static_calls,
        runtime,
    }
}

fn backend(name: &str) -> &'static Backend {
    BACKENDS
        .iter()
        .find(|backend| backend.marker == name)
        .expect("backend metadata contains the requested implementation")
}

fn dominates(stronger: &Backend, weaker: &Backend) -> bool {
    stronger.priority > weaker.priority && feature_set(stronger).is_superset(&feature_set(weaker))
}

fn feature_set(backend: &Backend) -> BTreeSet<&'static str> {
    backend.required_features.iter().copied().collect()
}

fn target_cfg(backend: &Backend) -> TokenStream {
    let target_arch = backend
        .target_arch
        .expect("hardware backend has a target architecture");
    let arch = Literal::string(target_arch);
    let features = backend
        .target_features
        .iter()
        .map(|feature| Literal::string(feature));
    quote!(all(target_arch = #arch, #(target_feature = #features),*))
}

fn marker_path(samd_path: &TokenStream, marker: &str) -> TokenStream {
    let marker = Ident::new(marker, Span::call_site());
    quote!(#samd_path::__private::#marker)
}
