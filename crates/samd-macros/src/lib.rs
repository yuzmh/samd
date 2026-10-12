//! Procedural macros for the SAMD public API.

mod backend;
mod kernel;

use proc_macro::TokenStream;
use syn::{Ident as SynIdent, ItemFn, parse_macro_input};

/// Turns an ordinary function into a statically specialized SIMD kernel with
/// automatic backend selection at each call boundary.
///
/// The attribute argument names the injected backend marker used by the body.
/// The public function keeps its original name, visibility, parameters, generic
/// parameters, lifetimes, and return type. The generated wrapper dispatches once
/// before invoking a private generic kernel; it adds inferred bounds for
/// generic chunk element types.
///
/// The attribute supports safe synchronous free functions with identifier
/// parameter patterns. Element and operator requirements for generic chunk
/// types are added to the generated wrapper for its selected backend. Async,
/// unsafe, variadic, ABI-qualified, and opaque `impl Trait` return signatures
/// are rejected because their wrapper semantics cannot be preserved safely.
///
/// # Examples
///
/// ```text
/// #[samd::simd(s)]
/// pub fn multiply(a: &[f32], b: &[f32], out: &mut [f32]) {
///     let mut index = 0;
///     while index < a.len() {
///         let chunk = s.chunk::<f32>(a.len() - index);
///         chunk.store(out, index, chunk.load(a, index) * chunk.load(b, index));
///         index += chunk.len();
///     }
/// }
///
/// multiply(&a, &b, &mut out); // ordinary call; no marker argument
/// ```
#[proc_macro_attribute]
pub fn simd(attribute: TokenStream, item: TokenStream) -> TokenStream {
    let backend_ident = parse_macro_input!(attribute as SynIdent);
    let function = parse_macro_input!(item as ItemFn);

    match kernel::expand(&backend_ident, &function) {
        Ok(expansion) => expansion.into(),
        Err(error) => error.into_compile_error().into(),
    }
}
