//! Attribute expansion for ordinary SAMD kernel functions.

use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use syn::visit::{self, Visit};
use syn::{
    Attribute, BinOp, Error, Expr, ExprBinary, ExprMethodCall, ExprUnary, FnArg, GenericArgument,
    GenericParam, Generics, ItemFn, Local, Pat, Result, ReturnType, Safety, Signature, Type, UnOp,
    Visibility, WherePredicate, parse_quote,
};

use crate::backend;

pub fn expand(backend_ident: &Ident, function: &ItemFn) -> Result<TokenStream> {
    validate_signature(backend_ident, function)?;
    let samd_path = samd_path()?;
    let helper_ident = format_ident!(
        "__samd_{}_kernel",
        function.sig.ident,
        span = Span::mixed_site()
    );
    let backend_type = format_ident!("__SamdBackend", span = Span::mixed_site());
    let argument_idents = function
        .sig
        .inputs
        .iter()
        .map(argument_ident)
        .collect::<Result<Vec<_>>>()?;
    let mut usage = KernelUsage::new(backend_ident);
    usage.visit_block(&function.block);
    let helper = build_helper(
        function,
        backend_ident,
        &backend_type,
        &helper_ident,
        &samd_path,
        &usage,
    );
    let selections = backend::selections(&samd_path);
    let wrappers = static_wrappers(
        function,
        &helper_ident,
        &argument_idents,
        &samd_path,
        &usage,
        &selections.static_calls,
    );
    let runtime_wrappers = selections.runtime.iter().map(|selection| {
        runtime_wrapper(
            function,
            &helper_ident,
            &argument_idents,
            &samd_path,
            &usage,
            selection,
        )
    });

    Ok(quote! {
        #helper
        #(#wrappers)*
        #(#runtime_wrappers)*
    })
}

fn build_helper(
    function: &ItemFn,
    backend_ident: &Ident,
    backend_type: &Ident,
    helper_ident: &Ident,
    samd_path: &TokenStream,
    usage: &KernelUsage,
) -> ItemFn {
    let mut helper = function.clone();
    helper.sig.ident = helper_ident.clone();
    helper.vis = Visibility::Inherited;
    helper.attrs = helper_attributes(&function.attrs);
    helper.attrs.push(parse_quote!(#[inline(always)]));
    helper
        .sig
        .inputs
        .insert(0, parse_quote!(#backend_ident: #backend_type));
    helper
        .sig
        .generics
        .params
        .push(GenericParam::Type(parse_quote!(#backend_type)));
    add_kernel_bounds(&mut helper, backend_type, samd_path, usage);

    let original_block = helper.block;
    helper.block = Box::new(parse_quote!({
        use #samd_path::{Element as _, Simd as _, SimdAbs as _, SimdMask as _, SimdSqrt as _, SimdVector as _};
        let _ = &#backend_ident;
        #original_block
    }));
    helper
}

fn static_wrappers(
    function: &ItemFn,
    helper_ident: &Ident,
    argument_idents: &[Ident],
    samd_path: &TokenStream,
    usage: &KernelUsage,
    selections: &[backend::StaticSelection],
) -> Vec<TokenStream> {
    selections
        .iter()
        .map(|selection| {
            let cfg = &selection.cfg;
            let marker = &selection.marker;
            let call = if selection.avx2_constructor {
                quote! {
                    // SAFETY: the cfg guarantees AVX2 is part of this crate's target baseline.
                    let backend = unsafe { #marker::new_unchecked() };
                    #helper_ident(backend, #(#argument_idents),*)
                }
            } else {
                quote!(#helper_ident(#marker, #(#argument_idents),*))
            };
            let mut wrapper = function.sig.clone();
            add_selected_bounds(&mut wrapper, marker, samd_path, usage);
            let attributes = &function.attrs;
            quote! {
                #[cfg(#cfg)]
                #(#attributes)*
                #wrapper {
                #call
                }
            }
        })
        .collect()
}

fn runtime_wrapper(
    function: &ItemFn,
    helper_ident: &Ident,
    argument_idents: &[Ident],
    samd_path: &TokenStream,
    usage: &KernelUsage,
    selection: &backend::RuntimeSelection,
) -> TokenStream {
    let cfg = &selection.cfg;
    let detector = &selection.detector;
    let enum_path = &selection.enum_path;
    let context = RuntimeContext {
        function,
        helper_ident,
        argument_idents,
        samd_path,
        usage,
        enum_path,
        selection_cfg: cfg,
    };
    let (entries, arms): (Vec<_>, Vec<_>) = selection
        .arms
        .iter()
        .map(|backend| runtime_arm(&context, backend))
        .unzip();
    let mut wrapper = function.sig.clone();
    add_runtime_wrapper_bounds(&mut wrapper, samd_path, &selection.arms, usage);
    let attributes = &function.attrs;
    quote! {
        #[cfg(#cfg)]
        #(#attributes)*
        #wrapper {
            match #detector() {
                #(#arms,)*
            }
        }
        #(#entries)*
    }
}

struct RuntimeContext<'a> {
    function: &'a ItemFn,
    helper_ident: &'a Ident,
    argument_idents: &'a [Ident],
    samd_path: &'a TokenStream,
    usage: &'a KernelUsage,
    enum_path: &'a TokenStream,
    selection_cfg: &'a TokenStream,
}

fn runtime_arm(
    context: &RuntimeContext<'_>,
    backend: &backend::RuntimeArm,
) -> (TokenStream, TokenStream) {
    let RuntimeContext {
        function,
        helper_ident,
        argument_idents,
        samd_path,
        usage,
        enum_path,
        selection_cfg,
    } = *context;
    let marker = &backend.marker;
    let variant = &backend.variant;
    let backend_initialization = if variant == "Avx2" {
        quote! {
            // SAFETY: the public wrapper enters this function only after the
            // AVX2 and OS YMM-state checks succeed.
            let backend = unsafe { #marker::new_unchecked() };
        }
    } else {
        quote!(let backend = #marker;)
    };
    let entry_ident = format_ident!(
        "__samd_{}_{}_entry",
        function.sig.ident,
        variant.to_string().to_lowercase(),
        span = Span::mixed_site()
    );
    let mut entry = function.sig.clone();
    entry.ident = entry_ident.clone();
    if backend.target_feature.is_some() {
        entry.safety = parse_quote!(unsafe);
    }
    add_selected_bounds(&mut entry, marker, samd_path, usage);
    let entry_attributes = dispatch_entry_attributes(&function.attrs);
    let target_feature = backend.target_feature.map(|feature| {
        let feature = syn::LitStr::new(feature, Span::call_site());
        quote!(#[target_feature(enable = #feature)])
    });
    let invoke = if backend.target_feature.is_some() {
        quote! {
            // SAFETY: this match arm is reached only after the runtime detector
            // selected the corresponding CPU and OS features.
            unsafe { #entry_ident(#(#argument_idents),*) }
        }
    } else {
        quote!(#entry_ident(#(#argument_idents),*))
    };
    let entry_tokens = quote! {
        #[cfg(#selection_cfg)]
        #(#entry_attributes)*
        #target_feature
        #entry {
            #backend_initialization
            #helper_ident(backend, #(#argument_idents),*)
        }
    };
    let arm_tokens = quote!(#enum_path::#variant => #invoke);
    (entry_tokens, arm_tokens)
}

fn add_runtime_wrapper_bounds(
    signature: &mut Signature,
    samd_path: &TokenStream,
    arms: &[backend::RuntimeArm],
    usage: &KernelUsage,
) {
    for arm in arms {
        add_selected_bounds(signature, &arm.marker, samd_path, usage);
    }
}

fn dispatch_entry_attributes(attributes: &[Attribute]) -> Vec<Attribute> {
    attributes
        .iter()
        .filter(|attribute| {
            attribute.path().is_ident("cfg")
                || attribute.path().is_ident("cfg_attr")
                || attribute.path().is_ident("allow")
                || attribute.path().is_ident("warn")
                || attribute.path().is_ident("deny")
                || attribute.path().is_ident("forbid")
                || attribute.path().is_ident("expect")
        })
        .cloned()
        .collect()
}

fn samd_path() -> Result<TokenStream> {
    match proc_macro_crate::crate_name("samd") {
        Ok(proc_macro_crate::FoundCrate::Itself) => Ok(quote!(::samd)),
        Ok(proc_macro_crate::FoundCrate::Name(name)) => {
            let ident = Ident::new(&name.replace('-', "_"), Span::call_site());
            Ok(quote!(::#ident))
        }
        Err(error) => Err(Error::new(
            Span::call_site(),
            format!("could not find the `samd` dependency for this macro: {error}"),
        )),
    }
}

fn validate_signature(backend_ident: &Ident, function: &ItemFn) -> Result<()> {
    let signature = &function.sig;
    if signature.asyncness.is_some()
        || signature.constness.is_some()
        || !matches!(signature.safety, Safety::Default)
        || signature.abi.is_some()
        || signature.variadic.is_some()
    {
        return Err(Error::new_spanned(
            signature,
            "`#[samd::simd]` supports safe, synchronous, non-const Rust functions only",
        ));
    }
    if matches!(&signature.output, ReturnType::Type(_, return_type) if matches!(return_type.as_ref(), Type::ImplTrait(_)))
    {
        return Err(Error::new_spanned(
            &signature.output,
            "`#[samd::simd]` does not support opaque `impl Trait` return types",
        ));
    }
    if signature.inputs.iter().any(|argument| {
        matches!(argument, FnArg::Typed(argument)
            if matches!(argument.pat.as_ref(), Pat::Ident(pattern) if pattern.ident == *backend_ident))
    }) {
        return Err(Error::new_spanned(
            signature,
            "the injected backend identifier must not shadow a function parameter",
        ));
    }
    if function.attrs.iter().any(|attribute| {
        attribute.path().is_ident("target_feature")
            || attribute.path().is_ident("no_mangle")
            || attribute.path().is_ident("export_name")
    }) {
        let attribute = function
            .attrs
            .iter()
            .find(|attribute| {
                attribute.path().is_ident("target_feature")
                    || attribute.path().is_ident("no_mangle")
                    || attribute.path().is_ident("export_name")
            })
            .expect("the attribute predicate found an unsupported function attribute");
        return Err(Error::new_spanned(
            attribute,
            "`#[samd::simd]` cannot be combined with `target_feature`, `no_mangle`, or `export_name`",
        ));
    }
    Ok(())
}

fn argument_ident(argument: &FnArg) -> Result<Ident> {
    match argument {
        FnArg::Typed(argument) => match argument.pat.as_ref() {
            Pat::Ident(pattern) if pattern.subpat.is_none() && pattern.by_ref.is_none() => {
                if argument.attrs.iter().any(|attribute| {
                    attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr")
                }) {
                    let attribute = argument
                        .attrs
                        .iter()
                        .find(|attribute| {
                            attribute.path().is_ident("cfg")
                                || attribute.path().is_ident("cfg_attr")
                        })
                        .expect("the attribute predicate found a conditional parameter");
                    return Err(Error::new_spanned(
                        attribute,
                        "conditional parameter attributes are unsupported on SIMD kernels",
                    ));
                }
                Ok(pattern.ident.clone())
            }
            pattern => Err(Error::new_spanned(
                pattern,
                "SIMD kernel parameters must use simple identifier patterns",
            )),
        },
        FnArg::Receiver(receiver) => Err(Error::new_spanned(
            receiver,
            "`#[samd::simd]` supports free functions, not methods with a receiver",
        )),
    }
}

fn helper_attributes(attributes: &[Attribute]) -> Vec<Attribute> {
    attributes
        .iter()
        .filter(|attribute| {
            attribute.path().is_ident("cfg")
                || attribute.path().is_ident("cfg_attr")
                || attribute.path().is_ident("cold")
                || attribute.path().is_ident("track_caller")
                || attribute.path().is_ident("allow")
                || attribute.path().is_ident("warn")
                || attribute.path().is_ident("deny")
                || attribute.path().is_ident("forbid")
                || attribute.path().is_ident("expect")
        })
        .cloned()
        .collect()
}

fn add_kernel_bounds(
    helper: &mut ItemFn,
    backend_type: &Ident,
    samd_path: &TokenStream,
    usage: &KernelUsage,
) {
    helper
        .sig
        .generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(#backend_type: #samd_path::Simd));

    for scalar in usage.elements.values() {
        add_element_bound(
            &mut helper.sig.generics,
            scalar,
            &quote!(#backend_type),
            samd_path,
        );
        let vector = quote!(<#scalar as #samd_path::Element<#backend_type>>::Vector);
        if let Some((_, capabilities)) = usage.capabilities.get(&quote!(#scalar).to_string()) {
            for capability in capabilities {
                let predicate =
                    capability.predicate(&vector, scalar, &quote!(#backend_type), samd_path);
                helper
                    .sig
                    .generics
                    .make_where_clause()
                    .predicates
                    .push(predicate);
            }
        }
    }
}

fn add_selected_bounds(
    signature: &mut Signature,
    backend: &TokenStream,
    samd_path: &TokenStream,
    usage: &KernelUsage,
) {
    for scalar in usage.elements.values() {
        add_element_bound(&mut signature.generics, scalar, backend, samd_path);
        let vector = quote!(<#scalar as #samd_path::Element<#backend>>::Vector);
        if let Some((_, capabilities)) = usage.capabilities.get(&quote!(#scalar).to_string()) {
            for capability in capabilities {
                signature
                    .generics
                    .make_where_clause()
                    .predicates
                    .push(capability.predicate(&vector, scalar, backend, samd_path));
            }
        }
    }
}

fn add_element_bound(
    generics: &mut Generics,
    scalar: &Type,
    backend: &TokenStream,
    samd_path: &TokenStream,
) {
    if let Type::Path(type_path) = scalar
        && type_path.qself.is_none()
        && type_path.path.segments.len() == 1
    {
        let scalar_ident = &type_path.path.segments[0].ident;
        if let Some(GenericParam::Type(type_parameter)) = generics.params.iter_mut().find(
            |parameter| matches!(parameter, GenericParam::Type(type_parameter) if type_parameter.ident == *scalar_ident),
        ) {
            type_parameter
                .bounds
                .push(parse_quote!(#samd_path::Element<#backend>));
            return;
        }
    }

    generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(#scalar: #samd_path::Element<#backend>));
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Capability {
    Mul,
    Div,
    Rem,
    Neg,
    BitAnd,
    BitOr,
    BitXor,
    Not,
    Shl,
    Shr,
    Abs,
    Sqrt,
    Select,
}

impl Capability {
    fn predicate(
        self,
        vector: &TokenStream,
        scalar: &Type,
        backend: &TokenStream,
        samd_path: &TokenStream,
    ) -> WherePredicate {
        let mask = quote!(<#vector as #samd_path::SimdVector<#scalar, #backend>>::Mask);
        match self {
            Self::Mul => parse_quote!(#vector: ::core::ops::Mul<Output = #vector>),
            Self::Div => parse_quote!(#vector: ::core::ops::Div<Output = #vector>),
            Self::Rem => parse_quote!(#vector: ::core::ops::Rem<Output = #vector>),
            Self::Neg => parse_quote!(#vector: ::core::ops::Neg<Output = #vector>),
            Self::BitAnd => parse_quote!(#vector: ::core::ops::BitAnd<Output = #vector>),
            Self::BitOr => parse_quote!(#vector: ::core::ops::BitOr<Output = #vector>),
            Self::BitXor => parse_quote!(#vector: ::core::ops::BitXor<Output = #vector>),
            Self::Not => parse_quote!(#vector: ::core::ops::Not<Output = #vector>),
            Self::Shl => parse_quote!(#vector: ::core::ops::Shl<u32, Output = #vector>),
            Self::Shr => parse_quote!(#vector: ::core::ops::Shr<u32, Output = #vector>),
            Self::Abs => parse_quote!(#vector: #samd_path::SimdAbs),
            Self::Sqrt => parse_quote!(#vector: #samd_path::SimdSqrt),
            Self::Select => parse_quote!(
                #vector: #samd_path::MaskSelect<#mask>
            ),
        }
    }
}

struct KernelUsage {
    backend_ident: Ident,
    elements: BTreeMap<String, Type>,
    chunks: BTreeMap<String, Type>,
    vectors: BTreeMap<String, Type>,
    capabilities: BTreeMap<String, (Type, BTreeSet<Capability>)>,
}

impl KernelUsage {
    fn new(backend_ident: &Ident) -> Self {
        Self {
            backend_ident: backend_ident.clone(),
            elements: BTreeMap::new(),
            chunks: BTreeMap::new(),
            vectors: BTreeMap::new(),
            capabilities: BTreeMap::new(),
        }
    }

    fn add_capability(&mut self, scalar: &Type, capability: Capability) {
        let key = quote!(#scalar).to_string();
        self.capabilities
            .entry(key)
            .or_insert_with(|| (scalar.clone(), BTreeSet::new()))
            .1
            .insert(capability);
    }

    fn element_for_expr(&self, expression: &Expr) -> Option<Type> {
        match expression {
            Expr::Path(path) => {
                let name = path.path.get_ident()?.to_string();
                self.vectors.get(&name).cloned()
            }
            Expr::Paren(paren) => self.element_for_expr(&paren.expr),
            Expr::Group(group) => self.element_for_expr(&group.expr),
            Expr::Unary(unary) => self.element_for_expr(&unary.expr),
            Expr::Binary(binary) => self
                .element_for_expr(&binary.left)
                .or_else(|| self.element_for_expr(&binary.right)),
            Expr::MethodCall(method) => match method.method.to_string().as_str() {
                "load" | "splat" => self.chunk_for_expr(&method.receiver),
                "abs" | "sqrt" | "select" | "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                    self.element_for_expr(&method.receiver)
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn chunk_for_expr(&self, expression: &Expr) -> Option<Type> {
        let Expr::Path(path) = expression else {
            return None;
        };
        let name = path.path.get_ident()?.to_string();
        self.chunks.get(&name).cloned()
    }

    fn infer_local_type(&mut self, expression: &Expr, name: &Ident) {
        if let Some(element) = self.chunk_type(expression) {
            self.chunks.insert(name.to_string(), element);
        } else if let Some(element) = self.element_for_expr(expression) {
            self.vectors.insert(name.to_string(), element);
        }
    }

    fn chunk_type(&self, expression: &Expr) -> Option<Type> {
        let Expr::MethodCall(method) = expression else {
            return None;
        };
        if method.method != "chunk"
            || !matches!(method.receiver.as_ref(), Expr::Path(path)
                if path.path.is_ident(&self.backend_ident))
        {
            return None;
        }
        match method.turbofish.as_ref()?.args.first()? {
            GenericArgument::Type(element) => Some(element.clone()),
            _ => None,
        }
    }

    fn add_element(&mut self, element: &Type) {
        self.elements
            .entry(quote!(#element).to_string())
            .or_insert_with(|| element.clone());
    }
}

impl<'ast> Visit<'ast> for KernelUsage {
    fn visit_local(&mut self, local: &'ast Local) {
        visit::visit_local(self, local);
        if let (Pat::Ident(pattern), Some(initializer)) = (&local.pat, local.init.as_ref()) {
            self.infer_local_type(&initializer.expr, &pattern.ident);
        }
    }

    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        if expression.method == "chunk"
            && matches!(expression.receiver.as_ref(), Expr::Path(path)
                if path.path.is_ident(&self.backend_ident))
            && let Some(GenericArgument::Type(element)) = expression
                .turbofish
                .as_ref()
                .and_then(|arguments| arguments.args.first())
        {
            self.add_element(element);
        }
        if let Some(scalar) = self.element_for_expr(&expression.receiver) {
            match expression.method.to_string().as_str() {
                "abs" => self.add_capability(&scalar, Capability::Abs),
                "sqrt" => self.add_capability(&scalar, Capability::Sqrt),
                "select" => self.add_capability(&scalar, Capability::Select),
                _ => {}
            }
        }
        visit::visit_expr_method_call(self, expression);
    }

    fn visit_expr_binary(&mut self, expression: &'ast ExprBinary) {
        let capability = match expression.op {
            BinOp::Mul(_) => Some(Capability::Mul),
            BinOp::Div(_) => Some(Capability::Div),
            BinOp::Rem(_) => Some(Capability::Rem),
            BinOp::BitAnd(_) => Some(Capability::BitAnd),
            BinOp::BitOr(_) => Some(Capability::BitOr),
            BinOp::BitXor(_) => Some(Capability::BitXor),
            BinOp::Shl(_) => Some(Capability::Shl),
            BinOp::Shr(_) => Some(Capability::Shr),
            _ => None,
        };
        if let Some(capability) = capability {
            if let Some(scalar) = self.element_for_expr(&expression.left) {
                self.add_capability(&scalar, capability);
            }
            if let Some(scalar) = self.element_for_expr(&expression.right) {
                self.add_capability(&scalar, capability);
            }
        }
        visit::visit_expr_binary(self, expression);
    }

    fn visit_expr_unary(&mut self, expression: &'ast ExprUnary) {
        match expression.op {
            UnOp::Neg(_) => {
                if let Some(scalar) = self.element_for_expr(&expression.expr) {
                    self.add_capability(&scalar, Capability::Neg);
                }
            }
            UnOp::Not(_) => {
                if let Some(scalar) = self.element_for_expr(&expression.expr) {
                    self.add_capability(&scalar, Capability::Not);
                }
            }
            _ => {}
        }
        visit::visit_expr_unary(self, expression);
    }
}
