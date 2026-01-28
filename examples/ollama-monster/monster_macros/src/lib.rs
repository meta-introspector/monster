// Monster Prime Introspection Macros for mistral.rs
// Instruments model inference to capture weights and activations

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Macro to instrument a function with Monster prime analysis
/// 
/// Usage:
/// ```
/// #[monster_introspect]
/// fn forward_pass(input: &Tensor) -> Tensor {
///     // ... model code
/// }
/// ```
#[proc_macro_attribute]
pub fn monster_introspect(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_block = &input.block;
    let fn_sig = &input.sig;
    
    let output = quote! {
        #fn_sig {
            // Capture entry
            let _guard = crate::monster::MonsterGuard::enter(stringify!(#fn_name));
            
            // Original function body
            let result = (|| #fn_block)();
            
            // Capture exit with result
            _guard.exit_with_result(&result);
            
            result
        }
    };
    
    TokenStream::from(output)
}

/// Macro to analyze tensor for Monster prime patterns
#[proc_macro]
pub fn analyze_tensor(input: TokenStream) -> TokenStream {
    let output = quote! {
        {
            let tensor_data = #input;
            crate::monster::analyze_prime_patterns(&tensor_data)
        }
    };
    
    TokenStream::from(output)
}

/// Macro to trace weight loading
#[proc_macro_attribute]
pub fn trace_weights(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_block = &input.block;
    let fn_sig = &input.sig;
    
    let output = quote! {
        #fn_sig {
            println!("🔬 Loading weights: {}", stringify!(#fn_name));
            
            let result = (|| #fn_block)();
            
            // Analyze loaded weights
            if let Some(weights) = result.as_ref() {
                crate::monster::analyze_weights(stringify!(#fn_name), weights);
            }
            
            result
        }
    };
    
    TokenStream::from(output)
}
