extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Error, Ident, LitInt, Result, Token,
};

struct AddEmptyDescriptorBindings {
    descriptor_builder: Ident,
    previous_non_empty_binding: i32,
    next_non_empty_binding: u32,
}

impl Parse for AddEmptyDescriptorBindings {
    fn parse(input: ParseStream) -> Result<Self> {
        let descriptor_builder: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let previous_non_empty_binding_lit: LitInt = input.parse()?;
        let previous_non_empty_binding = previous_non_empty_binding_lit.base10_parse::<i32>()?;
        if previous_non_empty_binding < -1 {
            return Err(Error::new(
                previous_non_empty_binding_lit.span(),
                format!("illegal layout binding id: {}", previous_non_empty_binding,),
            ));
        }
        input.parse::<Token![,]>()?;
        let next_non_empty_binding_lit: LitInt = input.parse()?;
        let next_non_empty_binding = next_non_empty_binding_lit.base10_parse::<u32>()?;
        if next_non_empty_binding as i32 <= previous_non_empty_binding {
            return Err(Error::new(
                next_non_empty_binding_lit.span(),
                format!(
                    "the previous_non_empty_binding({}) is not less than the \
                    next_non_empty_binding({})",
                    previous_non_empty_binding, next_non_empty_binding
                ),
            ));
        }
        Ok(Self {
            descriptor_builder,
            previous_non_empty_binding,
            next_non_empty_binding,
        })
    }
}

#[proc_macro]
pub fn add_empty_descriptor_bindings(input: TokenStream) -> TokenStream {
    let AddEmptyDescriptorBindings {
        descriptor_builder,
        previous_non_empty_binding,
        next_non_empty_binding,
    } = parse_macro_input!(input as AddEmptyDescriptorBindings);
    let mut res = TokenStream::new();
    res.extend(
        (((previous_non_empty_binding + 1) as u32)..next_non_empty_binding).map(
            |i| -> TokenStream {
                let gen = quote! {
                    let #descriptor_builder = #descriptor_builder.add_empty().chain_err(|| {
                        format!("fail to add empty descriptor set binding at {}", #i)
                    })?;
                };
                gen.into()
            },
        ),
    );
    res
}
