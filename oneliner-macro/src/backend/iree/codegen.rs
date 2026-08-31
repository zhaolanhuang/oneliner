use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemStruct, LitStr, Meta, NestedMeta};

use super::{vmcu, IoLayout, IreeArtifacts};
use crate::args::ArenaArg;
use crate::utils::{path_lit, rust_ident};

pub(super) struct IoCodegen {
    pub(super) has_model_storage: bool,
    pub(super) model_field: TokenStream,
    pub(super) constructor_field: TokenStream,
    pub(super) execute_args: TokenStream,
    pub(super) execute_error: &'static str,
    pub(super) run_setup: TokenStream,
    pub(super) run_finish: TokenStream,
    pub(super) module_items: TokenStream,
    pub(super) storage_size: usize,
    pub(super) input_offset: usize,
    pub(super) output_offset: usize,
}

pub(super) fn expand(
    input_struct: ItemStruct,
    artifacts: IreeArtifacts,
    arena: ArenaArg,
) -> TokenStream {
    let struct_ident = &input_struct.ident;
    let struct_vis = &input_struct.vis;
    let struct_attrs = &input_struct.attrs;
    let module_ident = format_ident!("__oneliner_iree_{}", rust_ident(&struct_ident.to_string()));
    let paths = &artifacts.paths;
    let flow_rs = path_lit(&paths.flow_rs);
    let model_path = path_lit(&paths.model);
    let compile_input_path = path_lit(&paths.compile_input);
    let object_path = path_lit(&paths.object);
    let ir_path = path_lit(&paths.ir);
    let metadata_json_path = path_lit(&paths.metadata_json);
    let input_size = artifacts.io.input_size();
    let output_size = artifacts.io.output_size();
    let params_size = artifacts.params_size;
    let code_size = artifacts.code_size;
    let rodata_size = artifacts.rodata_size;
    let total_flash_size = params_size + code_size + rodata_size;
    let ram_size = artifacts.ram.transient_size;
    let stack_size = artifacts.ram.stack_size;
    let total_ram_size = artifacts.ram.total_size;
    let execute_fns = &artifacts.execute_fns;
    let query_fn = &artifacts.query_fn;
    let query_link_name = LitStr::new(&artifacts.query_link_name, Span::call_site());
    let input_type = artifacts.input_tensor.element_type.rust_tokens();
    let output_type = artifacts.output_tensor.element_type.rust_tokens();
    let [input_d0, input_d1, input_d2, input_d3] = artifacts.input_tensor.shape;
    let [output_d0, output_d1, output_d2, output_d3] = artifacts.output_tensor.shape;

    let io_codegen = match &artifacts.io {
        IoLayout::Separate { .. } => standard_fragments(&output_type),
        IoLayout::InPlace {
            storage_size,
            input,
            output,
        } => vmcu::compact_fragments(*storage_size, *input, *output, &output_type, &module_ident),
    };
    let IoCodegen {
        has_model_storage,
        model_field,
        constructor_field,
        execute_args,
        execute_error,
        run_setup,
        run_finish,
        module_items,
        storage_size: io_pool_size,
        input_offset,
        output_offset,
    } = io_codegen;

    let model_definition = match (arena, has_model_storage) {
        (ArenaArg::Owned, true) => quote! {
            #(#struct_attrs)*
            #struct_vis struct #struct_ident {
                __arena: ::oneliner::runtime::OwnedArena<#module_ident::Workspace>,
                #model_field
            }
        },
        (ArenaArg::Owned, false) => quote! {
            #(#struct_attrs)*
            #struct_vis struct #struct_ident {
                __arena: ::oneliner::runtime::OwnedArena<#module_ident::Workspace>,
            }
        },
        (ArenaArg::Shared, true) => quote! {
            #(#struct_attrs)*
            #struct_vis struct #struct_ident {
                #model_field
            }
        },
        (ArenaArg::Shared, false) => quote! {
            #input_struct
        },
    };

    let shared_arena_static = match arena {
        ArenaArg::Owned => quote! {},
        ArenaArg::Shared => quote! {
            pub(super) static ARENA_VAL:
                ::oneliner::runtime::ArenaStorage<Workspace> =
                ::oneliner::runtime::ArenaStorage::new(Workspace::new());
            pub(super) static ARENA:
                ::oneliner::runtime::SharedArena<Workspace> =
                ::oneliner::runtime::SharedArena::new(&ARENA_VAL);
        },
    };

    let model_constructor = match (arena, has_model_storage) {
        (ArenaArg::Owned, true) => quote! {
            impl #struct_ident {
                /// Creates a model with an arena owned exclusively by this instance.
                pub fn new() -> Self {
                    Self {
                        __arena: ::oneliner::runtime::OwnedArena::new(
                            #module_ident::Workspace::new(),
                        ),
                        #constructor_field
                    }
                }
            }
        },
        (ArenaArg::Owned, false) => quote! {
            impl #struct_ident {
                /// Creates a model with an arena owned exclusively by this instance.
                pub fn new() -> Self {
                    Self {
                        __arena: ::oneliner::runtime::OwnedArena::new(
                            #module_ident::Workspace::new(),
                        ),
                    }
                }
            }
        },
        (ArenaArg::Shared, true) => quote! {
            impl #struct_ident {
                /// Creates a model instance backed by the model type's shared static arena.
                pub const fn new() -> Self {
                    Self {
                        #constructor_field
                    }
                }
            }
        },
        (ArenaArg::Shared, false) => quote! {
            impl #struct_ident {
                /// Creates a model instance backed by the model type's shared static arena.
                pub const fn new() -> Self {
                    Self
                }
            }
        },
    };

    let default_impl = if derives_default(&input_struct) {
        quote! {}
    } else {
        quote! {
            impl ::core::default::Default for #struct_ident {
                fn default() -> Self {
                    Self::new()
                }
            }
        }
    };

    let execute = quote! {
        #(
            #module_ident::#execute_fns(arena, #execute_args)
                .expect(#execute_error);
        )*
    };

    let run_with_arena = match arena {
        ArenaArg::Owned => quote! {
            let arena = self.__arena.get_mut();
            #execute
        },
        ArenaArg::Shared => quote! {
            #module_ident::ARENA.with(|arena| {
                #execute
            });
        },
    };

    let run_body = quote! {
            #run_setup
            #run_with_arena
            #run_finish
    };

    let inference_impl = quote! {
        impl ::oneliner::runtime::ModelInference for #struct_ident {
            type InputTensor = ::oneliner::runtime::Tensor<
                #input_type,
                #input_d0,
                #input_d1,
                #input_d2,
                #input_d3,
            >;
            type OutputTensor = ::oneliner::runtime::Tensor<
                #output_type,
                #output_d0,
                #output_d1,
                #output_d2,
                #output_d3,
            >;

            fn create_input_tensor() -> Self::InputTensor {
                Self::InputTensor::new(0 as #input_type)
            }

            fn run(&mut self, input: &Self::InputTensor) -> Self::OutputTensor {
                #run_body
            }
        }
    };

    quote! {
        #model_definition

        #[allow(improper_ctypes, non_camel_case_types, non_snake_case, non_upper_case_globals)]
        #[link(name = #object_path, kind = "static", modifiers = "+verbatim")]
        unsafe extern "C" {}

        #[allow(
            dead_code,
            improper_ctypes,
            mutable_transmutes,
            non_camel_case_types,
            non_snake_case,
            non_upper_case_globals,
            unused_imports,
            unused_macros,
            unused_mut,
            unused_variables
        )]


        mod #module_ident {

            use ::oneliner::runtime::{
                concurrent, dispatch_fn_from_library, fill, try_dispatch, Access, Aligned,
                AlignedType, AnyBufferRange, Buffer, BufferMut, BufferSource, Error,
                iree_hal_executable_environment_v0_t, iree_hal_executable_library_header_t,
                iree_hal_executable_library_query_fn_t,
            };


            unsafe extern "C" {
                #[link_name = #query_link_name]
                pub unsafe fn #query_fn(
                    max_version: u32,
                    environment: *const iree_hal_executable_environment_v0_t,
                ) -> *const *const iree_hal_executable_library_header_t;
            }

            static QUERY_FN_PTR: iree_hal_executable_library_query_fn_t = #query_fn;

            #module_items

            include!(#flow_rs);

            #shared_arena_static
        }

        #model_constructor
        #default_impl

        impl ::oneliner::runtime::ModelSource for #struct_ident {
            const MODEL_PATH: &'static str = #model_path;
            const ARTIFACTS: ::oneliner::runtime::ModelArtifacts = ::oneliner::runtime::ModelArtifacts {
                backend: "iree",
                expansion: "static-flow",
                model_path: #model_path,
                compile_input_path: #compile_input_path,
                object_path: #object_path,
                link_path: #object_path,
                ir_path: #ir_path,
                flow_rs_path: #flow_rs,
                metadata_json_path: #metadata_json_path,
                input_size: #input_size,
                output_size: #output_size,
                io_pool_size: #io_pool_size,
                input_offset: #input_offset,
                output_offset: #output_offset,
                params_size: #params_size,
                code_size: #code_size,
                rodata_size: #rodata_size,
                total_flash_size: #total_flash_size,
                ram_size: #ram_size,
                stack_size: #stack_size,
                total_ram_size: #total_ram_size,
            };
        }

        #inference_impl
    }
}

fn standard_fragments(output_type: &TokenStream) -> IoCodegen {
    IoCodegen {
        has_model_storage: false,
        model_field: quote! {},
        constructor_field: quote! {},
        execute_args: quote! { input_buffer, output_buffer },
        execute_error: "Oneliner inference dispatch failed",
        run_setup: quote! {
            let input_buffer = ::oneliner::runtime::Buffer::new(
                input.as_ptr().cast::<u8>(),
                input.byte_len(),
            );
            let mut output = Self::OutputTensor::new(0 as #output_type);
            let output_buffer = ::oneliner::runtime::BufferMut::new(
                output.as_mut_ptr().cast::<u8>(),
                output.byte_len(),
            );
        },
        run_finish: quote! { output },
        module_items: quote! {},
        storage_size: 0,
        input_offset: 0,
        output_offset: 0,
    }
}

fn derives_default(input_struct: &ItemStruct) -> bool {
    input_struct.attrs.iter().any(|attribute| {
        if !attribute.path.is_ident("derive") {
            return false;
        }

        let Ok(Meta::List(derive)) = attribute.parse_meta() else {
            return false;
        };

        derive.nested.iter().any(|item| {
            let NestedMeta::Meta(Meta::Path(path)) = item else {
                return false;
            };

            path.segments
                .last()
                .is_some_and(|segment| segment.ident == "Default")
        })
    })
}
