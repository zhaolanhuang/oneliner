use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use super::super::codegen::IoCodegen;
use super::super::IoView;

pub(crate) fn compact_fragments(
    storage_size: usize,
    input: IoView,
    output: IoView,
    output_type: &TokenStream,
    module_ident: &Ident,
) -> IoCodegen {
    let input_offset = input.offset;
    let input_size = input.size;
    let output_offset = output.offset;
    let output_size = output.size;
    IoCodegen {
        has_model_storage: true,
        model_field: quote! {
            __io_buffer: #module_ident::IoBuffer<#storage_size>,
        },
        constructor_field: quote! {
            __io_buffer: #module_ident::IoBuffer::new(0),
        },
        execute_args: quote! { inout_buffer },
        execute_error: "Oneliner in-place inference dispatch failed",
        run_setup: quote! {
            let input_bytes = unsafe {
                ::core::slice::from_raw_parts(
                    input.as_ptr().cast::<u8>(),
                    input.byte_len(),
                )
            };
            self.__io_buffer.as_bytes_mut()[
                #input_offset..#input_offset + #input_size
            ].copy_from_slice(input_bytes);
            let inout_buffer = ::oneliner::runtime::BufferMut::new(
                self.__io_buffer.as_mut_ptr(),
                self.__io_buffer.len(),
            );
        },
        run_finish: quote! {
            let mut output = Self::OutputTensor::new(0 as #output_type);
            let output_bytes = unsafe {
                ::core::slice::from_raw_parts_mut(
                    output.as_mut_ptr().cast::<u8>(),
                    output.byte_len(),
                )
            };
            output_bytes.copy_from_slice(&self.__io_buffer.as_bytes()[
                #output_offset..#output_offset + #output_size
            ]);
            output
        },
        module_items: quote! {
            pub(super) struct IoBuffer<const N: usize> {
                storage: Aligned<AlignedType, [u8; N]>,
            }

            impl<const N: usize> IoBuffer<N> {
                pub(super) const fn new(value: u8) -> Self {
                    Self {
                        storage: Aligned([value; N]),
                    }
                }

                pub(super) fn as_bytes(&self) -> &[u8] {
                    &self.storage[..]
                }

                pub(super) fn as_bytes_mut(&mut self) -> &mut [u8] {
                    &mut self.storage[..]
                }

                pub(super) fn as_mut_ptr(&mut self) -> *mut u8 {
                    self.as_bytes_mut().as_mut_ptr()
                }

                pub(super) const fn len(&self) -> usize {
                    N
                }
            }

            impl<const N: usize> Default for IoBuffer<N> {
                fn default() -> Self {
                    Self::new(0)
                }
            }
        },
        storage_size,
        input_offset,
        output_offset,
    }
}
