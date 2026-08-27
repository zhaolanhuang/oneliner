/* Fully connected (GEMM) segment kernel.
 *
 * Same dual compilation scheme as oneliner_vmcu_conv2d.c: the generic build
 * reads rows/columns/zero points from the config tensor; a specialized build
 * (VMCU_SPECIALIZED + VMCU_ROWS/VMCU_CIN/VMCU_COUT/VMCU_INPUT_ZP/
 * VMCU_OUTPUT_ZP macros) exports VMCU_ENTRY_NAME without the config argument.
 *
 * Generic config layout: rows at [2], input columns at [3], output columns at
 * [4]; zero points at [20]/[21]; scratch at [35]; magic at [36]. */
#include "oneliner_vmcu_common.h"

#ifndef VMCU_FC_ENTRY
#define VMCU_FC_ENTRY oneliner_vmcu_fc_s8
#endif

#define VMCU_FC_IDX_ROWS 2
#define VMCU_FC_IDX_CIN 3
#define VMCU_FC_IDX_COUT 4
#define VMCU_FC_IDX_INPUT_ZP 20
#define VMCU_FC_IDX_OUTPUT_ZP 21

#ifdef VMCU_SPECIALIZED
#define VMCU_FC_DIM(field) ((size_t)VMCU_##field)
#define VMCU_FC_ZP(field) VMCU_##field
#else
#define VMCU_FC_DIM(field) ((size_t)config[VMCU_FC_IDX_##field])
#define VMCU_FC_ZP(field) config[VMCU_FC_IDX_##field]
#endif

static int oneliner_vmcu_fc_kernel(const int8_t *input, const int8_t *weight,
                     const int32_t *bias, const int32_t *multiplier,
                     const int8_t *shift,
#ifdef VMCU_SPECIALIZED
                     int8_t *output, int8_t *scratch) {
#else
                     const int32_t *config, int8_t *output,
                     int8_t *scratch) {
#endif
  size_t m;
  size_t rows;
  size_t input_columns;
  size_t output_columns;

  int32_t input_zero_point;
  int32_t output_zero_point;

  if (input == NULL || weight == NULL || bias == NULL || multiplier == NULL ||
      shift == NULL ||
#ifndef VMCU_SPECIALIZED
      config == NULL ||
#endif
      output == NULL || scratch == NULL) {
    return 0;
  }
  if (((uintptr_t)bias % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)multiplier % _Alignof(int32_t)) != 0U) {
    return 0;
  }
  rows = VMCU_FC_DIM(ROWS);
  input_columns = VMCU_FC_DIM(CIN);
  output_columns = VMCU_FC_DIM(COUT);
  input_zero_point = VMCU_FC_ZP(INPUT_ZP);
  output_zero_point = VMCU_FC_ZP(OUTPUT_ZP);

  for (m = 0U; m < rows; ++m) {
    size_t oc;
    for (oc = 0U; oc < output_columns; ++oc) {
      uint32_t accumulator = (uint32_t)bias[oc];
      size_t ic;
      for (ic = 0U; ic < input_columns; ++ic) {
        accumulator = accumulate_product(
            accumulator, input[m * input_columns + ic],
            weight[oc * input_columns + ic], input_zero_point);
      }
      output[m * output_columns + oc] = requantize_i8(
          i32_from_bits(accumulator), multiplier[oc], (uint8_t)shift[oc],
          output_zero_point);
    }
  }
  return 1;
}

#ifdef VMCU_SPECIALIZED
void VMCU_FC_ENTRY(VMCU_SINGLE_ARGUMENTS_NO_CONFIG) {
  if (!VMCU_SINGLE_BASES_VALID_NO_CONFIG) {
    return;
  }
  (void)oneliner_vmcu_fc_kernel(VMCU_SINGLE_CALL_NO_CONFIG);
}
#else
void VMCU_FC_ENTRY(VMCU_SINGLE_ARGUMENTS) {
  if (!VMCU_SINGLE_BASES_VALID) {
    return;
  }
  (void)oneliner_vmcu_fc_kernel(VMCU_SINGLE_CALL);
}
#endif