/* Fully connected (GEMM) segment kernel. */
#include "oneliner_vmcu_common.h"

/* Config layout: rows at [2], input columns at [3], output columns at [4];
 * zero points at [20]/[21]; scratch at [35]; magic at [36]. */
#define VMCU_FLAG_FC_MATRIX 1

static int oneliner_vmcu_fc_kernel(const int8_t *input, const int8_t *weight,
                     const int32_t *bias, const int32_t *multiplier,
                     const int8_t *shift, const int32_t *config,
                     int8_t *output, int8_t *scratch) {
  size_t m;
  size_t rows;
  size_t input_columns;
  size_t output_columns;
  size_t input_count;
  size_t output_count;
  size_t weight_count;
  int32_t input_zero_point;
  int32_t output_zero_point;

  if (input == NULL || weight == NULL || bias == NULL || multiplier == NULL ||
      shift == NULL || config == NULL || output == NULL || scratch == NULL) {
    return 0;
  }
  if (((uintptr_t)bias % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)multiplier % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)config % _Alignof(int32_t)) != 0U) {
    return 0;
  }
  if (config[0] != 1 || config[36] != VMCU_CONFIG_MAGIC ||
      (config[1] & ~VMCU_FLAG_FC_MATRIX) != 0) {
    return 0;
  }
  rows = (size_t)config[2];
  input_columns = (size_t)config[3];
  output_columns = (size_t)config[4];
  input_zero_point = config[20];
  output_zero_point = config[21];
  if (rows <= 0U || input_columns <= 0U || output_columns <= 0U ||
      !valid_zero_point(input_zero_point) ||
      !valid_zero_point(output_zero_point) || config[35] < 0) {
    return 0;
  }
  for (m = 5U; m <= 19U; ++m) {
    if (config[m] != 0) {
      return 0;
    }
  }
  for (m = 22U; m <= 34U; ++m) {
    if (config[m] != 0) {
      return 0;
    }
  }
  if (!checked_mul_size(rows, input_columns, &input_count) ||
      !checked_mul_size(rows, output_columns, &output_count) ||
      !checked_mul_size(output_columns, input_columns, &weight_count)) {
    return 0;
  }
  if ((size_t)config[35] < output_columns) {
    return 0;
  }
  for (m = 0U; m < output_columns; ++m) {
    if (!valid_shift(shift[m])) {
      return 0;
    }
  }

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

void oneliner_vmcu_fc_s8(VMCU_SINGLE_ARGUMENTS) {
  if (!VMCU_SINGLE_BASES_VALID) {
    return;
  }
  (void)oneliner_vmcu_fc_kernel(VMCU_SINGLE_CALL);
}
