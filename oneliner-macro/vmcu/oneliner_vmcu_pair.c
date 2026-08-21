/* Pointwise pair segment kernel: two adjacent int8 matmuls fused with a
 * row-sized intermediate segment (element offsets). */
#include "oneliner_vmcu_common.h"

static int8_t saturate_i8(int32_t value) {
  if (value < -128) {
    return -128;
  }
  if (value > 127) {
    return 127;
  }
  return (int8_t)value;
}

void oneliner_vmcu_pointwise_pair_s8(
    const int8_t *input_base, size_t input_offset,
    const int8_t *weight0_base, size_t weight0_offset,
    const int8_t *weight1_base, size_t weight1_offset,
    const int32_t *config_base, size_t config_offset, int8_t *output_base,
    size_t output_offset, int8_t *segment_base, size_t segment_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int8_t *weight0 = CONST_TENSOR(int8_t, weight0_base, weight0_offset);
  const int8_t *weight1 = CONST_TENSOR(int8_t, weight1_base, weight1_offset);
  const int32_t *config =
      CONST_TENSOR(int32_t, config_base, config_offset);
  int8_t *segment = TENSOR(int8_t, segment_base, segment_offset);
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  const size_t rows = (size_t)config[0];
  const size_t input_channels = (size_t)config[1];
  const size_t intermediate_channels = (size_t)config[2];
  const size_t output_channels = (size_t)config[3];
  size_t row;

  if (input == NULL || weight0 == NULL || weight1 == NULL || config == NULL ||
      output == NULL || segment == NULL) {
    return;
  }

  for (row = 0U; row < rows; ++row) {
    size_t mid;
    for (mid = 0U; mid < intermediate_channels; ++mid) {
      int32_t accumulator = 0;
      size_t in;
      for (in = 0U; in < input_channels; ++in) {
        accumulator += (int32_t)input[row * input_channels + in] *
                       (int32_t)weight0[in * intermediate_channels + mid];
      }
      segment[mid] = saturate_i8(accumulator);
    }
    for (mid = 0U; mid < output_channels; ++mid) {
      int32_t accumulator = 0;
      size_t in;
      for (in = 0U; in < intermediate_channels; ++in) {
        accumulator += (int32_t)segment[in] *
                       (int32_t)weight1[in * output_channels + mid];
      }
      output[row * output_channels + mid] = saturate_i8(accumulator);
    }
  }
}
