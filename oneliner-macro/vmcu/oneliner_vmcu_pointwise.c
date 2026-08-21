#include <stddef.h>
#include <stdint.h>

#define CONST_TENSOR(type, base, byte_offset)                                  \
  ((const type *)((const uint8_t *)(base) + (byte_offset)))
#define TENSOR(type, base, byte_offset)                                        \
  ((type *)((uint8_t *)(base) + (byte_offset)))

static int8_t saturate_i8(int32_t value) {
  if (value < -128) {
    return -128;
  }
  if (value > 127) {
    return 127;
  }
  return (int8_t)value;
}

// Config: rows, input channels, intermediate channels, output channels.
// The intermediate activation is one row-sized segment instead of a complete
// rows-by-intermediate-channels tensor.
void oneliner_vmcu_pointwise_pair_s8(
    const int8_t *input_base, size_t input_offset,
    const int8_t *weight0_base, size_t weight0_offset,
    const int8_t *weight1_base, size_t weight1_offset,
    const int32_t *config_base, size_t config_offset, int8_t *output_base,
    size_t output_offset, int8_t *segment_base, size_t segment_offset) {
  const int8_t *input = CONST_TENSOR(int8_t, input_base, input_offset);
  const int8_t *weight0 = CONST_TENSOR(int8_t, weight0_base, weight0_offset);
  const int8_t *weight1 = CONST_TENSOR(int8_t, weight1_base, weight1_offset);
  int8_t *segment = TENSOR(int8_t, segment_base, segment_offset);
  const int32_t *config =
      CONST_TENSOR(int32_t, config_base, config_offset);
  int8_t *output = TENSOR(int8_t, output_base, output_offset);
  const size_t rows = (size_t)config[0];
  const size_t input_channels = (size_t)config[1];
  const size_t intermediate_channels = (size_t)config[2];
  const size_t output_channels = (size_t)config[3];

  for (size_t row = 0; row < rows; ++row) {
    for (size_t mid = 0; mid < intermediate_channels; ++mid) {
      int32_t accumulator = 0;
      for (size_t in = 0; in < input_channels; ++in) {
        accumulator += input[row * input_channels + in] *
                       weight0[in * intermediate_channels + mid];
      }
      segment[mid] = saturate_i8(accumulator);
    }

    for (size_t out = 0; out < output_channels; ++out) {
      int32_t accumulator = 0;
      for (size_t mid = 0; mid < intermediate_channels; ++mid) {
        accumulator += segment[mid] * weight1[mid * output_channels + out];
      }
      output[row * output_channels + out] = saturate_i8(accumulator);
    }
  }
}
