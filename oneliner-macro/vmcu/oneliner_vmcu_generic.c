/* Generic vMCU ukernel library: inverted bottleneck, pointwise pair,
 * single 2D convolution, and fully connected segment kernels.
 *
 * This source is compiled to LLVM bitcode and linked into IREE executables
 * as a ukernel library. Each exported function receives IREE element offsets
 * for every tensor; the CONST_TENSOR/TENSOR macros apply them as typed
 * pointer arithmetic.
 */

#include "oneliner_vmcu_mcunet.c"

/* Config layout shared by conv2d and fc (37 int32 entries):
 *   [0] version, [1] flags, [2] batch, [3] input height, [4] input width,
 *   [5] input channels, [6] output height, [7] output width, [8] output
 *   channels, [9] kernel height, [10] kernel width, [11] stride height,
 *   [12] stride width, [13] dilation height, [14] dilation width,
 *   [15] pad top, [16] pad left, [17] pad bottom, [18] pad right,
 *   [19] input zero point, [20] output zero point, [21..34] reserved,
 *   [35] scratch bytes, [36] VMCU_CONFIG_MAGIC.
 */
#define VMCU_FLAG_FC_MATRIX 1

static int8_t saturate_i8(int32_t value) {
  if (value < -128) {
    return -128;
  }
  if (value > 127) {
    return 127;
  }
  return (int8_t)value;
}

static int conv2d_kernel(const int8_t *input, const int8_t *weight,
                         const int32_t *bias, const int32_t *multiplier,
                         const int8_t *shift, const int32_t *config,
                         int8_t *output, int8_t *scratch) {
  size_t n;
  size_t input_count;
  size_t output_count;
  size_t weight_count;
  size_t cache_bytes;
  size_t required_scratch;
  size_t effective_h;
  size_t effective_w;
  size_t padded_h;
  size_t padded_w;
  size_t computed_oh;
  size_t computed_ow;
  size_t tmp;
  size_t batches;
  size_t input_height;
  size_t input_width;
  size_t input_channels;
  size_t output_height;
  size_t output_width;
  size_t output_channels;
  size_t kernel_height;
  size_t kernel_width;
  size_t stride_height;
  size_t stride_width;
  size_t dilation_height;
  size_t dilation_width;
  size_t pad_top;
  size_t pad_left;
  size_t pad_bottom;
  size_t pad_right;
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
  if (config[0] != 1 || config[36] != VMCU_CONFIG_MAGIC || config[1] != 0) {
    return 0;
  }
  for (n = 2U; n <= 14U; ++n) {
    if (config[n] <= 0) {
      return 0;
    }
  }
  for (n = 15U; n <= 18U; ++n) {
    if (config[n] < 0) {
      return 0;
    }
  }
  for (n = 21U; n <= 34U; ++n) {
    if (config[n] != 0) {
      return 0;
    }
  }

  batches = (size_t)config[2];
  input_height = (size_t)config[3];
  input_width = (size_t)config[4];
  input_channels = (size_t)config[5];
  output_height = (size_t)config[6];
  output_width = (size_t)config[7];
  output_channels = (size_t)config[8];
  kernel_height = (size_t)config[9];
  kernel_width = (size_t)config[10];
  stride_height = (size_t)config[11];
  stride_width = (size_t)config[12];
  dilation_height = (size_t)config[13];
  dilation_width = (size_t)config[14];
  pad_top = (size_t)config[15];
  pad_left = (size_t)config[16];
  pad_bottom = (size_t)config[17];
  pad_right = (size_t)config[18];
  input_zero_point = config[19];
  output_zero_point = config[20];

  if (!valid_zero_point(input_zero_point) ||
      !valid_zero_point(output_zero_point) || config[35] < 0) {
    return 0;
  }
  if (!checked_mul_size(kernel_height - 1U, dilation_height, &effective_h) ||
      !checked_add_size(effective_h, 1U, &effective_h) ||
      !checked_mul_size(kernel_width - 1U, dilation_width, &effective_w) ||
      !checked_add_size(effective_w, 1U, &effective_w) ||
      !checked_add_size(input_height, pad_top, &padded_h) ||
      !checked_add_size(padded_h, pad_bottom, &padded_h) ||
      !checked_add_size(input_width, pad_left, &padded_w) ||
      !checked_add_size(padded_w, pad_right, &padded_w) ||
      padded_h < effective_h || padded_w < effective_w) {
    return 0;
  }
  computed_oh = (padded_h - effective_h) / stride_height + 1U;
  computed_ow = (padded_w - effective_w) / stride_width + 1U;
  if (computed_oh != output_height || computed_ow != output_width) {
    return 0;
  }

  if (!checked_mul_size(batches, input_height, &tmp) ||
      !checked_mul_size(tmp, input_width, &tmp) ||
      !checked_mul_size(tmp, input_channels, &input_count) ||
      !checked_mul_size(batches, output_height, &tmp) ||
      !checked_mul_size(tmp, output_width, &tmp) ||
      !checked_mul_size(tmp, output_channels, &output_count) ||
      !checked_mul_size(kernel_height, kernel_width, &tmp) ||
      !checked_mul_size(tmp, input_channels, &tmp) ||
      !checked_mul_size(tmp, output_channels, &weight_count) ||
      !checked_mul_size(kernel_height, input_width, &tmp) ||
      !checked_mul_size(tmp, input_channels, &cache_bytes)) {
    return 0;
  }
  if (!checked_add_size(cache_bytes, 0U, &required_scratch) ||
      required_scratch > (size_t)INT32_MAX ||
      (size_t)config[35] < required_scratch) {
    return 0;
  }
  for (n = 0U; n < output_channels; ++n) {
    if (!valid_shift(shift[n])) {
      return 0;
    }
  }

  for (n = 0U; n < batches; ++n) {
    int8_t *const cache = scratch;
    size_t ring_start = 0U;
    size_t oy;
    const int8_t *batch_input = input + n * input_count / batches;
    int8_t *batch_output = output + n * output_count / batches;

    for (oy = 0U; oy < output_height; ++oy) {
      size_t first_new_ky = 0U;
      size_t last_new_ky = kernel_height;
      size_t ky;
      size_t ox;
      size_t ring_index;

      if (oy != 0U && stride_height % dilation_height == 0U) {
        const size_t rotation = stride_height / dilation_height;
        if (rotation < kernel_height) {
          ring_start = (ring_start + rotation) % kernel_height;
          first_new_ky = kernel_height - rotation;
        } else {
          ring_start = 0U;
        }
      } else if (oy != 0U) {
        ring_start = 0U;
      }

      for (ky = first_new_ky; ky < last_new_ky; ++ky) {
        const size_t padded_y = oy * stride_height + ky * dilation_height;
        size_t iy;
        size_t ring_index = ring_start + ky;
        if (padded_y < pad_top ||
            (iy = padded_y - pad_top) >= input_height) {
          continue;
        }
        if (ring_index >= kernel_height) {
          ring_index -= kernel_height;
        }
        copy_bytes(cache + ring_index * input_width * input_channels,
                   batch_input + iy * input_width * input_channels,
                   input_width * input_channels);
      }

      for (ox = 0U; ox < output_width; ++ox) {
        size_t oc;
        for (oc = 0U; oc < output_channels; ++oc) {
          uint32_t accumulator = (uint32_t)bias[oc];
          for (ky = 0U; ky < kernel_height; ++ky) {
            const size_t padded_y = oy * stride_height + ky * dilation_height;
            size_t iy;
            size_t kx;
            const int8_t *cache_row;
            if (padded_y < pad_top ||
                (iy = padded_y - pad_top) >= input_height) {
              continue;
            }
            (void)iy;
            ring_index = ring_start + ky;
            if (ring_index >= kernel_height) {
              ring_index -= kernel_height;
            }
            cache_row = cache + ring_index * input_width * input_channels;
            for (kx = 0U; kx < kernel_width; ++kx) {
              const size_t padded_x = ox * stride_width + kx * dilation_width;
              size_t ix;
              size_t ic;
              if (padded_x < pad_left ||
                  (ix = padded_x - pad_left) >= input_width) {
                continue;
              }
              for (ic = 0U; ic < input_channels; ++ic) {
                accumulator = accumulate_product(
                    accumulator, cache_row[ix * input_channels + ic],
                    weight[((ky * kernel_width + kx) * input_channels + ic) *
                               output_channels +
                           oc],
                    input_zero_point);
              }
            }
          }
          batch_output[(oy * output_width + ox) * output_channels + oc] =
              requantize_i8(i32_from_bits(accumulator), multiplier[oc],
                            (uint8_t)shift[oc], output_zero_point);
        }
      }
    }
  }
  return 1;
}

static int fc_kernel(const int8_t *input, const int8_t *weight,
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

#define VMCU_GENERIC_ARGUMENTS                                               \
  const int8_t *input_base, size_t input_offset,                             \
      const int8_t *weight_base, size_t weight_offset,                       \
      const int32_t *bias_base, size_t bias_offset,                          \
      const int32_t *multiplier_base, size_t multiplier_offset,              \
      const int8_t *shift_base, size_t shift_offset,                         \
      const int32_t *config_base, size_t config_offset,                      \
      int8_t *output_base, size_t output_offset, int8_t *scratch_base,       \
      size_t scratch_offset

#define VMCU_GENERIC_CALL                                                     \
  CONST_TENSOR(int8_t, input_base, input_offset),                             \
      CONST_TENSOR(int8_t, weight_base, weight_offset),                       \
      CONST_TENSOR(int32_t, bias_base, bias_offset),                          \
      CONST_TENSOR(int32_t, multiplier_base, multiplier_offset),              \
      CONST_TENSOR(int8_t, shift_base, shift_offset),                         \
      CONST_TENSOR(int32_t, config_base, config_offset),                      \
      TENSOR(int8_t, output_base, output_offset),                             \
      TENSOR(int8_t, scratch_base, scratch_offset)

#define VMCU_GENERIC_BASES_VALID                                              \
  (input_base != NULL && weight_base != NULL && bias_base != NULL &&          \
   multiplier_base != NULL && shift_base != NULL && config_base != NULL &&    \
   output_base != NULL && scratch_base != NULL)

void oneliner_vmcu_conv2d_s8(VMCU_GENERIC_ARGUMENTS) {
  if (!VMCU_GENERIC_BASES_VALID) {
    return;
  }
  (void)conv2d_kernel(VMCU_GENERIC_CALL);
}

void oneliner_vmcu_fc_s8(VMCU_GENERIC_ARGUMENTS) {
  if (!VMCU_GENERIC_BASES_VALID) {
    return;
  }
  (void)fc_kernel(VMCU_GENERIC_CALL);
}

/* Pointwise pair kernel (element offsets). */
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
      output == NULL || segment == NULL || rows == 0U ||
      input_channels == 0U || intermediate_channels == 0U ||
      output_channels == 0U) {
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