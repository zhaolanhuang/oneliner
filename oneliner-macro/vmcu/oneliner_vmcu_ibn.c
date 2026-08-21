/* Inverted bottleneck (pointwise-depthwise-pointwise + optional residual)
 * segment kernel, from the vMCU paper's multi-layer pattern. */
#include "oneliner_vmcu_common.h"

static void oneliner_vmcu_ibn_kernel(
    const int8_t *input, const int8_t *w_expand,
    const int8_t *w_depthwise, const int8_t *w_project,
    const int32_t *b_expand, const int32_t *b_depthwise,
    const int32_t *b_project, const int32_t *m_expand,
    const int32_t *m_depthwise, const int32_t *m_project,
    const int8_t *s_expand, const int8_t *s_depthwise,
    const int8_t *s_project, const int32_t *config, int8_t *output,
    int8_t *scratch, size_t input_offset, size_t w_expand_offset,
    size_t w_depthwise_offset, size_t w_project_offset,
    size_t b_expand_offset, size_t b_depthwise_offset,
    size_t b_project_offset, size_t m_expand_offset,
    size_t m_depthwise_offset, size_t m_project_offset,
    size_t s_expand_offset, size_t s_depthwise_offset,
    size_t s_project_offset, size_t config_offset, size_t output_offset,
    size_t scratch_offset) {
  size_t n;
  size_t input_count;
  size_t output_count;
  size_t expand_weight_count;
  size_t depthwise_weight_count;
  size_t project_weight_count;
  size_t expanded_i32_bytes;
  size_t output_i32_bytes;
  size_t cache_bytes;
  size_t depth_row_bytes;
  size_t delay_row_bytes;
  size_t delay_bytes = 0U;
  size_t required_scratch;
  size_t effective_h;
  size_t effective_w;
  size_t padded_h;
  size_t padded_w;
  size_t computed_oh;
  size_t computed_ow;
  size_t tmp;
  size_t delay_rows;
  int flags;
  int residual;
  int alias;
  size_t batches;
  size_t input_height;
  size_t input_width;
  size_t input_channels;
  size_t output_height;
  size_t output_width;
  size_t expanded_channels;
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
  int32_t expansion_zero_point;
  int32_t depthwise_zero_point;
  int32_t projection_zero_point;
  int32_t output_zero_point;

  if (input == NULL || w_expand == NULL || w_depthwise == NULL ||
      w_project == NULL || b_expand == NULL || b_depthwise == NULL ||
      b_project == NULL || m_expand == NULL || m_depthwise == NULL ||
      m_project == NULL || s_expand == NULL || s_depthwise == NULL ||
      s_project == NULL || config == NULL || output == NULL ||
      scratch == NULL) {
    return;
  }
  if (((uintptr_t)b_expand % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)b_depthwise % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)b_project % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)m_expand % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)m_depthwise % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)m_project % _Alignof(int32_t)) != 0U ||
      ((uintptr_t)config % _Alignof(int32_t)) != 0U) {
    return;
  }
  if (config[0] != 1 || config[36] != VMCU_CONFIG_MAGIC || config[1] < 0 ||
      (config[1] & ~(VMCU_FLAG_RESIDUAL | VMCU_FLAG_IN_PLACE)) != 0) {
    return;
  }
  for (n = 2U; n <= 19U; ++n) {
    if (config[n] <= 0 && n <= 15U) {
      return;
    }
    if (config[n] < 0) {
      return;
    }
  }

  flags = config[1];
  residual = (flags & VMCU_FLAG_RESIDUAL) != 0;
  batches = (size_t)config[2];
  input_height = (size_t)config[3];
  input_width = (size_t)config[4];
  input_channels = (size_t)config[5];
  output_height = (size_t)config[6];
  output_width = (size_t)config[7];
  expanded_channels = (size_t)config[8];
  output_channels = (size_t)config[9];
  kernel_height = (size_t)config[10];
  kernel_width = (size_t)config[11];
  stride_height = (size_t)config[12];
  stride_width = (size_t)config[13];
  dilation_height = (size_t)config[14];
  dilation_width = (size_t)config[15];
  pad_top = (size_t)config[16];
  pad_left = (size_t)config[17];
  pad_bottom = (size_t)config[18];
  pad_right = (size_t)config[19];
  input_zero_point = config[20];
  expansion_zero_point = config[21];
  depthwise_zero_point = config[22];
  projection_zero_point = config[23];
  output_zero_point = config[24];
  delay_rows = pad_bottom + 1U;

  if (!valid_zero_point(input_zero_point) ||
      !valid_zero_point(expansion_zero_point) ||
      !valid_zero_point(depthwise_zero_point) ||
      !valid_zero_point(projection_zero_point) ||
      !valid_zero_point(output_zero_point) || config[35] < 0) {
    return;
  }
  if (residual &&
      (config[26] < 1 || config[26] > 62 || config[30] < 1 ||
       config[30] > 62 || config[34] < 1 || config[34] > 62 ||
       (config[27] != 0 && (config[28] < 1 || config[28] > 62)) ||
       (config[31] != 0 && (config[32] < 1 || config[32] > 62)))) {
    return;
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
    return;
  }
  computed_oh = (padded_h - effective_h) / stride_height + 1U;
  computed_ow = (padded_w - effective_w) / stride_width + 1U;
  if (computed_oh != output_height || computed_ow != output_width) {
    return;
  }
  if (residual && (output_height != input_height ||
                   output_width != input_width ||
                   output_channels != input_channels)) {
    return;
  }

  if (!checked_mul_size(batches, input_height, &tmp) ||
      !checked_mul_size(tmp, input_width, &tmp) ||
      !checked_mul_size(tmp, input_channels, &input_count) ||
      !checked_mul_size(batches, output_height, &tmp) ||
      !checked_mul_size(tmp, output_width, &tmp) ||
      !checked_mul_size(tmp, output_channels, &output_count) ||
      !checked_mul_size(expanded_channels, input_channels,
                        &expand_weight_count) ||
      !checked_mul_size(kernel_height, kernel_width, &tmp) ||
      !checked_mul_size(tmp, expanded_channels, &depthwise_weight_count) ||
      !checked_mul_size(output_channels, expanded_channels,
                        &project_weight_count) ||
      !checked_mul_size(kernel_height, input_width, &tmp) ||
      !checked_mul_size(tmp, expanded_channels, &cache_bytes) ||
      !checked_mul_size(output_width, expanded_channels, &depth_row_bytes) ||
      !checked_mul_size(output_width, output_channels, &delay_row_bytes) ||
      !checked_mul_size(expanded_channels, sizeof(int32_t),
                        &expanded_i32_bytes) ||
      !checked_mul_size(output_channels, sizeof(int32_t),
                        &output_i32_bytes)) {
    return;
  }
  if ((flags & VMCU_FLAG_IN_PLACE) != 0) {
    if (!checked_mul_size(delay_rows, delay_row_bytes, &delay_bytes)) {
      return;
    }
  }
  if (!checked_add_size(cache_bytes, depth_row_bytes, &required_scratch) ||
      !checked_add_size(required_scratch, delay_bytes, &required_scratch) ||
      required_scratch > (size_t)INT32_MAX ||
      (size_t)config[35] < required_scratch) {
    return;
  }

  if (input_offset > SIZE_MAX - input_count ||
      output_offset > SIZE_MAX - output_count ||
      w_expand_offset > SIZE_MAX - expand_weight_count ||
      w_depthwise_offset > SIZE_MAX - depthwise_weight_count ||
      w_project_offset > SIZE_MAX - project_weight_count ||
      scratch_offset > SIZE_MAX - required_scratch ||
      b_expand_offset > SIZE_MAX - expanded_i32_bytes ||
      b_depthwise_offset > SIZE_MAX - expanded_i32_bytes ||
      b_project_offset > SIZE_MAX - output_i32_bytes ||
      m_expand_offset > SIZE_MAX - expanded_i32_bytes ||
      m_depthwise_offset > SIZE_MAX - expanded_i32_bytes ||
      m_project_offset > SIZE_MAX - output_i32_bytes ||
      s_expand_offset > SIZE_MAX - expanded_channels ||
      s_depthwise_offset > SIZE_MAX - expanded_channels ||
      s_project_offset > SIZE_MAX - output_channels ||
      config_offset > SIZE_MAX - 37U) {
    return;
  }

  alias = input == output;
  if (alias && ((flags & VMCU_FLAG_IN_PLACE) == 0 ||
                output_height != input_height ||
                output_width != input_width ||
                output_channels != input_channels)) {
    return;
  }
  for (n = 0U; n < expanded_channels; ++n) {
    if (!valid_shift(s_expand[n]) || !valid_shift(s_depthwise[n])) {
      return;
    }
  }
  for (n = 0U; n < output_channels; ++n) {
    if (!valid_shift(s_project[n])) {
      return;
    }
  }

  for (n = 0U; n < batches; ++n) {
    int8_t *const expansion_cache = scratch;
    int8_t *const depthwise_row = scratch + cache_bytes;
    int8_t *const delayed_rows = depthwise_row + depth_row_bytes;
    size_t ring_start = 0U;
    size_t step;

    for (step = 0U; step < output_height; ++step) {
      const size_t oy = alias ? output_height - 1U - step : step;
      size_t first_new_ky = 0U;
      size_t last_new_ky = kernel_height;
      size_t ky;
      size_t ox;
      size_t ring_index;
      int8_t *projected_row;

      if (alias && step >= delay_rows) {
        const size_t old_step = step - delay_rows;
        const size_t old_y = output_height - 1U - old_step;
        copy_bytes(output + ((n * output_height + old_y) * output_width) *
                                output_channels,
                   delayed_rows + (step % delay_rows) * delay_row_bytes,
                   delay_row_bytes);
      }

      if (step != 0U && stride_height % dilation_height == 0U) {
        const size_t rotation = stride_height / dilation_height;
        if (rotation < kernel_height) {
          if (alias) {
            ring_start += kernel_height - rotation;
            if (ring_start >= kernel_height) {
              ring_start -= kernel_height;
            }
            last_new_ky = rotation;
          } else {
            ring_start += rotation;
            if (ring_start >= kernel_height) {
              ring_start -= kernel_height;
            }
            first_new_ky = kernel_height - rotation;
          }
        } else {
          ring_start = 0U;
        }
      } else if (step != 0U) {
        ring_start = 0U;
      }

      for (ky = first_new_ky; ky < last_new_ky; ++ky) {
        const size_t padded_y = oy * stride_height + ky * dilation_height;
        size_t iy;
        size_t ix;
        size_t ring_index = ring_start + ky;
        int8_t *cache_row;
        if (ring_index >= kernel_height) {
          ring_index -= kernel_height;
        }
        cache_row = expansion_cache + ring_index * input_width *
                                          expanded_channels;
        if (padded_y < pad_top ||
            (iy = padded_y - pad_top) >= input_height) {
          continue;
        }
        for (ix = 0U; ix < input_width; ++ix) {
          size_t ec;
          for (ec = 0U; ec < expanded_channels; ++ec) {
            uint32_t accumulator = (uint32_t)b_expand[ec];
            size_t ic;
            for (ic = 0U; ic < input_channels; ++ic) {
              const int8_t activation =
                  input[((n * input_height + iy) * input_width + ix) *
                            input_channels +
                        ic];
              accumulator = accumulate_product(
                  accumulator, activation,
                  w_expand[ec * input_channels + ic], input_zero_point);
            }
            cache_row[ix * expanded_channels + ec] = requantize_i8(
                i32_from_bits(accumulator), m_expand[ec],
                (uint8_t)s_expand[ec], expansion_zero_point);
          }
        }
      }

      for (ox = 0U; ox < output_width; ++ox) {
        size_t ec;
        for (ec = 0U; ec < expanded_channels; ++ec) {
          uint32_t accumulator = (uint32_t)b_depthwise[ec];
          for (ky = 0U; ky < kernel_height; ++ky) {
            const size_t padded_y =
                oy * stride_height + ky * dilation_height;
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
            cache_row = expansion_cache + ring_index * input_width *
                                              expanded_channels;
            for (kx = 0U; kx < kernel_width; ++kx) {
              const size_t padded_x =
                  ox * stride_width + kx * dilation_width;
              size_t ix;
              if (padded_x < pad_left ||
                  (ix = padded_x - pad_left) >= input_width) {
                continue;
              }
              accumulator = accumulate_product(
                  accumulator, cache_row[ix * expanded_channels + ec],
                  w_depthwise[(ky * kernel_width + kx) * expanded_channels +
                              ec],
                  expansion_zero_point);
            }
          }
          depthwise_row[ox * expanded_channels + ec] = requantize_i8(
              i32_from_bits(accumulator), m_depthwise[ec],
              (uint8_t)s_depthwise[ec], depthwise_zero_point);
        }
      }

      projected_row =
          alias
              ? delayed_rows + (step % delay_rows) * delay_row_bytes
              : output + ((n * output_height + oy) * output_width) *
                             output_channels;
      for (ox = 0U; ox < output_width; ++ox) {
        size_t oc;
        for (oc = 0U; oc < output_channels; ++oc) {
          uint32_t accumulator = (uint32_t)b_project[oc];
          size_t ec;
          int8_t projected;
          for (ec = 0U; ec < expanded_channels; ++ec) {
            accumulator = accumulate_product(
                accumulator, depthwise_row[ox * expanded_channels + ec],
                w_project[oc * expanded_channels + ec],
                depthwise_zero_point);
          }
          projected = requantize_i8(
              i32_from_bits(accumulator), m_project[oc],
              (uint8_t)s_project[oc], projection_zero_point);
          if (residual) {
            int32_t new_value =
                (int32_t)projected - projection_zero_point;
            int32_t skip_value =
                (int32_t)input[((n * input_height + oy) * input_width + ox) *
                                   input_channels +
                               oc] -
                input_zero_point;
            new_value = scale_single(new_value, config[25],
                                     (uint8_t)config[26]);
            if (config[27] != 0) {
              new_value = scale_double(new_value, config[27],
                                       (uint8_t)config[28]);
            }
            skip_value = scale_single(skip_value, config[29],
                                      (uint8_t)config[30]);
            if (config[31] != 0) {
              skip_value = scale_double(skip_value, config[31],
                                        (uint8_t)config[32]);
            }
            projected = requantize_i8(
                add_wrap_i32(new_value, skip_value), config[33],
                (uint8_t)config[34], output_zero_point);
          }
          projected_row[ox * output_channels + oc] = projected;
        }
      }
    }

    if (alias) {
      const size_t first_queued =
          output_height > delay_rows ? output_height - delay_rows : 0U;
      size_t queued_step;
      for (queued_step = first_queued; queued_step < output_height;
           ++queued_step) {
        const size_t oy = output_height - 1U - queued_step;
        copy_bytes(output + ((n * output_height + oy) * output_width) *
                                output_channels,
                   delayed_rows + (queued_step % delay_rows) * delay_row_bytes,
                   delay_row_bytes);
      }
    }
  }
}

#define VMCU_IBN_ARGUMENTS                                                     \
  const int8_t *input_base, size_t input_offset,                               \
      const int8_t *w_expand_base, size_t w_expand_offset,                     \
      const int8_t *w_depthwise_base, size_t w_depthwise_offset,               \
      const int8_t *w_project_base, size_t w_project_offset,                   \
      const int32_t *b_expand_base, size_t b_expand_offset,                    \
      const int32_t *b_depthwise_base, size_t b_depthwise_offset,              \
      const int32_t *b_project_base, size_t b_project_offset,                  \
      const int32_t *m_expand_base, size_t m_expand_offset,                    \
      const int32_t *m_depthwise_base, size_t m_depthwise_offset,              \
      const int32_t *m_project_base, size_t m_project_offset,                  \
      const int8_t *s_expand_base, size_t s_expand_offset,                     \
      const int8_t *s_depthwise_base, size_t s_depthwise_offset,               \
      const int8_t *s_project_base, size_t s_project_offset,                   \
      const int32_t *config_base, size_t config_offset, int8_t *output_base,   \
      size_t output_offset, int8_t *scratch_base, size_t scratch_offset

#define VMCU_IBN_CALL                                                          \
  oneliner_vmcu_ibn_kernel(                                                    \
      CONST_TENSOR(int8_t, input_base, input_offset),                          \
      CONST_TENSOR(int8_t, w_expand_base, w_expand_offset),                    \
      CONST_TENSOR(int8_t, w_depthwise_base, w_depthwise_offset),              \
      CONST_TENSOR(int8_t, w_project_base, w_project_offset),                  \
      CONST_TENSOR(int32_t, b_expand_base, b_expand_offset),                   \
      CONST_TENSOR(int32_t, b_depthwise_base, b_depthwise_offset),             \
      CONST_TENSOR(int32_t, b_project_base, b_project_offset),                 \
      CONST_TENSOR(int32_t, m_expand_base, m_expand_offset),                   \
      CONST_TENSOR(int32_t, m_depthwise_base, m_depthwise_offset),             \
      CONST_TENSOR(int32_t, m_project_base, m_project_offset),                 \
      CONST_TENSOR(int8_t, s_expand_base, s_expand_offset),                    \
      CONST_TENSOR(int8_t, s_depthwise_base, s_depthwise_offset),              \
      CONST_TENSOR(int8_t, s_project_base, s_project_offset),                  \
      CONST_TENSOR(int32_t, config_base, config_offset),                       \
      TENSOR(int8_t, output_base, output_offset),                              \
      TENSOR(int8_t, scratch_base, scratch_offset), input_offset,              \
      w_expand_offset, w_depthwise_offset, w_project_offset, b_expand_offset,  \
      b_depthwise_offset, b_project_offset, m_expand_offset,                   \
      m_depthwise_offset, m_project_offset, s_expand_offset,                   \
      s_depthwise_offset, s_project_offset, config_offset, output_offset,      \
      scratch_offset)

#define VMCU_IBN_BASES_VALID                                                   \
  (input_base != NULL && w_expand_base != NULL &&                              \
   w_depthwise_base != NULL && w_project_base != NULL &&                       \
   b_expand_base != NULL && b_depthwise_base != NULL &&                        \
   b_project_base != NULL && m_expand_base != NULL &&                          \
   m_depthwise_base != NULL && m_project_base != NULL &&                       \
   s_expand_base != NULL && s_depthwise_base != NULL &&                        \
   s_project_base != NULL && config_base != NULL && output_base != NULL &&     \
    scratch_base != NULL)

void oneliner_vmcu_ibn_s8(VMCU_IBN_ARGUMENTS) {
  if (!VMCU_IBN_BASES_VALID) {
    return;
  }
  VMCU_IBN_CALL;
}

/* Keep the name used by the original MCUNet bitcode prototype. */
void oneliner_ibn_s8(VMCU_IBN_ARGUMENTS) {
  if (!VMCU_IBN_BASES_VALID) {
    return;
  }
  VMCU_IBN_CALL;
}
