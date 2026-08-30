import sys
import unittest
from math import prod
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "oneliner-macro" / "python"))

from oneliner_vmcu.compact_analysis import (  # noqa: E402
    _conv_events,
    _fc_events,
    _flatten_nhwc,
    _ibn_events,
)


def _event_order(reads, writes):
    events = [
        (event, "read", index)
        for index, event in enumerate(reads)
        if event > 0
    ]
    events.extend((event, "write", index) for index, event in enumerate(writes))
    return tuple((kind, index) for _, kind, index in sorted(events))


def _brute_conv(
    input_shape, output_shape, kernel_shape, strides, dilations, padding_low, depthwise
):
    reads = [-1] * prod(input_shape)
    writes = [0] * prod(output_shape)
    step = 0
    for n in range(output_shape[0]):
        for oh in range(output_shape[1]):
            for ow in range(output_shape[2]):
                for oc in range(output_shape[3]):
                    channels = (oc,) if depthwise else range(input_shape[3])
                    for kh in range(kernel_shape[0]):
                        ih = oh * strides[0] + kh * dilations[0] - padding_low[1]
                        if not 0 <= ih < input_shape[1]:
                            continue
                        for kw in range(kernel_shape[1]):
                            iw = ow * strides[1] + kw * dilations[1] - padding_low[2]
                            if not 0 <= iw < input_shape[2]:
                                continue
                            for channel in channels:
                                step += 1
                                reads[_flatten_nhwc(input_shape, (n, ih, iw, channel))] = step
                    step += 1
                    writes[_flatten_nhwc(output_shape, (n, oh, ow, oc))] = step
    return tuple(max(0, event) for event in reads), tuple(writes)


def _brute_fc(rows, input_channels, output_channels):
    reads = [0] * (rows * input_channels)
    writes = [0] * (rows * output_channels)
    step = 0
    for row in range(rows):
        for output_channel in range(output_channels):
            for input_channel in range(input_channels):
                step += 1
                reads[row * input_channels + input_channel] = step
            step += 1
            writes[row * output_channels + output_channel] = step
    return tuple(reads), tuple(writes)


def _brute_ibn(candidate, input_shape):
    output_shape = candidate.output_shape
    depthwise = candidate.depthwise
    kernel_h, kernel_w = depthwise.weight_shape[:2]
    segment_lanes = min(input_shape[3], output_shape[3])
    expanded_channels = candidate.expansion.output_shape[3]
    reads = [-1] * prod(input_shape)
    writes = [0] * prod(output_shape)
    step = 0
    for n in range(output_shape[0]):
        for oh in range(output_shape[1]):
            for ow in range(output_shape[2]):
                for chunk in range(0, expanded_channels, segment_lanes):
                    for kh in range(kernel_h):
                        ih = (
                            oh * depthwise.strides[0]
                            + kh * depthwise.dilations[0]
                            - depthwise.padding_low[1]
                        )
                        if not 0 <= ih < input_shape[1]:
                            continue
                        for kw in range(kernel_w):
                            iw = (
                                ow * depthwise.strides[1]
                                + kw * depthwise.dilations[1]
                                - depthwise.padding_low[2]
                            )
                            if not 0 <= iw < input_shape[2]:
                                continue
                            valid_lanes = min(segment_lanes, expanded_channels - chunk)
                            for _lane in range(valid_lanes):
                                for channel in range(input_shape[3]):
                                    step += 1
                                    reads[_flatten_nhwc(input_shape, (n, ih, iw, channel))] = step
                for output_channel in range(output_shape[3]):
                    if candidate.residual is not None:
                        step += 1
                        reads[_flatten_nhwc(input_shape, (n, oh, ow, output_channel))] = step
                    step += 1
                    writes[_flatten_nhwc(output_shape, (n, oh, ow, output_channel))] = step
    return tuple(max(0, event) for event in reads), tuple(writes)


class CompactEventTests(unittest.TestCase):
    def test_analytical_events_preserve_scalar_emitter_order(self):
        input_shape = (1, 4, 5, 3)
        kernel_shape = (3, 2, 3, 4)
        strides = (2, 1)
        dilations = (1, 2)
        padding_low = (0, 1, 1, 0)
        for depthwise, output_shape in (
            (False, (1, 2, 4, 4)),
            (True, (1, 2, 4, 3)),
        ):
            analytical = _conv_events(
                input_shape,
                output_shape,
                kernel_shape,
                strides,
                dilations,
                padding_low,
                depthwise=depthwise,
            )
            brute = _brute_conv(
                input_shape,
                output_shape,
                kernel_shape,
                strides,
                dilations,
                padding_low,
                depthwise,
            )
            self.assertEqual(_event_order(*analytical), _event_order(*brute))

        fc = SimpleNamespace(rows=2, input_channels=3, output_channels=4, output_shape=(2, 4))
        self.assertEqual(
            _event_order(*_fc_events(fc)),
            _event_order(*_brute_fc(2, 3, 4)),
        )

        depthwise = SimpleNamespace(
            weight_shape=(3, 3, 5, 1),
            strides=(1, 1),
            dilations=(1, 1),
            padding_low=(0, 1, 1, 0),
        )
        ibn = SimpleNamespace(
            output_shape=(1, 4, 5, 3),
            depthwise=depthwise,
            expansion=SimpleNamespace(output_shape=(1, 4, 5, 5)),
            residual=object(),
        )
        self.assertEqual(
            _event_order(*_ibn_events(ibn, input_shape)),
            _event_order(*_brute_ibn(ibn, input_shape)),
        )


if __name__ == "__main__":
    unittest.main()
