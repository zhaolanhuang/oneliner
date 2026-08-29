import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "oneliner-macro" / "python"))

from oneliner_vmcu.compact_memory import (  # noqa: E402
    KernelAccessSchedule,
    ScheduleSearchMode,
    TensorPlacement,
    VirtualTensor,
    plan_compact_graph,
    replay_compact_graph_plan,
    segment_last_reads,
)
from oneliner_vmcu.schedules import InvertedBottleneckSegmentSchedule  # noqa: E402


def _paper_fc_graph():
    # Figure 3: M=2, K=3, N=2. Each scalar is one logical segment.
    tensors = (
        VirtualTensor("input", 6, None, ("fc",), is_graph_input=True),
        VirtualTensor("output", 4, "fc", (), is_graph_output=True),
    )
    kernel = KernelAccessSchedule(
        "fc",
        ("input",),
        "output",
        # The fourth logical output is scheduled first into the one free slot;
        # the other outputs overwrite input only after its final segment read.
        (("input", (5, 5, 5, 5, 5, 5)),),
        (10, 11, 12, 1),
        1,
        kind="fully_connected",
    )
    return tensors, (kernel,)


class CompactMemoryTests(unittest.TestCase):
    def test_paper_gemm_uses_seven_segments(self):
        tensors, kernels = _paper_fc_graph()
        plan = plan_compact_graph(
            tensors, kernels, search_mode=ScheduleSearchMode.OPTIMAL, alignment=1
        )
        self.assertEqual(plan.logical_pool_bytes, 7)
        self.assertEqual(plan.logical_pool_bytes, max(2 * 2, 2 * 3) + min(2, 3) - 1)
        replay_compact_graph_plan(plan)

    def test_paper_gemm_rejects_a_six_segment_early_overwrite(self):
        tensors, kernels = _paper_fc_graph()
        plan = plan_compact_graph(
            tensors, kernels, search_mode=ScheduleSearchMode.OPTIMAL, alignment=1
        )
        unsafe = replace(
            plan,
            logical_pool_bytes=6,
            allocated_pool_bytes=6,
            placements=(
                TensorPlacement("input", 0, 6, 6),
                TensorPlacement("output", 2, 4, 6),
            ),
        )
        with self.assertRaisesRegex(ValueError, "overwrite hazard"):
            replay_compact_graph_plan(unsafe)

    def test_segment_lifetime_is_maximum_element_lifetime(self):
        self.assertEqual(
            segment_last_reads((0, 3, 1, 2, 9), 2),
            (3, 3, 2, 2, 9),
        )

    def test_diamond_preserves_input_until_its_last_consumer(self):
        tensors = (
            VirtualTensor("input", 4, None, ("left", "right"), is_graph_input=True),
            VirtualTensor("left_value", 3, "left", ("join",)),
            VirtualTensor("right_value", 3, "right", ("join",)),
            VirtualTensor("output", 2, "join", (), is_graph_output=True),
        )
        kernels = (
            KernelAccessSchedule(
                "left", ("input",), "left_value", (("input", (0, 1, 2, 3)),),
                (4, 5, 6), 1,
            ),
            KernelAccessSchedule(
                "right", ("input",), "right_value", (("input", (0, 1, 2, 3)),),
                (4, 5, 6), 1,
            ),
            KernelAccessSchedule(
                "join", ("left_value", "right_value"), "output",
                (("left_value", (0, 1, 2)), ("right_value", (0, 1, 2))),
                (3, 4), 1,
            ),
        )
        plan = plan_compact_graph(tensors, kernels, search_mode="optimal", alignment=1)
        replay_compact_graph_plan(plan)
        self.assertGreaterEqual(plan.logical_pool_bytes, 7)

    def test_search_modes_are_deterministic_and_safe(self):
        tensors, kernels = _paper_fc_graph()
        footprints = {}
        for mode in ScheduleSearchMode:
            first = plan_compact_graph(
                tensors, kernels, search_mode=mode, search_state_limit=64, alignment=1
            )
            second = plan_compact_graph(
                tensors, kernels, search_mode=mode, search_state_limit=64, alignment=1
            )
            self.assertEqual(first.to_dict(), second.to_dict())
            replay_compact_graph_plan(first)
            footprints[mode] = first.logical_pool_bytes
        self.assertLessEqual(footprints[ScheduleSearchMode.OPTIMAL], footprints[ScheduleSearchMode.GREEDY])

        bounded = plan_compact_graph(
            tensors,
            kernels,
            search_mode=ScheduleSearchMode.BOUNDED,
            search_state_limit=1,
            alignment=1,
        )
        self.assertEqual(bounded.explored_states, 1)
        self.assertFalse(bounded.optimal)
        replay_compact_graph_plan(bounded)

    def test_ibn_workspace_is_kernel_squared_plus_two(self):
        expected = {3: 11, 5: 27, 7: 51}
        for kernel, segments in expected.items():
            schedule = InvertedBottleneckSegmentSchedule(16, 48, 24, kernel, kernel)
            report = schedule.to_dict()
            self.assertEqual(report["workspace_segments"], segments)
            self.assertEqual(report["buffers"][0]["count"], kernel * kernel)


if __name__ == "__main__":
    unittest.main()
