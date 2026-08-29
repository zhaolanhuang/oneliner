# vMCU compact activation scheduling

vMCU minimizes activation RAM by scheduling reads and writes at activation-
segment granularity. Each virtual tensor records its producer, all consumers,
segment size, last-read events, and first-write events. The planner chooses a
topological kernel order and circular-pool bases so that an output byte can
overwrite an input byte immediately after its final read.

This pipeline runs between IREE's `preprocessing` and dispatch-creation phases:

```rust
#[model(
    "models/model.tflite",
    vmcu = "auto",
    vmcu_schedule = "bounded",
    vmcu_search_states = 1_000_000,
)]
struct MyModel;
```

`vmcu = "off"` retains the regular immutable `ModelInference` API. `auto` and
`strict` use the destructive `InPlaceModelInference` API and one
`VmcuIoBuffer`: the caller fills the borrowed input view, calls
`run_in_place`, and consumes the returned borrowed output view. The output view
must be dropped before the same pool can be modified or used for another run.

## Planning

The compact graph folds reshape/collapse/expand views and absorbs static
padding as zero-point reads. It supports quantized int8 Conv2D, Depthwise,
fully connected, and inverted bottleneck candidates. Unsupported operations
form materialization boundaries in `auto`; `strict` requires complete compact
coverage from the model input to output.

Static identity-map quantized residual expression trees and terminal NHWC sum
pooling are scalarized directly from pool reads to pool writes. They remain DAG
nodes in the plan but do not allocate a materialized activation; schema v4
lists these under `boundaries` and reserves `materialized_boundaries` for true
unsupported fallbacks.

An inverted bottleneck uses the general `K²+2` local schedule: `K²` i8 B
segments, one i8 C segment, and one `Cout`-lane i32 D accumulator. Therefore
3×3, 5×5, and 7×7 kernels use 11, 27, and 51 local segments respectively.
Padding never allocates a padded activation.

The available deterministic search policies are:

| Policy | Behavior |
| --- | --- |
| `bounded` | Branch-and-bound up to the configured state limit; returns the best replay-verified plan found. |
| `optimal` | Exhausts the search space and proceeds only with a proven optimum. |
| `greedy` | Uses deterministic topological selection and placement without optimality search. |

Planning is byte-addressed. The logical minimum is retained in the plan and the
physical pool is rounded to 64 bytes. The final output is contiguous and does
not cross the ring boundary. A byte-by-byte replay rejects every plan that
writes over a still-live input segment.

## Lowering and ABI

Every compact kernel is emitted as an explicit `flow.dispatch.workgroups` with
one tied read/write pool operand. Pool SSA versions serialize overwrite
dependencies, while `RAMLoad` and `RAMStore` lower logical offsets modulo the
logical pool size. Conv2D, Depthwise, fully connected, and composite IBN
kernels accumulate locally and write directly to their planned output base;
they do not return full intermediate activation tensors.

The public MLIR boundary is one pool tensor in and the same tied pool tensor
out. The Flow converter validates the import/export pair, folds it to one
external `inout` resource, removes the full-pool wrapper copy, and does not add
the pool to `Workspace`. Generated Rust therefore receives exactly one
`BufferMut` for model I/O.

Schema-v4 `vmcu.plan.json` records the virtual DAG, execution order, search
statistics, tensor bases and ranges, per-kernel access events and workspace,
materialization boundaries, and logical/aligned pool sizes. Flow metadata also
records the pool and borrowed input/output offsets; the build rejects a size or
aliasing mismatch between these two artifacts.

## Safety and resource accounting

Initial matching is read-only. The source is reparsed and candidate identities
and semantic facts must reproduce exactly before mutation. A compact region is
emitted atomically and old operations are removed in reverse topological order,
without retaining handles to deleted MLIR operations. Direct tensor-ABI test
registries use the transactional fallback: emit one candidate, discard the
entire analysis, normalize, run every registered analyzer again, apply the SRAM
gate, and only then select the next candidate. Both paths verify, serialize,
independently reparse, and verify the result again.

Post-lowering SRAM is reported as:

```text
aligned I/O pool + unsupported Stream transient arena + object maximum stack
```

Local B/C/D workspace is reported separately but is not added twice when it is
already resident in the measured object stack. `vmcu_sram` is a deployment
gate: `strict` fails above the limit; `auto` can fall back to the immutable
preprocessing module and validates that deployment independently.

The main artifacts are `vmcu.preprocessing.mlir`, `vmcu.rewritten.mlir`,
`vmcu.plan.json`, generated Flow Rust/JSON metadata, and the final object-level
resource evidence.
