# vMCU compact activation scheduling

The paper-to-source index in [vmcu-paper-map.md](vmcu-paper-map.md) records the
exact section, PDF page, figure/equation, and current source line for each
implemented paper concept. It also marks repository-specific extensions and
paper optimizations that are not yet implemented.

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

All modes implement the same `ModelInference` API. In `auto` and `strict`, the
generated model instance owns the compact I/O pool: `run` copies the owned
input tensor into its planned input view, executes in place, and copies the
planned output view into the returned owned output tensor. The pool is not
exposed to callers.

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
not cross the ring boundary. Circular-range/segment replay rejects every plan
that writes over a still-live input segment.

Compact planning runs only in the first, read-only parse: graph discovery,
analytical access-event construction, schedule search, and replay produce an
immutable plan with no MLIR handles. The transactional second parse reruns the
semantic registry to recover live candidate and boundary values, checks their
stable IDs, semantic signatures, and DAG/boundary signature, then replays the
existing plan and emits it. It does not rebuild access events or search again.

Conv2D, Depthwise, fully connected, and IBN lifetimes use fixed-radix execution
coordinates rather than Python simulation of every MAC. Spatial Conv/Depthwise
last consumers are solved by inverse stride/dilation/padding mapping once per
input H/W coordinate; FC and IBN last reads are computed directly from their
last output/channel coordinates. Event integers encode only the access total
order required by the overwrite proof, not an executed-MAC count.

## Lowering and ABI

Every compact kernel is emitted as an explicit `flow.dispatch.workgroups` with
one tied read/write pool operand. Pool SSA versions serialize overwrite
dependencies, while `RAMLoad` and `RAMStore` lower logical offsets modulo the
logical pool size. Conv2D, Depthwise, fully connected, and composite IBN
kernels accumulate locally and write directly to their planned output base;
they do not return full intermediate activation tensors.

The public MLIR boundary is one pool tensor in and the same tied pool tensor
out. IREE lowers this to one external resource that is both read and written.
The generic Flow converter infers its `inout` role solely from those accesses
and does not read a vMCU plan. The Rust build layer reads schema-v4 pool/view
metadata and validates it against that external resource. The pool is not
added to `Workspace`; generated Rust receives exactly one `BufferMut`.

Schema-v4 `vmcu.plan.json` records the virtual DAG, execution order, search
statistics, tensor bases and ranges, compact grouped-affine output-write
schedules, workspace, materialization boundaries, and logical/aligned pool
sizes. Input last reads remain compiler-internal and are stored once per
activation segment because they are only consumed by planning and replay. Flow
metadata remains backend-generic; the Rust build combines it with the plan and
rejects a pool size, logical view, or aliasing mismatch.

## Safety and resource accounting

Initial matching and compact planning are read-only. The source is reparsed;
candidate identities, semantic facts, and graph boundaries must reproduce
exactly before mutation. Only the context-free first-pass plan crosses this
parse boundary. A compact region is emitted atomically from second-context
handles and old operations are removed in reverse topological order. Direct
tensor-ABI test registries use the transactional fallback: emit one candidate,
discard the entire analysis, normalize, run every registered analyzer again,
apply the SRAM gate, and only then select the next candidate. Both paths verify,
serialize, independently reparse, and verify the result again.

Post-lowering SRAM is reported as:

```text
aligned I/O pool + unsupported Stream transient arena + object maximum stack
```

This is the compiler-managed deployment footprint. The current unified
`ModelInference` wrapper additionally keeps the caller-owned input and returned
output tensors live while `run` copies them into and out of the model-owned
pool. Those interface tensors are shown separately in the Rust build report.

Local B/C/D workspace is reported separately but is not added twice when it is
already resident in the measured object stack. `vmcu_sram` is a deployment
gate: `strict` fails above the limit; `auto` can fall back to the immutable
preprocessing module and validates that deployment independently.

The main artifacts are `vmcu.preprocessing.mlir`, `vmcu.rewritten.mlir`,
`vmcu.plan.json`, generated Flow Rust/JSON metadata, and the final object-level
resource evidence.
