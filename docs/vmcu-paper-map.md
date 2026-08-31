# vMCU paper-to-source map

This index maps the implementation to *vMCU: A Memory-efficient DNN Training
and Inference Framework for Microcontrollers* (MLSys 2024), using the repository
copy `vMCU.pdf`.

The PDF has no printed line numbers. A paper citation therefore uses the stable
tuple **section + PDF page + figure/equation/pseudocode operation**. Source
citations use exact line numbers in this revision. Re-run the line-number audit
after moving code.

## Paper-derived planning

| Paper locator | Implemented content | Source location |
| --- | --- | --- |
| §2.4, PDF pp.3–4, Figure 1(c) | FC input/output overlap and rejection of an early overwrite | [`tests/test_vmcu_compact_memory.py:23`](../tests/test_vmcu_compact_memory.py#L23), [`tests/test_vmcu_compact_memory.py:45`](../tests/test_vmcu_compact_memory.py#L45), [`tests/test_vmcu_compact_memory.py:55`](../tests/test_vmcu_compact_memory.py#L55) |
| §3, PDF p.4, Figure 2 | Compiler-level coordination of graph analysis, memory planning, and kernel generation | [`compact_analysis.py:860`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L860), [`rewrite.py:242`](../oneliner-macro/python/oneliner_vmcu/rewrite.py#L242), [`pool_emitter.py:1033`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L1033) |
| §4, PDF p.4, circular pool and row-major `Laddr` | Virtual activation representation and logical-to-linear address mapping | [`compact_memory.py:39`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L39), [`pool_emitter.py:77`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L77) |
| §4, PDF p.4, `Pool[addr % (MemCap/Seg)]` | Byte-equivalent circular physical addressing | [`pool_emitter.py:89`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L89) |
| §4, PDF pp.4–5, iteration instances and access functions | Segment-compressed last reads and grouped-affine first writes, encoded analytically in emitter order | [`compact_memory.py:80`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L80), [`compact_memory.py:122`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L122), [`compact_analysis.py:393`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L393), [`compact_analysis.py:451`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L451), [`compact_analysis.py:522`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L522) |
| §4, PDF p.5, `bIn`/`bOut` | Physical tensor placement in the pool | [`compact_memory.py:253`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L253) |
| §4, PDF p.5, Equation (1) | Select an output base by intersecting circular physical ranges and comparing only overlapping segment lifetimes | [`compact_memory.py:441`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L441) |
| §4, PDF p.5, Figure 3 | GEMM pool size `max(MN,MK)+min(N,K)-1`; `M=2,K=3,N=2` gives seven segments | [`compact_memory.py:678`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L678), [`tests/test_vmcu_compact_memory.py:45`](../tests/test_vmcu_compact_memory.py#L45) |
| §5.2, PDF p.6, Equation (2), graph `G=(V,E)` | Producer/all-consumer DAG lifetimes, including branches and residual uses | [`compact_analysis.py:596`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L596), [`compact_memory.py:441`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L441), [`tests/test_vmcu_compact_memory.py:79`](../tests/test_vmcu_compact_memory.py#L79) |
| Introduction, PDF p.2, segment lifetime | Store the maximum element lifetime once for each activation segment | [`compact_memory.py:358`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L358) |
| §5.3, PDF p.7, segment-size trade-off | Select Conv/IBN channel-lane segment widths | [`compact_analysis.py:743`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L743), [`schedules/inverted_bottleneck.py:23`](../oneliner-macro/python/oneliner_vmcu/schedules/inverted_bottleneck.py#L23) |

## Paper-derived kernel lowering

| Paper locator | Implemented content | Source location |
| --- | --- | --- |
| §5.1, PDF pp.5–6, Figures 4–5, `RAMLoad` | Load one i8 from the planned circular address | [`pool_emitter.py:102`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L102) |
| §5.1, PDF pp.5–6, Figures 4–5, `RAMStore`/implicit `RAMFree` | Store at `bOut`; freeing is represented by a later proven-safe overwrite | [`pool_emitter.py:125`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L125), [`compact_memory.py:441`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L441) |
| §5.1, PDF pp.5–6, two-level tiling | Generated activation/reduction loop structure | [`pool_emitter.py:210`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L210) |
| §5.1, PDF p.5, Figure 4 | FC/GEMM compact traversal and output overwrite | [`compact_analysis.py:522`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L522), [`pool_emitter.py:391`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L391) |
| §5.1, PDF p.6, Figure 5 | Conv input access schedule, reduction, quantization, and direct compact output | [`compact_analysis.py:393`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L393), [`pool_emitter.py:236`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L236) |
| §5.2, PDF pp.6–7, Figure 6 | IBN expansion/depthwise/projection/residual access schedule | [`compact_analysis.py:451`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L451), [`pool_emitter.py:439`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L439) |
| §5.2, PDF p.7, Figure 6, B | `K²` expansion patch segments; 3×3 gives nine B segments | [`pool_emitter.py:487`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L487) |
| §5.2, PDF p.7, Figure 6, C | One post-depthwise i8 segment | [`pool_emitter.py:571`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L571) |
| §5.2, PDF p.7, Figure 6, D | Projection accumulator segment | [`pool_emitter.py:633`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L633) |
| §5.2, PDF p.7, Figure 6, E | Optional residual read/add followed by the final compact-pool store | [`pool_emitter.py:682`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L682) |
| §5.2, PDF p.7, Figure 6, 11 segments | 3×3 `9+1+1`; generalized schedule reports `K²+2` | [`schedules/inverted_bottleneck.py:23`](../oneliner-macro/python/oneliner_vmcu/schedules/inverted_bottleneck.py#L23), [`tests/test_vmcu_compact_memory.py:147`](../tests/test_vmcu_compact_memory.py#L147) |
| §5.2, PDF p.6, Equation (2), residual E edge | Evaluate validated residual chains directly from and back to the pool | [`compact_analysis.py:189`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L189), [`pool_emitter.py:780`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L780) |

## Compiler and runtime realization

| Paper locator | Implemented content | Source location |
| --- | --- | --- |
| §6, PDF p.7, compiler support | Transactional MLIR analysis, one source-pass plan, second-pass rebind/replay, and graph emission | [`rewrite.py:203`](../oneliner-macro/python/oneliner_vmcu/rewrite.py#L203), [`compact_analysis.py:860`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L860), [`compact_analysis.py:899`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L899) |
| §3 Figure 2 and §6, PDF pp.4, 7 | One generated dispatch per scheduled kernel, chained through a tied read/write pool | [`pool_emitter.py:168`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L168), [`pool_emitter.py:1033`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L1033) |
| §4, PDF pp.4–5, shared input/output pool | Rewrite the public MLIR ABI to one tied pool | [`pool_emitter.py:1000`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L1000) |
| §4, PDF pp.4–5, shared input/output pool | Preserve the same physical resource through generic IREE Stream `inout` inference and a model-owned generated Rust buffer | [`stream_flow_to_rust.py:410`](../oneliner-macro/python/oneliner_iree/stream_flow_to_rust.py#L410), [`stream_flow_to_rust.py:1213`](../oneliner-macro/python/oneliner_iree/stream_flow_to_rust.py#L1213), [`vmcu/codegen.rs:8`](../oneliner-macro/src/backend/iree/vmcu/codegen.rs#L8), [`vmcu/codegen.rs:29`](../oneliner-macro/src/backend/iree/vmcu/codegen.rs#L29), [`vmcu/codegen.rs:58`](../oneliner-macro/src/backend/iree/vmcu/codegen.rs#L58) |
| §4 Equation (1) and §5.2 Equation (2) | Independently replay every circular physical overlap at segment granularity before accepting a plan | [`compact_memory.py:631`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L631) |

## Deliberate engineering extensions and current gaps

These items support the implementation but must not be attributed to the paper
as written:

- `greedy` and exhaustive/budgeted `optimal` are repository search policies
  ([`compact_memory.py:26`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L26),
  [`compact_memory.py:677`](../oneliner-macro/python/oneliner_vmcu/compact_memory.py#L677)).
  The paper instead suggests solving its offset constraints with ILP in §4.
- General arbitrary-static-DAG extraction, conservative unsupported boundaries,
  segment/range replay, 64-byte allocation alignment, schema-v4 plans, and
  transactional reparse/verify are repository safety extensions
  ([`compact_analysis.py:596`](../oneliner-macro/python/oneliner_vmcu/compact_analysis.py#L596),
  [`rewrite.py:203`](../oneliner-macro/python/oneliner_vmcu/rewrite.py#L203)).
- IREE tied dispatch resources, Stream copy folding, and Rust borrow-checked
  output views are backend/ABI adaptations, not paper interfaces
  ([`pool_emitter.py:168`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L168),
  [`stream_flow_to_rust.py:1061`](../oneliner-macro/python/oneliner_iree/stream_flow_to_rust.py#L1061),
  [`interface.rs:248`](../oneliner-runtime/src/interface.rs#L248)).
- Arbitrary static standalone Conv2D/depthwise, generic direct boundaries, `K²+2` IBN,
  and a `Cout`-lane D accumulator generalize the paper's shown 3×3 IBN
  ([`conv2d.py:33`](../oneliner-macro/python/oneliner_vmcu/patterns/conv2d.py#L33),
  [`pool_emitter.py:236`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L236),
  [`pool_emitter.py:312`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L312),
  [`pool_emitter.py:633`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L633)).
- vMCU §6.1 describes vectorized `RAMLoad`/`Dot`/`Broadcast`. The active pool
  emitter currently performs scalar one-byte loads/stores
  ([`pool_emitter.py:102`](../oneliner-macro/python/oneliner_vmcu/pool_emitter.py#L102));
  vectorized contiguous-segment lowering is not implemented yet.
