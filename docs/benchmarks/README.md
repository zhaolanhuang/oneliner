# vMCU performance optimization summary

Date: 2026-08-21
All latency numbers are QEMU guest SysTick counts over 30 iterations
(best / avg of non-wrapped runs); relative comparison only.

## Step-by-step results (Cortex-M4, MCUNet, best ticks)

| Step | Change | vMCU best | vs previous | Verdict |
| --- | --- | --- | --- | --- |
| baseline | committed state | 1,065,347 | - | - |
| opt1 | always_inline on scale/requantize | 1,079,877 | ~0% | no-op, reverted (already inlined) |
| opt2 | ring-cache modulo -> conditional subtract | 811,994 | **-24%** | kept, committed |
| opt3 | zero-point folded into bias (stack VLAs) | 1,115,841 | +37% | regressed, reverted |
| opt4 | word-wise row copies (aligned fast path) | 723,345 | **-12%** | kept, committed |
| opt5 | padded cache, branch-free depthwise MAC | 845,859 | +17% | regressed, reverted |
| opt6 | ring rotation modulo -> conditional subtract | 738,025 | -2% | kept, committed |

## Final numbers

### Cortex-M4 (mps2-an386), MCUNet
| | standard | vMCU auto | delta |
| --- | --- | --- | --- |
| Flash total | 552,328 B | 430,156 B | -22% |
| RAM arena | 119,552 B | 34,880 B | -71% |
| Latency best | ~823,182 | ~738,025 | **-10%** |
| Latency avg | ~871,044 | ~756,968 | **-13%** |

### Cortex-M7 (mps2-an500), MCUNet
| | standard | vMCU auto | delta |
| --- | --- | --- | --- |
| Latency best | 829,488 | 748,877 | **-10%** |

### Cortex-M4, LeNet5 (small model, ukernel overhead dominant)
| | standard | vMCU auto |
| --- | --- | --- |
| Latency best | 36,758 | 82,130 (+2.2x) |

## Findings

1. The dominant cost was the runtime ring-cache modulo: each (row, tap)
   computed `(ring_start + ky) % kernel_height` as udiv+mls (hardware
   division, no compiler folding for runtime divisors). Removing it cut
   latency by 24% and made vMCU faster than the standard path.
2. Word-wise row copies (aligned fast path with byte tail) gave another
   12%.
3. Zero-point folding and padded-cache branch removal both regressed:
   the stack VLAs hurt code generation and the pad-region fill added
   memory traffic that outweighed the removed branches.
4. Per-step reports are in this directory (baseline-m4-mcunet.txt,
   opt1-*.txt, opt2-*.txt, opt3-*.txt, opt4-*.txt, opt5-*.txt,
   opt6-*.txt).