/* vMCU ukernel library aggregator.
 *
 * This source is compiled to LLVM bitcode and linked into IREE executables
 * as a ukernel library. Each exported function receives IREE element offsets
 * for every tensor; the CONST_TENSOR/TENSOR macros apply them as typed
 * pointer arithmetic. The library provides the inverted bottleneck,
 * pointwise pair, single 2D convolution, and fully connected segment
 * kernels; the rewriter emits one ukernel descriptor per matched module.
 */

#include "oneliner_vmcu_ibn.c"
#include "oneliner_vmcu_conv2d.c"
#include "oneliner_vmcu_fc.c"
#include "oneliner_vmcu_pair.c"
