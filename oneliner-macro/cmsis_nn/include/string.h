// Minimal freestanding <string.h> for the CMSIS-NN bitcode build.
//
// The CMSIS-NN headers unconditionally include <string.h>, but the ukernel
// bitcode is freestanding and must not depend on a host C library. Clang's
// builtin headers provide <stdint.h>, <stddef.h>, <stdbool.h>, <limits.h>,
// etc., but not <string.h>. These declarations are sufficient for the linked
// CMSIS-NN sources (only memset is called) and resolve to the symbols defined
// in oneliner_cmsis_nn.c.
#ifndef ONELINER_CMSIS_NN_STRING_H
#define ONELINER_CMSIS_NN_STRING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *memcpy(void *dest, const void *src, size_t n);
void *memmove(void *dest, const void *src, size_t n);
void *memset(void *s, int c, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);

#ifdef __cplusplus
}
#endif

#endif
