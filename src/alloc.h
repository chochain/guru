/*! @file
  @brief
  GURU memory management.

  <pre>
  Copyright (C) 2019 GreenII.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in GURU.

  </pre>
*/

#ifndef GURU_SRC_ALLOC_H_
#define GURU_SRC_ALLOC_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

// GURU memory management unit
// calling and return pointers are to raw space instead of block head
__GURU__ void guru_memory_clear();
__GURU__ void *guru_alloc(U32 sz);
__GURU__ void *guru_realloc(void *ptr, U32 sz);
__GURU__ void guru_free(void *ptr);

__GPU__  void guru_memory_init(void *mem, U32 sz);

// CUDA memory management functions
__HOST__ void *cuda_malloc(U32 sz, U32 mem_type);		// mem_type: 0=>managed, 1=>device
__HOST__ void guru_dump_alloc_stat(U32 trace);

#ifdef __cplusplus
}
#endif
#endif
