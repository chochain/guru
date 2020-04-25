/*! @file
  @brief
  GURU memory management.

  <pre>
  Copyright (C) 2019 GreenII.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in GURU.

  </pre>
*/

#ifndef GURU_SRC_MMU_H_
#define GURU_SRC_MMU_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RMem {
	U32 	total;
	U32		nfree;
	U32		free;
	U32 	nused;
	U32		used;
	U32		nblk;
	U32		nfrag;
	U32		pct_used;
} guru_mstat;

// GURU memory management unit
// calling and return pointers are to raw space instead of block head
__GPU__  void 			guru_mmu_init(void *mem, U32 sz);
__GURU__ void 			guru_mmu_clear();
__GURU__ void  			guru_mmu_stat(guru_mstat *s);

__GURU__ void 			*guru_alloc(U32 sz);
__GURU__ void 			*guru_realloc(void *ptr, U32 sz);
__GURU__ void 			guru_free(void *ptr);

__GURU__ GV				*guru_gv_alloc(U32 n);					// array of gv
__GURU__ GV				*guru_gv_realloc(GV *gv, U32 sz);

// CUDA memory management functions
__HOST__ void 			*cuda_malloc(U32 sz, U32 mem_type);		// mem_type: 0=>managed, 1=>device
__HOST__ void           cuda_free(void *mem);

__HOST__ U32 			guru_mmu_check(U32 level);

#ifdef __cplusplus
}
#endif
#endif
