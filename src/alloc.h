/*! @file
  @brief
  mrubyc memory management.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in mruby/c.

  </pre>
*/

#ifndef MRBC_SRC_ALLOC_H_
#define MRBC_SRC_ALLOC_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MRBC_ALLOC_MEMSIZE_T
#define MRBC_ALLOC_MEMSIZE_T     uint16_t
#endif

// define flags
#define FLAG_TAIL_BLOCK     1
#define FLAG_NOT_TAIL_BLOCK 0
#define FLAG_FREE_BLOCK     1
#define FLAG_USED_BLOCK     0

// memory block header
typedef struct USED_BLOCK {
  unsigned int 			t : 1;       //!< FLAG_TAIL_BLOCK or FLAG_NOT_TAIL_BLOCK
  unsigned int 			f : 1;       //!< FLAG_FREE_BLOCK or BLOCK_IS_NOT_FREE
  uint8_t			   	u;

  MRBC_ALLOC_MEMSIZE_T 	size;        //!< block size, header included
  MRBC_ALLOC_MEMSIZE_T 	prev_offset; //!< offset of previous physical block
} USED_BLOCK;

typedef struct FREE_BLOCK {
  unsigned int         	t : 1;       //!< FLAG_TAIL_BLOCK or FLAG_NOT_TAIL_BLOCK
  unsigned int         	f : 1;       //!< FLAG_FREE_BLOCK or BLOCK_IS_NOT_FREE
  uint8_t				u;

  MRBC_ALLOC_MEMSIZE_T size;        //!< block size, header included
  MRBC_ALLOC_MEMSIZE_T prev_offset; //!< offset of previous physical block

  struct FREE_BLOCK *next_free;
  struct FREE_BLOCK *prev_free;
} FREE_BLOCK;

__GURU__ void  mrbc_init_alloc(void *ptr, unsigned int size);
__GURU__ void  mrbc_raw_free(void *ptr);
__GURU__ void *mrbc_raw_alloc(unsigned int size);
__GURU__ void *mrbc_raw_realloc(void *ptr, unsigned int size);

__GURU__ void *mrbc_alloc(unsigned int size);
__GURU__ void *mrbc_realloc(void *ptr, unsigned int size);
__GURU__ void  mrbc_free(void *ptr);
__GURU__ void  mrbc_free_all();

// for statistics or debug. (need #define MRBC_DEBUG)
__GURU__ void  mrbc_alloc_statistics(int *total, int *used, int *free, int *fragmentation);
__GURU__ int   mrbc_alloc_used();

#ifdef __cplusplus
}
#endif
#endif
