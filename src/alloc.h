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

#ifndef mrbc_memsize_t
#define mrbc_memsize_t     	uint16_t
#endif

// define flags
#define FLAG_TAIL_BLOCK     1
#define FLAG_NOT_TAIL_BLOCK 0
#define FLAG_FREE_BLOCK     1
#define FLAG_USED_BLOCK     0

// memory block header
typedef struct used_block {
  uint8_t 				tail;      	//!< FLAG_TAIL_BLOCK or FLAG_NOT_TAIL_BLOCK
  uint8_t 				free;      	//!< FLAG_FREE_BLOCK or BLOCK_IS_NOT_FREE
  uint16_t				u;			// align 32

  mrbc_memsize_t 		size;       //!< block size, header included
  mrbc_memsize_t 		offset; 	//!< offset of previous physical block
} used_block;

typedef struct free_block {
  uint8_t 				tail;      	//!< FLAG_TAIL_BLOCK or FLAG_NOT_TAIL_BLOCK
  uint8_t 				free;      	//!< FLAG_FREE_BLOCK or BLOCK_IS_NOT_FREE
  uint16_t				u;

  mrbc_memsize_t 		size;       //!< block size, header included
  mrbc_memsize_t 		offset; 	//!< offset of previous physical block

  struct free_block 	*next;
  struct free_block 	*prev;
} free_block;

__GURU__ void *mrbc_alloc(unsigned int size);
__GURU__ void *mrbc_realloc(void *ptr, unsigned int size);
__GURU__ void  mrbc_free(void *ptr);
__GURU__ void  mrbc_free_all();

// for statistics or debug. (need #define MRBC_DEBUG)
__global__ void guru_memory_init(void *mem, unsigned int sz);

void *guru_malloc(size_t sz, int mem_type);		// mem_type: 0=>managed, 1=>device
void dump_alloc_stat(void);

#ifdef __cplusplus
}
#endif
#endif
