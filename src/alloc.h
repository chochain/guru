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

#ifndef GURU_SRC_ALLOC_H_
#define GURU_SRC_ALLOC_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

// define flags
#define FLAG_TAIL_BLOCK     1
#define FLAG_NOT_TAIL_BLOCK 0
#define FLAG_FREE_BLOCK     1
#define FLAG_USED_BLOCK     0

// 31-bit max memory block header
typedef struct used_block {
  uint32_t				size:31;    //!< block size, header included
  uint32_t 				tail:1;     //!< FLAG_TAIL_BLOCK or FLAG_NOT_TAIL_BLOCK
  uint32_t				psize:31; 	//!< offset of previous physical block
  uint32_t 				free:1;     //!< FLAG_FREE_BLOCK or BLOCK_IS_NOT_FREE
} used_block;

typedef struct free_block {
  uint32_t				size:31;    //!< block size, header included
  uint32_t 				tail:1;     //!< FLAG_TAIL_BLOCK or FLAG_NOT_TAIL_BLOCK
  uint32_t				psize:31; 	//!< offset of previous physical block
  uint32_t 				free:1;     //!< FLAG_FREE_BLOCK or BLOCK_IS_NOT_FREE

  struct free_block 	*next;
  struct free_block 	*prev;
} free_block;

#define BLOCKHEAD(p) ((uint8_t *)p - sizeof(used_block))
#define BLOCKDATA(p) ((uint8_t *)p + sizeof(used_block))
#define BLOCKSIZE(p) (p->size - sizeof(used_block))

__GURU__ void *mrbc_alloc(unsigned int size);
__GURU__ void *mrbc_realloc(void *ptr, unsigned int size);
__GURU__ void  mrbc_free(void *ptr);
__GURU__ void  mrbc_free_all();

// for statistics or debug. (need #define GURU_DEBUG)
__GPU__ void guru_memory_init(void *mem, unsigned int sz);

void *guru_malloc(size_t sz, int mem_type);		// mem_type: 0=>managed, 1=>device
void guru_malloc_stat(int stat[]);
void guru_dump_alloc_stat(void);

#ifdef __cplusplus
}
#endif
#endif
