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

typedef struct used_block {			// 8-bytes
  uint32_t				size : 31;  //!< block size, header included (max 2G)
  uint32_t 				tail : 1;   //!< flag of the tail block
  uint32_t				poff : 31; 	//!< offset of previous physical block
  uint32_t 				free : 1;   //!< flag of free block
} used_block;

typedef struct free_block {			// 16-bytes (i.e. mininum allocation per block)
  uint32_t				size : 31;
  uint32_t 				tail : 1;
  uint32_t				poff : 31;
  uint32_t 				free : 1;

  struct free_block 	*next;		// pointer to next free block
  struct free_block 	*prev;		// pointer to previous free block
} free_block;

#define BLOCKHEAD(p) ((uint8_t *)p - sizeof(used_block))
#define BLOCKDATA(p) ((uint8_t *)p + sizeof(used_block))
#define BLOCKSIZE(p) (p->size - sizeof(used_block))

__GURU__ void *mrbc_alloc(unsigned int size);
__GURU__ void *mrbc_realloc(void *ptr, unsigned int size);
__GURU__ void  mrbc_free(void *ptr);
__GURU__ void  mrbc_free_all();

__GPU__ void guru_memory_init(void *mem, unsigned int sz);

void *guru_malloc(size_t sz, int mem_type);		// mem_type: 0=>managed, 1=>device
void guru_malloc_stat(int stat[]);
void guru_dump_alloc_stat(void);

#ifdef __cplusplus
}
#endif
#endif
