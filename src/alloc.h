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

// TODO: utilize lower 2 bits for flags
/*
#define BLOCK_FREE_FLAG	0x1
#define BLOCK_TAIL_FLAG	0x2
#define BLOCK_SIZE(b) 	((b)->size & ~(BLOCK_FREE_FLAG|BLOCK_TAIL_FLAG))
#define IS_FREE(b)		((b)->size & BLOCK_FREE_FLAG)
#define IS_TAIL(b)		((b)->size & BLOCK_TAIL_FLAG)
#define SET_FREE(b)		((b)->size &= ~BLOCK_FREE_FLAG)
#define SET_TAIL(b)		((b)->size &= ~BLOCK_TAIL_FLAG)
#define NEXT_BLOCK(b)   ((U8P)b + b->size)
#define PREV_BLOCK(b)	((U8P)b - b->poff)

typedef struct used_block {			// 4-bytes
  U32 size;   						// lower 2 bits used as flags
} used_block;

typedef struct free_block {			// 4-bytes (+ 4 bytes free space)
  U32 size;   						//
  U32 poff;							// offset to previous block
}
*/

typedef struct used_block {			// 8-bytes
  U32 free : 1;   					//!< flag of free block
  U32 size : 31;  					//!< block size, header included (max 2G)
  U32 tail : 1;   					//!< flag of the tail block
  U32 poff : 31;  					//!< offset of previous physical block
} used_block;

typedef struct free_block {			// 16-bytes (i.e. mininum allocation per block)
  U32 free : 1;   //!< flag of free block
  U32 size : 31;  //!< block size, header included (max 2G)
  U32 tail : 1;   //!< flag of the tail block
  U32 poff : 31;  //!< offset of previous physical block

  struct free_block 	*next;		// pointer to next free block
  struct free_block 	*prev;		// pointer to previous free block
} free_block;

#define BLOCKHEAD(p) 	((U8P)p - sizeof(used_block))
#define BLOCKDATA(p) 	((U8P)p + sizeof(used_block))
#define BLOCKSIZE(p) 	(p->size - sizeof(used_block))

__GURU__ void *mrbc_alloc(U32 size);
__GURU__ void *mrbc_realloc(void *ptr, U32 size);
__GURU__ void  mrbc_free(void *ptr);
__GURU__ void  mrbc_free_all();

__GPU__ void guru_memory_init(void *mem, U32 sz);

void *guru_malloc(U32 sz, U32 mem_type);		// mem_type: 0=>managed, 1=>device
void guru_malloc_stat(U32 stat[]);
void guru_dump_alloc_stat(U32 trace);

#ifdef __cplusplus
}
#endif
#endif
