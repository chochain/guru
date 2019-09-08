/*! @file
  @brief
  GURU memory management.

  <pre>
  Copyright (C) 2019 GreenII.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in GURU.

  </pre>
*/

#ifndef GURU_SRC_ALLOC32_H_
#define GURU_SRC_ALLOC32_H_
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
  U32 poff;							// positive offset to previous block
}
*/
typedef struct used_block {			// 8-bytes
  U32 free : 1;   					//!< flag of free block
  U32 size : 31;  					//!< block size, header included (max 2G)
  U32 tail : 1;   					//!< flag of the tail block
  U32 poff : 31;  					//!< positive offset of previous memory block
} used_block;

typedef struct free_block {			// 16-bytes (i.e. mininum allocation per block)
  U32 free : 1;   //!< flag of free block
  U32 size : 31;  //!< block size, header included (max 2G)
  U32 tail : 1;   //!< flag of the tail block
  U32 poff : 31;  //!< offset of previous physical block

  struct free_block 	*next;		// pointer to next free block
  struct free_block 	*prev;		// pointer to previous free block
} free_block;

#define BLOCKHEAD(p) 	U8PSUB(p, sizeof(used_block))
#define BLOCKDATA(p) 	U8PADD(p, sizeof(used_block))

#define L1_BITS     24  // 00000000 00000000 XXXXXXXX 00000000  // 16+8 levels
#define L2_BITS     4   // 00000000 00000000 00000000 XXXX0000  // 16 entires
#define MN_BITS		4	// 00000000 00000000 00000000 0000XXXX  // 16-bytes smallest blocksize
#define L2_MASK 	((1<<L2_BITS)-1)
#define MIN_BLOCK	(1 << MN_BITS)
#define BASE_BITS   (L2_BITS+MN_BITS)

#define L1(i) 		((i) >> L2_BITS)
#define L2(i) 		((i) & L2_MASK)
#define MSB_BIT 	31                                      // 32-bit MMU
#define FL_SLOTS	(L1_BITS * (1 << L2_BITS))				// slots for free_list pointers (24 * 16 entries)

#define NEXT(p) 	U8PADD(p, p->size + sizeof(used_block))	// size is the raw space only
#define PREV(p) 	U8PSUB(p, p->poff)						// positive offset to previous block
#define CHECK_MINSZ(sz)	assert((sz)>=MIN_BLOCK)

// semaphore
__GURU__ volatile U32 	_mutex_mem;

// memory pool
__GURU__ U32			_memory_pool_size;
__GURU__ U8				*_memory_pool;

// free memory bitmap
__GURU__ U32 			_l1_map;								// use lower 24 bits
__GURU__ U16 			_l2_map[L1_BITS];						// use all 16 bits
__GURU__ free_block		*_free_list[FL_SLOTS];

#define L1_MAP(i)       (_l1_map)
#define L2_MAP(i)       (_l2_map[L1(i)])
#define TIC(n)      	(1 << n)
#define INDEX(l1, l2)   ((l1<<L2_BITS) | l2)

#define SET_L1(i)		(L1_MAP(i) |= TIC(L1(i)))
#define CLR_L1(i)	    (L1_MAP(i) &= ~TIC(L1(i)))
#define SET_L2(i)	    (L2_MAP(i) |= TIC(L2(i)))
#define CLR_L2(i)		(L2_MAP(i) &= ~TIC(L2(i)))
#define SET_MAP(i)      { SET_L1(i); SET_L2(i); }
#define CLEAR_MAP(i)	{ CLR_L2(i); if (L2_MAP(i)==0) CLR_L1(i); }

#ifdef __cplusplus
}
#endif
#endif
