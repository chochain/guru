/*! @file
  @brief
  GURU memory management.

  <pre>
  Copyright (C) 2019 GreenII.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in GURU.

  </pre>
*/

#ifndef GURU_SRC_MMU32_H_
#define GURU_SRC_MMU32_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif
/* TODO: 8-byte aligned so lower 3 bits can be utilized for flags

GURU_MEM_BLOCK
 	 bsz : block size, header included (max 2G)
 	 psz : prior adjacent memory block size
*/
#define GURU_MEM_BLOCK				\
	U32 bsz;						\
	U32 psz

typedef struct used_block {			// 8-bytes
	GURU_MEM_BLOCK;
} used_block;

typedef struct free_block {			// 16-bytes (i.e. mininum allocation per block)
	GURU_MEM_BLOCK;
	S32 next;						// offset to next free block
	S32 prev;						// offset to previous free block
} free_block;

#define FREE_FLAG		0x1
#define IS_FREE(b)		((b)->psz & FREE_FLAG)
#define IS_USED(b)		(!IS_FREE(b))
#define SET_FREE(b)		((b)->psz |=  FREE_FLAG)
#define SET_USED(b)		((b)->psz &= ~FREE_FLAG)

#define NEXT_FREE(b)	((free_block*)((b)->next ? ((b)->next<0 ? U8PSUB((b), -(b)->next) : U8PADD((b), (b)->next)) : NULL))
#define PREV_FREE(b)	((free_block*)((b)->prev ? ((b)->prev<0 ? U8PSUB((b), -(b)->prev) : U8PADD((b), (b)->prev)) : NULL))

#define BLK_AFTER(b) 	(((b)->bsz           ) ? U8PADD(b, (b)->bsz             ) : NULL)		// following adjacent memory block
#define BLK_BEFORE(b) 	(((b)->psz&~FREE_FLAG) ? U8PSUB(b, ((b)->psz&~FREE_FLAG)) : NULL)		// prior adjacent memory block
#define BLK_DATA(b) 	(U8PADD(b, sizeof(used_block)))		// pointer to raw data space
#define BLK_HEAD(p) 	(U8PSUB(p, sizeof(used_block)))		// pointer block given raw data pointer

#define MN_BITS			3									// 00000000 00000000 00000000 00000XXX  // 8-bytes smallest blocksize
#define L2_BITS     	3   								// 00000000 00000000 00000000 00XXX000  // 8 entires
#define L1_BITS     	12									// 00000000 000000XX XXXXXXXX XX000000  // 12 levels (for 256K), check GURU_HEAP_SIZE
#define BASE_BITS   	(L2_BITS+MN_BITS)

#define TIC(n)      	(1 << (n))
#define L2_MASK 		((1<<L2_BITS)-1)
#define MIN_BLOCK		(1 << MN_BITS)

#define L1(i) 			((i) >> L2_BITS)
#define L2(i) 			((i) & L2_MASK)
#define FL_SLOTS		(L1_BITS * (1 << L2_BITS))			// slots for free_list pointers (12*8 entries for 256K memory)

#define L1_MAP(i)       (_l1_map)
#define L2_MAP(i)       (_l2_map[L1(i)])
#define INDEX(l1, l2)   ((l1<<L2_BITS) | l2)

#define SET_L1(i)		(L1_MAP(i) |= TIC(L1(i)))
#define SET_L2(i)	    (L2_MAP(i) |= TIC(L2(i)))
#define SET_MAP(i)      { SET_L1(i); SET_L2(i); }
#define CLR_L1(i)	    (L1_MAP(i) &= ~TIC(L1(i)))
#define CLR_L2(i)		(L2_MAP(i) &= ~TIC(L2(i)))
#define CLEAR_MAP(i)	{ CLR_L2(i); if ((L2_MAP(i))==0) CLR_L1(i); }

#define MIN_BLOCK_SIZE	(sizeof(free_block))				// 16-byte (need space for prev/next pointers)
#define CHECK_MEMSZ(sz)	ASSERT((((sz)&7)==0) && ((sz)>=MIN_BLOCK_SIZE))

#ifdef __cplusplus
}
#endif
#endif // GURU_SRC_MMU32_H_
