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
  U32 bsz;   						// lower 2 bits used as flags
} used_block;

typedef struct free_block {			// 4-bytes (+ 4 bytes free space)
  U32 bsz;   						//
  U32 poff;							// positive offset to previous block
}
*/
typedef struct used_block {			// 8-bytes
	U32 bsz; 						//!< block size, header included (max 2G)
	union {
		U32 psz	:   32;  			//!< prior adjacent memory block size
		U32 flag: 	3;				// lower 4 bit might be used
	};
} used_block;

typedef struct free_block {			// 16-bytes (i.e. mininum allocation per block)
	U32 bsz; 						//!< block size, header included (max 2G)
	union {
		U32 psz	:   32;  			//!< prior adjacent memory block size
		U32 flag: 	3;				// lower 4 bit might be used
	};

	S32 next;						// offset to next free block
	S32 prev;						// offset to previous free block
} free_block;

#define FREE_FLAG		0x1
#define IS_FREE(b)		((b)->flag & FREE_FLAG)
#define IS_USED(b)		(!IS_FREE(b))
#define SET_FREE(b)		((b)->flag |=  FREE_FLAG)
#define SET_USED(b)		((b)->flag &= ~FREE_FLAG)

#define NEXT_FREE(b)	((free_block*)((b)->next ? ((b)->next<0 ? U8PSUB((b), -(b)->next) : U8PADD((b), (b)->next)) : NULL))
#define PREV_FREE(b)	((free_block*)((b)->prev ? ((b)->prev<0 ? U8PSUB((b), -(b)->prev) : U8PADD((b), (b)->prev)) : NULL))

#define BLK_AFTER(b) 	((b)->bsz 			   ? U8PADD(b, (b)->bsz) 				: NULL)		// following adjacent memory block
#define BLK_BEFORE(b) 	(((b)->psz&~FREE_FLAG) ? U8PSUB(b, ((b)->psz&~FREE_FLAG)) 	: NULL)		// prior adjacent memory block
#define BLK_DATA(b) 	(U8PADD(b, sizeof(used_block)))		// pointer to raw data space
#define BLK_HEAD(p) 	(U8PSUB(p, sizeof(used_block)))		// pointer block given raw data pointer

#define MN_BITS			4									// 00000000 00000000 00000000 0000XXXX  // 16-bytes smallest blocksize
#define L2_BITS     	3   								// 00000000 00000000 00000000 0XXX0000  // 8 entires
#define L1_BITS     	10									// 00000000 00000000 00XXXXXX X0000000  // 7 levels (for 16K)
#define BASE_BITS   	(L2_BITS+MN_BITS)

#define TIC(n)      	(1 << n)
#define L2_MASK 		((1<<L2_BITS)-1)
#define MIN_BLOCK		(1 << MN_BITS)

#define L1(i) 			((i) >> L2_BITS)
#define L2(i) 			((i) & L2_MASK)
#define FL_SLOTS		(L1_BITS * (1 << L2_BITS))			// slots for free_list pointers (10*8 entries for 16K memory)

#define L1_MAP(i)       (_l1_map)
#define L2_MAP(i)       (_l2_map[L1(i)])
#define INDEX(l1, l2)   ((l1<<L2_BITS) | l2)



#define SET_L1(i)		(L1_MAP(i) |= TIC(L1(i)))
#define CLR_L1(i)	    (L1_MAP(i) &= ~TIC(L1(i)))
#define SET_L2(i)	    (L2_MAP(i) |= TIC(L2(i)))
#define CLR_L2(i)		(L2_MAP(i) &= ~TIC(L2(i)))
#define SET_MAP(i)      { SET_L1(i); SET_L2(i); }
#define CLEAR_MAP(i)	{ CLR_L2(i); if ((L2_MAP(i))==0) CLR_L1(i); }

#define CHECK_MINSZ(sz)	assert((sz)>=MIN_BLOCK)

#ifdef __cplusplus
}
#endif
#endif
