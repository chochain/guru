/*! @file
  @brief
  GURU 32-bit memory management.

  <pre>
  Copyright (C) 2019 GreenII.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in GURU.

  </pre>
*/
#include "guru_config.h"
#include "guru.h"
#include "util.h"
#include "mmu.h"
#include "mmu32.h"

// TLSF: Two-Level Segregated Fit allocator with O(1) time complexity.
// Layer 1st(f), 2nd(s) model, smallest block 16-bytes, 16-byte alignment
// TODO: multiple-pool, thread-safe
// semaphore
#define _LOCK			{ MUTEX_LOCK(_mutex_mem); }
#define _UNLOCK			{ MUTEX_FREE(_mutex_mem); }

// memory pool
__GURU__ volatile U32 	_mutex_mem;
__GURU__ U8				*_memory_pool;

// free memory bitmap
__GURU__ U32 			_l1_map;								// use lower 24 bits
__GURU__ U16 			_l2_map[L1_BITS];						// 8-bit now, TODO: 16-bit?
__GURU__ free_block		*_free_list[FL_SLOTS];

//================================================================
// MMU JTAG sanity check - memory pool walker
//
#if GURU_DEBUG
__GURU__ int
__mmu_ok()											// mmu sanity check
{
	used_block *p0 = (used_block*)_memory_pool;
	used_block *p1 = (used_block*)BLK_AFTER(p0);
	U32 tot = sizeof(free_block);
	while (p1) {
		if (p0->bsz != (p1->psz&~FREE_FLAG)) {
			return 0;
		}
		tot += p0->bsz;
		p0  = p1;
		p1  = (used_block*)BLK_AFTER(p0);
	}
	if (tot!=BLOCK_MEMORY_SIZE) {
		return 0;
	}
	return (!p1 && tot==BLOCK_MEMORY_SIZE);
}

#define MMU_CHECK		ASSERT(__mmu_ok())
//#define MMU_CHECK		__mmu_ok()
#else
#define MMU_CHECK
#endif // CC_DEBUG

//================================================================
// most significant bit that is set
// __xls(i) = 32-__ffs(__brev(i))
//
__GURU__ __INLINE__ U32
__xls(U32 x)
{
	U32 n;
	asm("bfind.u32 %0, %1;\n\t" : "=r"(n) : "r"(x));
	return n;
}
// least significant bit that is set
// __xfs(i) = __ffs(i)+1;
//
__GURU__ __INLINE__ U32
__xfs(U32 x)
{
	U32 n;
	asm(
		"brev.b32 %0, %1;\n\t"
		"clz.b32 %0, %0;\n\t"
		: "=r"(n) : "r"(x)
	);
	return n;
}
//================================================================
/*! calc f and s, and returns fli,sli of free_blocks

  @param  alloc_size	alloc size
  @retval int			index of free_blocks
*/
__GURU__ U32
__idx(U32 sz, U32 *l1, U32 *l2)
{
	U32 v = __xls(sz);
	*l1 = v<BASE_BITS ? 0 : v - BASE_BITS + 1;	// 1st level index

	U32 n = *l1<2 ? 0 : *l1 - 1;				// down shifting bit
    *l2 = (sz >> (n+MN_BITS)) & L2_MASK; 		// 2nd level index (with lower bits)

    return INDEX(*l1, *l2);
}

//================================================================
/*! wipe the free_block from linked list

  @param  blk	pointer to free block.
*/
__GURU__ void
__unmap(free_block *blk)
{
	ASSERT(IS_FREE(blk));						// ensure block is free

	U32 l1, l2;
	U32 index 	  = __idx(blk->bsz, &l1, &l2);
	free_block *n = _free_list[index] = NEXT_FREE(blk);
    if (n) {									// up link
    	// blk->next->prev = blk->prev;
    	n->prev = blk->prev ? U8POFF(n, PREV_FREE(blk)) : 0;
    	ASSERT((n->prev&7)==0);
    }
    else {										// 1st of the link
        U32 l1x= L1(index);
        U32 l2x= L2(index);
        U32 t1 = TIC(l1x);
        U32 t2 = TIC(l2x);
        U32 m1 = L1_MAP(index);
        U16 m2 = L2_MAP(index);

        CLEAR_MAP(index);						// clear the index bit

        U16 m2x = L2_MAP(index);
        U32 m1x = L1_MAP(index);
        t1 = m1x;
    }
    if (blk->prev) {							// down link
    	free_block *p = PREV_FREE(blk);
    	// blk->prev->next = blk->next;
    	p->next = blk->next ? U8POFF(n, p) : 0;
    }
    blk->next = blk->prev = 0xeeeeeeee;			// wipe for debugging

    MMU_CHECK;
}

//================================================================
/*! merge p0 and p1 adjacent free blocks.
  ptr2 will disappear

  @param  ptr1	pointer to free block 1
  @param  ptr2	pointer to free block 2
*/
__GURU__ void
__pack(free_block *b0, free_block *b1)
{
	ASSERT((free_block*)BLK_AFTER(b0)==b1);
	ASSERT(IS_FREE(b1));

	// remove b0, b1 from free list first (sizes will not change)
    __unmap(b1);

	// merge b0 and b1, retain b0.FREE_FLAG
	used_block *b2 = (used_block *)BLK_AFTER(b1);
	b2->psz += b1->psz & ~FREE_FLAG;	// watch for the block->flag
    b0->bsz += b1->bsz;					// include the block header

#if GURU_DEBUG
    *((U64*)b1) = 0xeeeeeeeeeeeeeeee;	// wipe b1 header
#endif
    MMU_CHECK;
}

//================================================================
/*! Mark that block free and register it in the free index table.

  @param  blk	Pointer to block to be freed.

  TODO: check thread safety
*/
__GURU__ void
_mark_free(free_block *blk)
{
	ASSERT(IS_USED(blk));

	U32 l1, l2;
	U32 index = __idx(blk->bsz, &l1, &l2);

    U32 l1x= L1(index);
    U32 l2x= L2(index);
    U32 t1 = TIC(l1x);
    U32 t2 = TIC(l2x);
    U32 m1 = L1_MAP(index);
    U16 m2 = L2_MAP(index);

    SET_MAP(index);								// set free block available ticks

    U32 m1x = L1_MAP(index);
    U16 m2x = L2_MAP(index);

    // update block attributes
    free_block *head = _free_list[index];

    ASSERT(head!=blk);

    SET_FREE(blk);
    blk->next = head ? U8POFF(head, blk) : 0;	// setup linked list
    ASSERT((blk->next&7)==0);
    blk->prev = 0;
    if (head) {									// non-end block, add backward link
    	head->prev = U8POFF(blk, head);
        ASSERT((head->prev&7)==0);
    	SET_FREE(head);							// turn the free flag back on
    }
    _free_list[index] = blk;					// new head of the linked list
}

__GURU__ free_block*
_mark_used(U32 index)
{
    free_block *blk  = _free_list[index];
    ASSERT(blk);
    ASSERT(IS_FREE(blk));

    __unmap(blk);
    SET_USED(blk);

    return blk;
}

__GURU__ void
_merge_with_next(free_block *b0)
{
	free_block *b1 = (free_block *)BLK_AFTER(b0);
	while (b1 && IS_FREE(b1) && b1->bsz!=0) {
		__pack(b0, b1);
		b1 = (free_block *)BLK_AFTER(b0);	// try the already expanded block again
	}
}

__GURU__ free_block*
_merge_with_prev(free_block *b1)
{
    free_block *b0 = (free_block *)BLK_BEFORE(b1);
	if (b0==NULL || IS_USED(b0)) return b1;

	__unmap(b0);							// take it out of free_list before merge
	__pack(b0, b1);							// take b1 out and merge with b0

	SET_USED(b0);							// _mark_free assume b0 to be a USED block
	_mark_free(b0);

    return b0;
}

//================================================================
/*! Find index to a free block

  @param  size	size
  @retval -1	not found
  @retval index to available _free_list
*/
__GURU__ S32
_find_free_index(U32 sz)
{
	U32 l1, l2;
    U32 index = __idx(sz, &l1, &l2);			// find free_list index by size

    if (_free_list[index]) return index;		// free block available, use it

    // no previous block exist, create a new one
    U32 avl = _l2_map[l1];			    		// check any 2nd level available
    if (avl >> l2) {
    	l2 = __xls(avl);						// get first available l2 index
    }
    else if ((avl = _l1_map)) {					// check if 1st level available
        l1 = __xls(avl);        				// allocate new 1st & 2nd level indices
        l2 = __xls(_l2_map[l1]);
    }
    else return -1;								// out of memory
    return INDEX(l1, l2);               		// index to freelist head
}

//================================================================
/*! Split free block by size (before allocating)

  @param  blk	pointer to free block
  @param  size	storage size
*/
__GURU__ void
_split(free_block *blk, U32 bsz)
{
	ASSERT(IS_USED(blk));

    if ((bsz + MIN_BLOCK + sizeof(free_block)) > blk->bsz) return;	// too small to split 											// too small to split

    // split block, free
    free_block *free = (free_block *)U8PADD(blk, bsz);	// future next block (i.e. alot bsz bytes)
    free_block *aft  = (free_block *)BLK_AFTER(blk);	// next adjacent block

    free->bsz = blk->bsz - bsz;							// carve out the acquired block
    free->psz = U8POFF(free, blk);						// positive offset to previous block
    blk->bsz  = bsz;									// allocate target block

    if (aft) {
        aft->psz = U8POFF(aft, free)|(aft->psz&FREE_FLAG);	// backward offset (positive)
        _merge_with_next(free);								// _combine if possible
    }
    _mark_free(free);			// add to free_list and set (free, tail, next, prev) fields

    MMU_CHECK;
}

//================================================================
/*! initialize

  @param  ptr	pointer to free memory block.
  @param  size	size. (max 4G)
*/
__GURU__ void
_init_mmu(void *mem, U32 size)
{
    ASSERT(size > 0);

    U32 bsz = size - sizeof(free_block);

    _mutex_mem	 = 0;
    _memory_pool = (U8*)mem;

    // initialize entire memory pool as the first block
    free_block *blk  = (free_block *)_memory_pool;
    blk->bsz = bsz;						// 1st (big) block
    blk->psz = 0;
    SET_USED(blk);

    _mark_free(blk);					// will set free, tail, next, prev

    blk = (free_block *)BLK_AFTER(blk);	// last block
    blk->bsz = blk->next = blk->prev = 0;
    blk->psz = bsz;
    SET_USED(blk);

    MMU_CHECK;
}

//================================================================
/*! allocate memory

  @param  size	request storage size.
  @return void* pointer to a guru memory block.
*/
__GURU__ void*
guru_alloc(U32 sz)
{
    U32 bsz = sz + sizeof(used_block);			// logical => physical size
    CHECK_MEMSZ(bsz);							// check alignment & sizing

    _LOCK;
	U32 index 		= _find_free_index(bsz);
	free_block *blk = _mark_used(index);		// take the indexed block off free list

	_split(blk, bsz);							// allocate the block, free up the rest
#if GURU_DEBUG
    U32 *p = (U32*)BLK_DATA(blk);				// point to raw space allocated
    sz >>= 2;
    for (U32 i=0; i < (sz>16 ? 16 : sz); i++) *p++ = 0xaaaaaaaa;
#endif
	_UNLOCK;

	return BLK_DATA(blk);						// pointer to raw space
}

//================================================================
/*! re-allocate memory

  @param  ptr	Return value of raw malloc()
  @param  size	request size
  @return void* pointer to allocated memory.
*/
__GURU__ void*
guru_realloc(void *p0, U32 sz)
{
	ASSERT(p0);

	U32 bsz = sz + sizeof(used_block);					// include the header
	CHECK_MEMSZ(bsz);									// assume it is aligned already

    used_block *blk = (used_block *)BLK_HEAD(p0);
    ASSERT(IS_USED(blk));								// make sure it is used

    if (bsz > blk->bsz) {
    	_merge_with_next((free_block *)blk);			// try to get the block bigger
    }
    if (bsz == blk->bsz) return p0;						// fits right in
    if (bsz < blk->bsz) {								// enough space now
    	if ((blk->bsz - bsz) > (sizeof(used_block)+MIN_BLOCK)) {	// but is it too big?
    		_split((free_block*)blk, bsz);				// allocate the block, free up the rest
    	}
    	return p0;
    }
    // not big enough block found, new alloc and deep copy
    void *p1 = guru_alloc(bsz);
    MEMCPY(p1, p0, sz);									// deep copy, !!using CUDA provided memcpy

    guru_free(p0);										// reclaim block

    return p1;
}

__GURU__ GV*
guru_gv_alloc(U32 n)
{
	return (GV*)guru_alloc(sizeof(GV) * n);
}

__GURU__ GV*
guru_gv_realloc(GV *gv, U32 n)
{
	return (GV*)guru_realloc(gv, sizeof(GV) * n);
}

//================================================================
/*! release memory
*/
__GURU__ void
guru_free(void *ptr)
{
	_LOCK;
    free_block *blk = (free_block *)BLK_HEAD(ptr);			// get block header

    _merge_with_next(blk);
#if GURU_DEBUG
    if (BLK_AFTER(blk)) {
    	U32 *p = (U32*)U8PADD(blk, sizeof(used_block));
    	U32 sz = blk->bsz ? (blk->bsz - sizeof(used_block))>>2 : 0;
    	for (U32 i=0; i< (sz>32 ? 32 : sz); i++) *p++=0xffffffff;
    }
#endif
    _mark_free(blk);

    // the block is free now, try to merge a free block before if exists
    blk = _merge_with_prev(blk);
    _UNLOCK;

    MMU_CHECK;
}

//================================================================
/*! release memory, vm used.

  @param  vm	pointer to VM.
*/
__GURU__ void
guru_mmu_clr()
{
	used_block *p = (used_block *)_memory_pool;
    while (p) {
    	if (IS_USED(p)) {
    		guru_free(BLK_DATA(p));		// pointer to raw space
    	}
    	p = (used_block *)BLK_AFTER(p);
    }
}

__GURU__ void
guru_mmu_stat(guru_mstat *s)
{
	used_block *p = (used_block *)_memory_pool;
	MEMSET(s, 0, sizeof(guru_mstat));	// wipe, !using CUDA provided memset

	U32 flag = IS_FREE(p);				// starting block type
	while (p) {							// walk the memory pool
		U32 bsz = p->bsz;				// current block size
		if (flag != IS_FREE(p)) {       // supposed to be merged
			s->nfrag++;
			flag = IS_FREE(p);
		}
		s->total += bsz;
		s->nblk  += 1;
		if (IS_FREE(p)) {
			s->nfree += 1;
			s->free  += bsz;
		}
		else {
			s->nused += 1;
			s->used  += bsz;
		}
		p = (used_block *)BLK_AFTER(p);
	}
	s->total    += sizeof(free_block);
	s->pct_used = (int)(100*(s->used+1)/s->total);
}

__GPU__ void
guru_mmu_init(void *ptr, U32 sz)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	_init_mmu(ptr, sz);
}


__HOST__ void*
cuda_malloc(U32 sz, U32 type)
{
	void *mem;

	// TODO: to add texture memory
	switch (type) {
	case 0: 	cudaMalloc(&mem, sz); break;			// allocate device memory
	default: 	cudaMallocManaged(&mem, sz);			// managed (i.e. paged) memory
	}
    if (cudaSuccess != cudaGetLastError()) return NULL;

    return mem;
}

__HOST__ void
cuda_free(void *mem) {
	cudaFree(mem);
}

#if !GURU_DEBUG
__HOST__ U32 guru_mmu_check(U32 trace);
#else
//================================================================
/*! statistics

  @param  *total	returns total memory.
  @param  *used		returns used memory.
  @param  *free		returns free memory.
  @param  *fragment	returns memory fragmentation
*/
#define bin2u32(x) ((x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24))

__GPU__ void
_dump_freelist()
{
	printf("%14s","");
	for (U32 i=0; i<FL_SLOTS; i++) {
		if (!_free_list[i]) continue;

		printf("%02x=>[", i);
		for (free_block *b = _free_list[i]; b!=NULL; b=NEXT_FREE(b)) {
			U32 a = (U32A)b;		// take 32-bits only, b is 64-bit will bleed into 2nd param
			ASSERT((a&7)==0);		// double check alignment
			printf(" %08x(%02x)", a, b->bsz);
			if (IS_USED(b)) {
				printf("<-USED?");
				break;				// something is wrong (link is broken here)
			}
		}
		printf(" ] ");
	}
	printf("\n");
}

__GPU__ void
_alloc_stat(guru_mstat *s)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	guru_mstat v;
	guru_mmu_stat(&v);

	*s = v;
}

__HOST__ U32
guru_mmu_check(U32 level)
{
	U32 err;

	if (level==0) return 0;
	if (level & 1) {
		guru_mstat *s;
		cudaMallocManaged(&s, sizeof(guru_mstat));				// allocate host memory

		_alloc_stat<<<1,1>>>(s);
		GPU_SYNC();

		printf("%14smem=%d(0x%x): free=%d(0x%x), used=%d(0x%x), nblk=%d, nfrag=%d, %d%% allocated\n",
			"", s->total, s->total, s->free, s->free, s->used, s->used, s->nblk, s->nfrag, s->pct_used);
		cudaFree(s);
	}
	if (level & 2) {
		_dump_freelist<<<1,1>>>();
		GPU_SYNC();
	}
	return 0;
}
#endif
