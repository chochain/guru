/*! @file
  @brief
  GURU 32-bit memory management.

  <pre>
  Copyright (C) 2019 GreenII.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in GURU.

  </pre>
*/
#include <stdio.h>
#include <assert.h>
#include "alloc.h"
#include "alloc32.h"

// TLSF: Two-Level Segregated Fit allocator with O(1) time complexity.
// Layer 1st(f), 2nd(s) model, smallest block 16-bytes, 16-byte alignment
// TODO: multiple-pool, thread-safe
// semaphore
__GURU__ volatile U32 	_mutex_mem;

// memory pool
__GURU__ U8				*_memory_pool;

// free memory bitmap
__GURU__ U32 			_l1_map;								// use lower 24 bits
__GURU__ U16 			_l2_map[L1_BITS];						// use all 16 bits
__GURU__ free_block		*_free_list[FL_SLOTS];

//================================================================
// most significant bit that is set
__GURU__ __INLINE__ U32
__fls(U32 x)
{
	U32 n;
	asm("bfind.u32 %0, %1;\n\t" : "=r"(n) : "r"(x));
	return n;
}

// least significant bit that is set
__GURU__ __INLINE__ U32
__ffs(U32 x)
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
  @retval int		index of free_blocks
*/
__GURU__ U32
__idx(U32 sz, U32P l1, U32P l2)
{
	U32 v = __fls(sz);
	U32 x = __ffs(sz);

    *l1 = v<BASE_BITS ? 0 : v - BASE_BITS;			// 1st level index
    *l2 = (sz >> (v - MN_BITS)) & L2_MASK;  // 2nd level index (with lower bits)

    return INDEX(*l1, *l2);
}

//================================================================
/*! wipe the free_block from linked list

  @param  blk	pointer to free block.
*/
__GURU__ void
__unmap(free_block *blk)
{
	assert(IS_FREE(blk));					// ensure block is free

    if (blk->prev) {						// down link exists
    	// blk->prev->next = blk->next;
    	PREV_FREE(blk)->next = U8POFF(NEXT_FREE(blk), PREV_FREE(blk));
    }
    else {			// top of the link, clear the map first (i.e. make available)
        U32 l1, l2;
        U32 index = __idx(blk->bsz, &l1, &l2);

        if ((_free_list[index]=NEXT_FREE(blk))==NULL) {
        	CLEAR_MAP(index);				// mark as unallocated
        }
    }
    if (blk->next) {						// up link
    	// blk->next->prev = blk->prev;
    	NEXT_FREE(blk)->prev = U8POFF(PREV_FREE(blk), NEXT_FREE(blk));
    }
    blk->next = blk->prev = 0;				// wipe for debugging
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
	assert((free_block*)BLK_AFTER(b0)==b1);
	assert(IS_FREE(b1));

	// remove b0, b1 from free list first (sizes will not change)
    __unmap(b1);

	// merge p0 and p1
	used_block *b2 = (used_block *)BLK_AFTER(b1);
	b2->psz += b1->psz & ~FREE_FLAG;	// watch for the block->flag
    b0->bsz += b1->bsz;					// include the block header

#if GURU_DEBUG
    *((U64*)b1) = 0xeeeeeeeeeeeeeeee;	// wipe b1 header
#endif
}

//================================================================
/*! Mark that block free and register it in the free index table.

  @param  blk	Pointer to block to be freed.

  TODO: check thread safety
*/
__GURU__ void
_mark_free(free_block *blk)
{
	assert(IS_USED(blk));

	U32 l1, l2;
	U32 index = __idx(blk->bsz, &l1, &l2);

    U32 l1x= L1(index);
    U32 l2x= L2(index);
    U32 t1 = TIC(l1x);
    U32 t2 = TIC(l2x);
    U32 m1 = L1_MAP(index);
    U16 m2 = L2_MAP(index);

    SET_MAP(index);										// set free block available ticks

    U32 m1x = L1_MAP(index);
    U16 m2x = L2_MAP(index);

    // update block attributes
    free_block *head = _free_list[index];

    assert(head != blk);

    SET_FREE(blk);
    blk->next = head ? U8POFF(head, blk) : 0;			// setup linked list
    blk->prev = 0;
    if (head) {											// non-end block, add backward link
    	head->prev = U8POFF(blk, head);
    }
    _free_list[index] = blk;							// new head of the linked list
}

__GURU__ free_block*
_mark_used(U32 index)
{
    free_block *blk  = _free_list[index];
    assert(blk);
    assert(IS_FREE(blk));

    free_block *next = NEXT_FREE(blk);
    if (next) {											// next free block exists
        free_block *prev = PREV_FREE(blk);
    	next->prev = prev ? U8POFF(prev, next) : 0;		// up link
        _free_list[index] = next;						// take it out of free list
    }
    else {
        U32 l1x= L1(index);
        U32 l2x= L2(index);
        U32 t1 = TIC(l1x);
        U32 t2 = TIC(l2x);
        U32 m1 = L1_MAP(index);
        U16 m2 = L2_MAP(index);

        CLEAR_MAP(index);						// release the index

        U32 m1x = L1_MAP(index);
        U16 m2x = L2_MAP(index);

        if (L1_MAP(index)==0 && L2_MAP(index)==0) {
        	_free_list[index] = NULL;
        }
    }
    SET_USED(blk);

    return blk;
}

__GURU__ void
_merge_with_next(free_block *b1)
{
	free_block *b2 = (free_block *)BLK_AFTER(b1);
	while (b2 && IS_FREE(b2)) {
		__pack(b1, b2);
		b2 = (free_block *)BLK_AFTER(b1);	// try the already expanded block again
	}
}

__GURU__ free_block*
_merge_with_prev(free_block *b1)
{
    free_block *b0 = (free_block *)BLK_BEFORE(b1);
	if (IS_USED(b0)) return b1;

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
    U32 index = __idx(sz, &l1, &l2);	// find free_list index by size

    if (_free_list[index]) return index;		// free block available, use it

    // no previous block exist, create a new one
    U32 avl = _l2_map[l1];			    // check any 2nd level available
    if (avl) {
    	l2 = __fls(avl);				// get first available l2 index
    }
    else if ((avl = _l1_map)) {			// check if 1st level available
        l1 = __fls(avl);        		// allocate new 1st & 2nd level indices
        l2 = __fls(_l2_map[l1]);
    }
    else return -1;						// out of memory
    return INDEX(l1, l2);               // index to freelist head
}

//================================================================
/*! Split free block by size (before allocating)

  @param  blk	pointer to free block
  @param  size	storage size
*/
__GURU__ void
_split(free_block *blk, U32 bsz)
{
	assert(IS_USED(blk));

    if ((bsz + MIN_BLOCK) > blk->bsz) return;	 				// too small to split 											// too small to split

    // split block, free
    free_block *free = (free_block *)U8PADD(blk, bsz);			// future next block (i.e. alot bsz bytes)
    free_block *aft  = (free_block *)BLK_AFTER(blk);			// next adjacent block

    free->bsz = blk->bsz - bsz;									// carve out the acquired block
    free->psz = U8POFF(free, blk);								// positive offset to previous block

    if (aft) {
        aft->psz = U8POFF(aft, free);							// backward offset (positive)
        _merge_with_next(free);									// _combine if possible
    }
    _mark_free(free);			// add to free_list and set (free, tail, next, prev) fields

    blk->bsz  = bsz;											// reduce size
}

//================================================================
/*! initialize

  @param  ptr	pointer to free memory block.
  @param  size	size. (max 4G)
*/
__GURU__ void
_init_mmu(void *mem, U32 size)
{
    assert(size > 0);

    U32 bsz = size - sizeof(free_block);

    _mutex_mem	 = 0;
    _memory_pool = (U8P)mem;

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

    CHECK_ALIGN(bsz);							// assume caller already align the size
    CHECK_MINSZ(bsz);							// check minimum allocation size

	MUTEX_LOCK(_mutex_mem);

	U32 index 		= _find_free_index(bsz);
	free_block *blk = _mark_used(index);		// take the indexed block off free list

	_split(blk, bsz);							// allocate the block, free up the rest
#if GURU_DEBUG
    U32P p = (U32P)BLK_DATA(blk);				// point to raw space allocated
    for (U32 i=0; i < sz>>2; i++) *p++ = 0xaaaaaaaa;
#endif
	MUTEX_FREE(_mutex_mem);

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
	U32 bsz = sz + sizeof(used_block);						// include the header

	CHECK_NULL(p0);
	CHECK_ALIGN(bsz);

    used_block *blk = (used_block *)BLK_HEAD(p0);
    assert(IS_USED(blk));									// make sure it is used

    if (bsz > blk->bsz) {
    	_merge_with_next((free_block *)blk);				// try to get the block bigger
    }
    if (bsz == blk->bsz) return p0;							// same size, good fit
    if (bsz < blk->bsz) {									// a little to big, split if we can
        _split((free_block *)blk, bsz);						// keep only the first bsz bytes
        return p0;
    }

    // not big enough block found, new alloc and deep copy
    void *p1 = guru_alloc(sz);
    memcpy(p1, p0, sz);										// deep copy

    guru_free(p0);											// reclaim block

    return p1;
}

//================================================================
/*! release memory
*/
__GURU__ void
guru_free(void *ptr)
{
	MUTEX_LOCK(_mutex_mem);

    free_block *blk = (free_block *)BLK_HEAD(ptr);			// get block header

    _merge_with_next(blk);
    _mark_free(blk);

    // the block is free now, try to merge a free block before if exists
    blk = _merge_with_prev(blk);

#if GURU_DEBUG
    if (BLK_AFTER(blk)) {
    	U32 *p = (U32*)U8PADD(blk, sizeof(free_block));
    	U32 sz = (blk->bsz - sizeof(free_block))>>2;
    	for (U32 i=0; i< (sz>32 ? 32 : sz); i++) *p++=0xffffffff;
    }
#endif

    MUTEX_FREE(_mutex_mem);
}

//================================================================
/*! release memory, vm used.

  @param  vm	pointer to VM.
*/
__GURU__ void
guru_memory_clr()
{
    used_block *p = (used_block *)_memory_pool;
    while (p) {
    	if (IS_USED(p)) {
    		guru_free(BLK_DATA(p));		// pointer to raw space
    	}
    	p = (used_block *)BLK_AFTER(p);
    }
}

#if GURU_DEBUG
//================================================================
/*! statistics

  @param  *total	returns total memory.
  @param  *used		returns used memory.
  @param  *free		returns free memory.
  @param  *fragment	returns memory fragmentation
*/
__GPU__ void
_alloc_stat(U32 v[])
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	U32 total=0, nfree=0, free=0, nused=0, used=0, nblk=0, nfrag=0;

	used_block *p = (used_block *)_memory_pool;

	U32 flag = IS_FREE(p);				// starting block type
	while (p) {	// walk the memory pool
		if (flag != IS_FREE(p)) {       // supposed to be merged
			nfrag++;
			flag = IS_FREE(p);
		}
		total += p->bsz;
		nblk  += 1;
		if (IS_FREE(p)) {
			nfree += 1;
			free  += p->bsz;
		}
		else {
			nused += 1;
			used  += p->bsz;
		}
		p = (used_block *)BLK_AFTER(p);
	}
	v[0] = total + sizeof(free_block);
	v[1] = nfree;
	v[2] = free;
	v[3] = nused;
	v[4] = used;
	v[5] = nblk;
	v[6] = nfrag;

	__syncthreads();
}

__GPU__ void
guru_memory_init(void *ptr, U32 sz)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	_init_mmu(ptr, sz);
}

__HOST__ void*
cuda_malloc(U32 sz, U32 type)
{
	void *mem;

	switch (type) {
	case 0: 	cudaMalloc(&mem, sz); break;			// allocate device memory
	default: 	cudaMallocManaged(&mem, sz);			// managed (i.e. paged) memory
	}
    if (cudaSuccess != cudaGetLastError()) return NULL;

    return mem;
}

__HOST__ void
_get_alloc_stat(U32 stat[])
{
	U32P v;
	cudaMallocManaged(&v, 8*sizeof(int));				// allocate host memory

	_alloc_stat<<<1,1>>>(v);
	cudaDeviceSynchronize();

	for (U32 i=0; i<8; i++) {
		stat[i] = v[i];									// mirror stat back from device
	}
	cudaFree(v);
}

__HOST__ void
guru_dump_alloc_stat(U32 trace)
{
	if (trace==0) return;

	U32 s[8];
	_get_alloc_stat(s);

	printf("\tmem=%d(0x%x): free=%d(0x%x), used=%d(0x%x), nblk=%d, nfrag=%d, %d%% allocated\n",
			s[0], s[0], s[1], s[2], s[3], s[4], s[5], s[6], (int)(100*(s[4]+1)/s[0]));
}
#endif
