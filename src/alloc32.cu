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

// TLSF: Two-Level Segregated Fit allocator with O(1) time complexity.
// Layer 1st(f), 2nd(s) model, smallest block 16-bytes, 16-byte alignment
// TODO: multiple-pool, thread-safe

#ifndef L1_BITS

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

#define NEXT(p) 	U8PADD(p, p->size)
#define PREV(p) 	U8PSUB(p, p->poff)						// poff is a positive number
#define CHECK_MINSZ(sz)	assert((sz)>=MIN_BLOCK)

#endif

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
__release(free_block *blk)
{
    if (blk->prev==NULL) {			// head of linked list?
    	U32 l1, l2;
        U32 index = __idx(blk->size, &l1, &l2);

        if ((_free_list[index]=blk->next)==NULL) {
            CLEAR_MAP(index);			// mark as unallocated
        }
    }
    else {								// down link
        blk->prev->next = blk->next;
    }
    if (blk->next != NULL) {			// up link
        blk->next->prev = blk->prev;
    }
}

//================================================================
/*! merge ptr1 and ptr2 block.
  ptr2 will disappear

  @param  ptr1	pointer to free block 1
  @param  ptr2	pointer to free block 2
*/
__GURU__ void
__merge(free_block *p0, free_block *p1)
{
    assert(p0 < p1);

    // merge ptr1 and ptr2
    p0->tail  = p1->tail;
    p0->size += p1->size;

    // update block info
    if (!p0->tail) {
        free_block *next = (free_block *)NEXT(p0);
        next->poff = U8POFF(next, p0);
    }
#if GURU_DEBUG
    *((U64*)p1) = 0xeeeeeeeeeeeeeeee;
#endif
}

__GURU__ free_block*
_merge_with_next(free_block *blk)
{
	if (blk->tail) return blk;

	free_block *next = (free_block *)NEXT(blk);

	if (!next->free) return blk;

	__release(next);
	__merge(blk, next);

	return blk;
}

__GURU__ free_block*
_merge_with_prev(free_block *blk)
{
	free_block *prev = (free_block *)PREV(blk);

	if (prev && prev->free) {			// merge with previous, needed?
		__release(prev);
		__merge(prev, blk);
		blk = prev;
	}
	return blk;
}

//================================================================
/*! Mark that block free and register it in the free index table.

  @param  blk	Pointer to block to be freed.

  TODO: check thread safety
*/
__GURU__ void
_mark_free(free_block *blk)
{
	U32 l1, l2;
    U32 index = __idx(blk->size, &l1, &l2);

    U32 l1x= L1(index);
    U32 l2x= L2(index);
    U32 t1 = TIC(l1x);
    U32 t2 = TIC(l2x);
    U32 m1 = L1_MAP(index);
    U16 m2 = L2_MAP(index);

    SET_MAP(index);							// set free block available ticks

    U32 m1x = L1_MAP(index);
    U16 m2x = L2_MAP(index);

    free_block *head = _free_list[index];

    blk->free = 1;
    blk->next = head;					// setup linked list
    blk->prev = NULL;
    if (head) {								// non-end block, add backward link
    	head->prev = blk;
    }

    _free_list[index] = blk;				// new head of the linked list
}

__GURU__ free_block*
_mark_used(U32 index)
{
    free_block *blk = _free_list[index];

    CHECK_NULL(blk);

    if (blk->next==NULL) {					// top of linked list
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
    else {
        _free_list[index] = blk->next;		// follow the linked list
    	blk->next->prev = blk->prev;		// 20190819 CC: is this necessary?
    }
    blk->free = 0;

    return blk;
}

//================================================================
/*! Find index to a free block

  @param  size	size
  @retval -1	not found
  @retval index to available _free_list
*/
__GURU__ S32
_find_free_block(U32 sz)
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
/*! Split block by size

  @param  blk	pointer to free block
  @param  size	storage size
*/
__GURU__ void
_split_free_block(free_block *blk, U32 sz)
{
	U32 blk_sz = sz + sizeof(used_block);						// add header overhead
    if (blk->size < (blk_sz + MIN_BLOCK)) return; 				// too small to split 											// too small to split

    // split block, free
    free_block *free = (free_block *)U8PADD(blk, blk_sz);		// future next block
    free_block *next = (free_block *)NEXT(blk);					// current next

    free->size   = blk->size - blk_sz;							// carve out the block
    free->poff   = U8POFF(free, blk);							// positive offset to previous block
    free->tail   = blk->tail;
    free->free   = 1;

    if (!free->tail) {
        next->poff = U8POFF(next, free);						// offset (positive)
    }
    _mark_free(free);											// add to free_list

    blk->size = sz;												// reduce size
    blk->tail = 0;
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

    _mutex_mem		  = 0;
    _memory_pool      = (U8P)mem;
    _memory_pool_size = size;

    // initialize entire memory pool as the first block
    free_block *blk  = (free_block *)_memory_pool;
    blk->tail = 1;
    blk->free = 1;
    blk->size = _memory_pool_size;
    blk->prev = 0;

    _mark_free(blk);
}

//================================================================
/*! allocate memory

  @param  size	request storage size.
  @return void* pointer to a guru memory block.
*/
__GURU__ void*
guru_alloc(U32 sz)
{
    U32 blk_sz = sz + sizeof(used_block);

    CHECK_ALIGN(sz);
    CHECK_MINSZ(blk_sz);			// check minimum alloc size

	MUTEX_LOCK(_mutex_mem);

	U32 index 		= _find_free_block(blk_sz);
	free_block *blk = _mark_used(index);

#if GURU_DEBUG
    U32P p = (U32P)BLOCKDATA(blk);
    for (U32 i=0; i < sz>>2; i++) *p++ = 0xaaaaaaaa;
#endif
	_split_free_block(blk, sz);

	MUTEX_FREE(_mutex_mem);

	return BLOCKDATA(blk);
}

//================================================================
/*! re-allocate memory

  @param  ptr	Return value of raw malloc()
  @param  size	request size
  @return void* pointer to allocated memory.
*/
__GURU__ void*
guru_realloc(void *ptr, U32 sz)
{
	CHECK_NULL(ptr);
	CHECK_ALIGN(sz);

    used_block *blk = (used_block *)BLOCKHEAD(ptr);
    assert(!blk->free);										// tyr to be careful

    if (sz > blk->size) {
    	_merge_with_next((free_block *)blk);				// try to get the block bigger
    }
    if (sz == blk->size) return ptr;						// same size, good fit
    if (sz < blk->size) {									// a little to big, split if we can
    	U32 blk_sz = sz + sizeof(used_block);
        _split_free_block((free_block *)blk, blk_sz);
        return ptr;											// return the same memory pointer
    }

    // not big enough block found, new alloc and deep copy
    void *nptr = guru_alloc(sz);

    U8 *s = (U8 *)ptr;
    U8 *d = (U8 *)nptr;
    for (U32 i=0; i<blk->size; i++) *d++ = *s++;			// deep copy

    guru_free(ptr);											// reclaim block

    return nptr;
}

//================================================================
/*! release memory
*/
__GURU__ void
guru_free(void *ptr)
{
	MUTEX_LOCK(_mutex_mem);

    free_block *blk = (free_block *)BLOCKHEAD(ptr);	// get block header

    blk = _merge_with_next(blk);
    blk = _merge_with_prev(blk);

    _mark_free(blk);

    MUTEX_FREE(_mutex_mem);
}

//================================================================
/*! release memory, vm used.

  @param  vm	pointer to VM.
*/
__GURU__ void
guru_memory_clear()
{
    used_block *p = (used_block *)_memory_pool;
    while (1) {
    	if (!p->free) {
    		guru_free(BLOCKDATA(p));
    	}
    	if (p->tail) break;
    	p = (used_block *)NEXT(p);
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

	U32 flag = p->free;				// starting block type
	while (1) {	// walk the memory pool
		if (flag != p->free) {       // supposed to be merged
			nfrag++;
			flag = p->free;
		}
		total += p->size;
		nblk  += 1;
		if (p->free) {
			nfree += 1;
			free  += p->size;
		}
		else {
			nused += 1;
			used  += p->size;
		}
		if (p->tail) break;
		p = (used_block *)NEXT(p);
	}
	v[0] = total;
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
guru_malloc(U32 sz, U32 type)
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
_get_malloc_stat(U32 stat[])
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
	_get_malloc_stat(s);

	printf("\tmem=%d(0x%x): free=%d(0x%x), used=%d(0x%x), nblk=%d, nfrag=%d, %d%% allocated\n",
			s[0], s[0], s[1], s[2], s[3], s[4], s[5], s[6], (int)(100*(s[4]+1)/s[0]));
}
#endif
