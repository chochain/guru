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
		 U8 			*guru_host_heap;						// guru host global memory
__GURU__ U8				*guru_device_heap;						// CUDA kernel global memory pool
__GURU__ U32 			_heap_size;
__GURU__ U32 			_mutex_mem;

// free memory bitmap
__GURU__ U32 			_l1_map;								// use lower 24 bits
__GURU__ U8 			_l2_map[L1_BITS];						// 8-bit, (16-bit requires too many FL_SLOTS)
__GURU__ free_block		*_free_list[FL_SLOTS];

#if GURU_DEBUG
//================================================================
/*! statistics

  @param  *total	returns total memory.
  @param  *used		returns used memory.
  @param  *free		returns free memory.
  @param  *fragment	returns memory fragmentation
*/
#define bin2u32(x) ((x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24))

__GURU__ void
_dump_freelist(const char *hdr, int sz)
{
	PRINTF("mmu#%6s(x%04x) L1=%03x: ", hdr, sz, _l1_map);
	for (int i=L1_BITS-1; i>=0; i--) { PRINTF("%02x%s", _l2_map[i], i%4==0 ? " " : ""); }
	for (int i=FL_SLOTS-1; i>=0; i--) {
		if (!_free_list[i]) continue;
		PRINTF("[%02x]=>[", i);
		for (free_block *b = _free_list[i]; b!=NULL; b=NEXT_FREE(b)) {
			U32 a = (U32A)b;		// when using b directly, higher bit will bleed into second parameter
			PRINTF(" %06x:%04x", a & 0xffffff, b->bsz);
			if (IS_USED(b)) {
				PRINTF("<-USED?");
				break;				// something is wrong (link is broken here)
			}
		}
		PRINTF(" ] ");
	}
	PRINTF("\n");
}

__GPU__ void
_mmu_freelist(U32 sz)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	_dump_freelist("check", sz);
}

//================================================================
// MMU JTAG sanity check - memory pool walker
//
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
	guru_mstat *s;

	if (level==0) return 0;

	cudaMallocManaged(&s, sizeof(guru_mstat));				// allocate host memory
	if (level & 1) {
		_alloc_stat<<<1,1>>>(s);
		GPU_SYNC();
		printf("%14smem=%d(0x%x): free=%d(0x%x), used=%d(0x%x), nblk=%d, nfrag=%d, %d%% allocated\n",
			"", s->total, s->total, s->free, s->free, s->used, s->used, s->nblk, s->nfrag, s->pct_used);
	}
	if (level & 2) {
		_mmu_freelist<<<1,1>>>(s->free);
		GPU_SYNC();
	}
	cudaFree(s);
	return 0;
}
#else
__HOST__ U32 guru_mmu_check(U32 level);
#endif // GURU_DEBUG

#if MMU_DEBUG
#define MMU_SCAN		ASSERT(__mmu_ok())
__GURU__ int
__mmu_ok()											// mmu sanity check
{
	used_block *p0 = (used_block*)guru_device_heap;
	used_block *p1 = (used_block*)BLK_AFTER(p0);
	U32 tot = sizeof(free_block);
	while (p1) {
		if (p0->bsz != (p1->psz&~FREE_FLAG)) {		// ERROR!
			return 0;								// memory integrity broken!
		}
		tot += p0->bsz;
		p0  = p1;
		p1  = (used_block*)BLK_AFTER(p0);
	}
#if CC_DEBUG
	if (tot!=_heap_size) {							// ERROR, tally off
		return 0;									// debug break point
	}
#endif // CC_DEBUG
	return (tot==_heap_size && !p1);				// last check
}
#else
#define MMU_SCAN
#endif // MMU_DEBUG

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

  mrbc:
    l1 = __xls(sz>>BASE_BITS);
    n  = (l1==0) ? (l1 + MN_BITS) : (l1 + MN_BITS - 1);
    l2 = (sz >> n) & L2_MASKS;

  original thesis:
    l1 = __xls(sz);
    l2 = (sz >> (l1 - MN_BITS)) - (1<<MN_BITS);
    l1 = __xls(sz);
    l2 = (sz ^ (1<<l1)) >> (l1 - L2_BITS);
*/
__GURU__ U32
__idx(U32 sz)
{
	U32 l1 = __xls(sz >> BASE_BITS) + 1;		// __xls returns -1 if no bit is set
	U32 l2 = (sz >> (l1 + MN_BITS - (l1!=0))) & L2_MASK;
#if MMU_DEBUG && CC_DEBUG
    PRINTF("mmu#__idx(%04x):       INDEX(%x,%x) => %x\n", sz, l1, l2, INDEX(l1, l2));
#endif // MMU_DEBUG && CC_DEBUG
    return INDEX(l1, l2);
}

//================================================================
/*! wipe the free_block from linked list

  @param  blk	pointer to free block.
*/
__GURU__ void
__unmap(free_block *blk)
{
	ASSERT(IS_FREE(blk));						// ensure block is free

	U32 index = __idx(blk->bsz);
    free_block *n = _free_list[index] = NEXT_FREE(blk);
    free_block *p = blk->prev ? PREV_FREE(blk) : NULL;
    if (n) {									// up link
    	// blk->next->prev = blk->prev;
    	if (blk->prev) {
    		n->prev = U8POFF(p, n);
    		ASSERT((n->prev&7)==0);
    		SET_FREE(n);
    	}
    	else n->prev = 0;
    }
    else {										// 1st of the link
        CLEAR_MAP(index);						// clear the index bit
    }
    if (blk->prev) {							// down link_l2_map[l2m2
    	// blk->prev->next = blk->next;
    	p->next = blk->next ? U8POFF(n, p) : 0;
    }
    blk->next = blk->prev = 0xeeeeeeee;			// wipe for debugging

#if MMU_DEBUG
    MMU_SCAN;
#endif // MMU_SCAN
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

#if MMU_DEBUG
    *((U64*)b1) = 0xeeeeeeeeeeeeeeee;	// wipe b1 header
#endif // MMU_DEBUG
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

	U32 index = __idx(blk->bsz);

    SET_MAP(index);								// set ticks for available maps

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
#if MMU_DEBUG
	MMU_SCAN
#endif // MMU_DEBUG
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

#if MMU_DEBUG
	MMU_SCAN;
#endif // MMU_DEBUG
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
    U32 index = __idx(sz);						// find free_list index by size

    if ((sz <= (1<<BASE_BITS+1)) && _free_list[index]) {
    	return index;
    }

    // no previous block exist, create a new one
    U32 l1 = L1(index);
    U32 l2 = L2(index);
    U32 o2 = l2;
    U32 m1, m2 = _l2_map[l1]>>(l2+1);
    if (m2) {									// check any 2nd level slot available
    	l2 = __ffs(m2 << l2);					// MSB represent the smallest slot that fits
    }
    else if ((m1=(_l1_map >> (l1+1)))!=0) {		// look one level up
    	l1 = __ffs(m1 << l1); 	       			// allocate lowest available bit
    	l2 = __ffs(_l2_map[l1]) - 1;			// get smallest size
    }
    else {
    	l1 = l2 = 0xff;							// out of memory
    }
#if MMU_DEBUG && CC_DEBUG
    PRINTF("mmu#found(%04x): %2x_%x INDEX(%x,%x) => %x, o2=%x, m2=%x\n", sz, m1, m2, l1, l2, INDEX(l1, l2), o2, _l2_map[l1]);
#endif // MMU_DEBUG && CC_DEBUG
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

    if ((bsz + MIN_BLOCK + sizeof(free_block)) > blk->bsz) return;	// too small to split


    // split block, free
    free_block *free = (free_block *)U8PADD(blk, bsz);				// future next block (i.e. alot bsz bytes)
    free_block *aft  = (free_block *)BLK_AFTER(blk);				// next adjacent block

    free->bsz = blk->bsz - bsz;										// carve out the acquired block
    free->psz = U8POFF(free, blk);									// positive offset to previous block
    blk->bsz  = bsz;												// allocate target block

    if (aft) {
        aft->psz = U8POFF(aft, free) | (aft->psz & FREE_FLAG);		// backward offset (positive)
        _merge_with_next(free);										// _combine if possible
    }
    _mark_free(free);			// add to free_list and set (free, tail, next, prev) fields

#if MMU_DEBUG
    MMU_SCAN;
#endif // MMU_SCAN
}

//================================================================
/*! initialize

  @param  ptr	pointer to free memory block.
  @param  size	size. (max 4G)
*/
__GURU__ void
_init_mmu(void *mem, U32 heap_size)
{
    ASSERT(heap_size > 0);

    U32 bsz = heap_size - sizeof(free_block);

    guru_device_heap = (U8*)mem;
    _heap_size   = heap_size;
    _mutex_mem	 = 0;

    // initialize entire memory pool as the first block
    free_block *head  = (free_block*)guru_device_heap;
    head->bsz = bsz;						// 1st (big) block
    head->psz = 0;
    SET_USED(head);

    _mark_free(head);						// will set free, tail, next, prev

    free_block *tail = (free_block*)BLK_AFTER(head);	// last block
    tail->bsz = tail->next = tail->prev = 0;
    tail->psz = bsz;
    SET_USED(tail);

#if MMU_DEBUG
    MMU_SCAN;
#endif // MMU_DEBUG
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
	_UNLOCK;

	ASSERT(blk->bsz >= bsz);					// make sure it provides big enough a block

#if MMU_DEBUG
	MMU_SCAN;
    _dump_freelist("alloc", sz);
    U32 *p = (U32*)BLK_DATA(blk);				// point to raw space allocated
    sz >>= 2;
    for (int i=0; i < (sz>16 ? 16 : sz); i++) *p++ = 0xaaaaaaaa;
#endif // MMU_DEBUG

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
#if MMU_DEBUG
	CHECK_MEMSZ(bsz);									// assume it is aligned already
#endif // MMU_DEBUG

    used_block *blk = (used_block *)BLK_HEAD(p0);
    ASSERT(IS_USED(blk));								// make sure it is used

    if (bsz > blk->bsz) {
    	_merge_with_next((free_block *)blk);			// try to get the block bigger
    }
    if (bsz == blk->bsz) return p0;						// fits right in
    if ((blk->bsz > bsz) &&
    		((blk->bsz - bsz) > GURU_STRBUF_SIZE)) {	// split a really big block
    	_LOCK;
    	_split((free_block*)blk, bsz);
    	_UNLOCK;
    	return p0;
    }
    //
    // compacting, mostly for str buffer
    // instead of splitting, since Ruby reuse certain sizes
    // it is better to allocate a block and release the original one
    //
    void *ret = guru_alloc(bsz);
    MEMCPY(ret, p0, sz);								// deep copy, !!using CUDA provided memcpy

    guru_free(p0);										// reclaim block

#if MMU_DEBUG
    MMU_SCAN;
#endif // MMU_SCAN
    return ret;
}

__GURU__ GR*
guru_gr_alloc(U32 n)
{
	return (GR*)guru_alloc(sizeof(GR) * n);
}

__GURU__ GR*
guru_gr_realloc(GR *gv, U32 n)
{
	return (GR*)guru_realloc(gv, sizeof(GR) * n);
}

//================================================================
/*! release memory
*/
__GURU__ void
guru_free(void *ptr)
{
	if (!ptr) return;

	_LOCK;
    free_block *blk = (free_block *)BLK_HEAD(ptr);			// get block header
#if MMU_DEBUG
    U32 bsz = blk->bsz;
#endif // MMU_DEBUG

    _merge_with_next(blk);

#if MMU_DEBUG
    if (BLK_AFTER(blk)) {
    	U32 *p = (U32*)U8PADD(blk, sizeof(used_block));
    	U32 sz = bsz ? (bsz - sizeof(used_block))>>2 : 0;
    	for (int i=0; i< (sz>32 ? 32 : sz); i++) *p++=0xffffffff;
    }
#endif // MMU_DEBUG
    _mark_free(blk);

    // the block is free now, try to merge a free block before if exists
    blk = _merge_with_prev(blk);
    _UNLOCK;

#if MMU_DEBUG
    MMU_SCAN;
	_dump_freelist("free", bsz);
#endif // MMU_DEBUG
}

//================================================================
/*! release memory, vm used.

  @param  vm	pointer to VM.
*/
__GURU__ void
guru_mmu_clr()
{
	used_block *p = (used_block *)guru_device_heap;
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
	used_block *p = (used_block *)guru_device_heap;
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

