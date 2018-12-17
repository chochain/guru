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
#include <stdio.h>
#include <assert.h>
#include "alloc.h"

// TLSF: Two-Level Segregated Fit allocator with O(1) time complexity.
// Layer 1st(f), 2nd(s) model, and ignored last 4bit (i.e. smallest block is 16-bytes)
// f : size
// 0 : 0000-007f
// 1 : 0080-00ff
// 2 : 0100-01ff
// 3 : 0200-03ff
// 4 : 0400-07ff
// 5 : 0800-0fff
// 6 : 1000-1fff
// 7 : 2000-3fff
// 8 : 4000-7fff
// 9 : 8000-ffff

#ifndef L1_BITS			// 0000 0000 0000 0000
#define L1_BITS 	9	// ~~~~~~~~~~~
#define L1_MASK 	((1<<L1_BITS)-1)
#endif
#ifndef L2_BITS			// 0000 0000 0000 0000
#define L2_BITS 	3	//            ~~~
#define L2_MASK 	((1<<L2_BITS)-1)
#endif
#ifndef XX_BITS			// 0000 0000 0000 0000
#define XX_BITS 	4	//                ~~~~
#define XX_BLOCK	(1 << XX_BITS)					// 16-bytes
#endif

#define L1(i) 			(((i) >> L2_BITS) & L1_MASK)
#define L2(i) 			((i) & L2_MASK)
#define MSB_BIT 		0x8000
#define L1_MAP(i)      	(MSB_BIT >> L1(i))
#define L2_MAP(i)		(MSB_BIT >> L2(i))

// free memory block index
#define BLOCK_SLOTS		((L1_BITS + 1) * (1 << L2_BITS))

#define NEXT(p) 		((uint8_t *)(p) + (p)->size)
#define PREV(p) 		((uint8_t *)(p) - (p)->psize)
#define OFF(p0,p1) 		((uint8_t *)(p1) - (uint8_t *)(p0))

// semaphore
__GURU__ volatile int 	_mutex_mem;

// memory pool
__GURU__ unsigned int 	_memory_pool_size;
__GURU__ uint8_t     	*_memory_pool;

__GURU__ free_block 	*_free_list[BLOCK_SLOTS + 1];

// free memory bitmap
__GURU__ uint16_t 		_l1_map;
__GURU__ uint16_t 		_l2_map[L1_BITS + 2]; 		// + sentinel

#define GET_L1(i)		(_l1_map & (L1_MAP(i) - 1))
#define SET_L1(i)		(_l1_map |= L1_MAP(i))
#define CLR_L1(i)	    (_l1_map &= ~L1_MAP(i))
#define L2_KEY(i)		(_l2_map[L1(i)])
#define GET_L2(i)		(L2_KEY(i) & (L2_MAP(i) - 1))
#define SET_L2(i)	    (L2_KEY(i) |= L2_MAP(i))
#define CLR_L2(i)		(L2_KEY(i) &= ~L2_MAP(i))
#define CLEAR_MAP(i)	{ CLR_L2(i); if (L2_KEY(i)==0) CLR_L1(i); }
//================================================================
/*! Number of leading zeros.

  @param  x	target (16bit unsined)
  @retval int	nlz value
*/
__GURU__ int
__nlz16(uint16_t x)
{
    if (x==0) return 16;

    int n = 1;
    if ((x>> 8)==0) { n+=8; x<<=8; }
    if ((x>>12)==0) { n+=4; x<<=4; }
    if ((x>>14)==0) { n+=2; x<<=2; }

    return n - (x>>15);
}

__GURU__ __INLINE__ int
__calc_index(int l1, int l2)
{
    assert(l1 >= 0);
    assert(l1 <= L1_BITS);
    assert(l2 >= 0);
    assert(l2 <= L2_MASK);

    return (l1 << L2_BITS) | l2;
}

//================================================================
/*! calc f and s, and returns fli,sli of free_blocks

  @param  alloc_size	alloc size
  @retval int		index of free_blocks
*/
__GURU__ int
_get_index(unsigned int alloc_size)
{
    if ((alloc_size >> (L1_BITS+L2_BITS+XX_BITS)) != 0) {			// overflow check
        return BLOCK_SLOTS;
    }
    int l1    = 16 - __nlz16(alloc_size >> (L2_BITS + XX_BITS));	// 1st level index
    int shift =	l1 + XX_BITS - ((l1==0) ? 0 : 1);
    int l2    = (alloc_size >> shift) & L2_MASK;					// 2nd level index

    return __calc_index(l1, l2);
}

__GURU__ int
_get_free_index(unsigned int alloc_size)
{
    int index = _get_index(alloc_size);	// find free memory block

    if (_free_list[index] != NULL) {
    	return index;					// allocated before, keep using the same block
    }

    // no previous block exist, create a new one
    int l1 = L1(index);
    int l2 = L2(index);

    int used = GET_L2(index);
    if (used) {							// check any 2nd level available
    	l2 = __nlz16(used);
    }
    else {								// go up to 1st level
    	used = GET_L1(index);
        if (used) {						// allocate new 1st & 2nd level indices
        	l1 = __nlz16(used);			// CC: this might have problem, 20181104 because used is changed
            l2 = __nlz16(_l2_map[l1]);
        }
        else return -1;					// out of memeory
    }
    return __calc_index(l1, l2);		// new index
}

//================================================================
/*! just remove the free_block *target from index

  @param  target	pointer to target block.
*/
__GURU__ void
_remove_index(free_block *target)
{
    if (target->prev==NULL) {	// head of linked list?
        int index = _get_index(target->size) - 1;

        if ((_free_list[index]=target->next)==NULL) {
            CLEAR_MAP(index);	// make slot available
        }
    }
    else {						// link previous to next
        target->prev->next = target->next;
    }
    if (target->next != NULL) {	// reverse link
        target->next->prev = target->prev;
    }
}

//================================================================
/*! Mark that block free and register it in the free index table.

  @param  target	Pointer to target block.
*/
__GURU__ void
_mark_free(free_block *target)
{
    int index = _get_index(target->size) - 1;

#ifdef GURU_DEBUG
    int l1 = L1(index);  					// debug: (index>>3) & 0xff
    int l2 = L2(index);  					// debug: index & 0x7
    free_block *blk = _free_list[index];	// debug:
#endif

    SET_L1(index);							// update maps
    SET_L2(index);

    target->free = FLAG_FREE_BLOCK;
    target->next = _free_list[index];		// current block
    target->prev = NULL;
    if (target->next != NULL) {				// non-end block
        target->next->prev = target;
    }
    _free_list[index] = target;				// keep target as last block
}

//================================================================
/*! merge ptr1 and ptr2 block.
  ptr2 will disappear

  @param  ptr1	pointer to free block 1
  @param  ptr2	pointer to free block 2
*/
__GURU__ void
_merge_blocks(free_block *p0, free_block *p1)
{
    assert(p0 < p1);

    // merge ptr1 and ptr2
    p0->tail  = p1->tail;
    p0->size += p1->size;

    // update block info
    if (p0->tail==FLAG_NOT_TAIL_BLOCK) {
        free_block *next = (free_block *)NEXT(p0);
        next->psize = OFF(p0, next);
    }
#ifdef GURU_DEBUG
    *((uint64_t *)p1) = 0xeeeeeeeeeeeeeeee;
#endif
}

__GURU__ void
_merge_with_next(free_block *target)
{
	if (target->tail!=FLAG_NOT_TAIL_BLOCK) return;

	free_block *next = (free_block *)NEXT(target);

	if (next->free!=FLAG_FREE_BLOCK) return;

	_remove_index(next);
	_merge_blocks(target, next);
}

//================================================================
/*! Split block by size

  @param  target	pointer to target block
  @param  size	size
  @retval NULL	no split.
  @retval FREE_BLOCK *	pointer to splitted free block.
*/
__GURU__ void
_split_free_block(free_block *target, unsigned int size, int merge)
{
    if (target->size < (size + sizeof(free_block) + XX_BLOCK)) return; // too small to split 											// too small to split

    // split block, free
    free_block *free = (free_block *)((uint8_t *)target + size);	// future next block
    free_block *next = (free_block *)NEXT(target);					// current next

    free->size   = target->size - size;								// carve out the block
    free->psize  = OFF(target, free);
    free->tail   = target->tail;

    target->size = size;
    target->tail = FLAG_NOT_TAIL_BLOCK;

    if (free->tail==FLAG_NOT_TAIL_BLOCK) {
        next->psize = OFF(free, next);
    }
    if (free != NULL) {
    	if (merge) _merge_with_next(free);
    	_mark_free(free);
    }
}

/*
 * TODO: refactor into _remove_index()
 */
__GURU__ free_block*
_mark_used(int index)
{
    free_block *target = _free_list[index];

    assert(target!=NULL);

    // remove free_blocks index
    target->free      = FLAG_USED_BLOCK;
    _free_list[index] = target->next;

    if (target->next==NULL) {					// top of linked list
        CLEAR_MAP(index);						// release the index
    }
    else {
    	target->next->prev = target->prev;		// CC: is this needed? 20181104
    }
    return target;
}

//================================================================
/*! initialize

  @param  ptr	pointer to free memory block.
  @param  size	size. (max 64KB. see mrbc_memsize_t)
*/
__GURU__ void
_init_mmu(void *mem, unsigned int size)
{
    assert(size != 0);
    assert(size < 0x80000000);		// 2G max

    _mutex_mem		  = 0;
    _memory_pool      = (uint8_t *)mem;
    _memory_pool_size = size;

    // initialize entire memory pool as the first block
    free_block *block  = (free_block *)_memory_pool;
    block->tail = FLAG_TAIL_BLOCK;
    block->free = FLAG_FREE_BLOCK;
    block->size = _memory_pool_size;
    block->prev = 0;

    _mark_free(block);
}

//================================================================
/*! allocate memory

  @param  size	request size.
  @return void * pointer to allocated memory.
  @retval NULL	error.
*/
__GURU__ void*
mrbc_alloc(unsigned int size)
{
    // TODO: maximum alloc size
    //  (1 << (L1_BITS + L2_BITS + XX_BITS)) - alpha
    unsigned int alloc_size = size + sizeof(free_block);

#if GURU_REQUIRE_64BIT_ALIGNMENT
    alloc_size += ((8 - alloc_size) & 7);	// 8-byte align
#endif
    // check minimum alloc size. if need.
#ifdef GURU_DEBUG
    assert(alloc_size >= XX_BLOCK);
#else
    if (alloc_size < XX_BLOCK) {
        alloc_size = XX_BLOCK;
    }
#endif

	MUTEX_LOCK(_mutex_mem);

	int index 			= _get_free_index(alloc_size);
	free_block *target 	= _mark_used(index);

	_split_free_block(target, alloc_size, 0);

	MUTEX_FREE(_mutex_mem);

#ifdef GURU_DEBUG
    uint8_t *p = BLOCKDATA(target);
    for (int i=0; i < BLOCKSIZE(target); i++) *p++ = 0xaa;
#endif
    return BLOCKDATA(target);
}

//================================================================
/*! re-allocate memory

  @param  ptr	Return value of mrbc_raw_alloc()
  @param  size	request size
  @return void * pointer to allocated memory.
  @retval NULL	error.
*/
__GURU__ void*
mrbc_realloc(void *ptr, unsigned int size)
{
    used_block *target    = (used_block *)BLOCKHEAD(ptr);
    int        alloc_size = size + sizeof(free_block);

    alloc_size += ((8 - alloc_size) & 7);				// CC: 20181030 from 4 to 8-byte align

    if (alloc_size > target->size) {
    	_merge_with_next((free_block *)target);					// try to get the block bigger
    }
    if (alloc_size==target->size) return ptr;					// same size, good fit
    if (alloc_size < target->size) {							// a little to big, split if we can
        _split_free_block((free_block *)target, alloc_size, 1);
        return ptr;
    }

    // not big enough block found, new alloc and deep copy
    void *new_ptr = mrbc_alloc(size);
    if (!new_ptr) return NULL;  								// ENOMEM

    uint8_t *d = (uint8_t *)new_ptr;
    uint8_t *s = (uint8_t *)ptr;
    for (int i=0; i < size; i++) *d++=*s++;						// deep copy

    mrbc_free(ptr);												// reclaim block

    return new_ptr;
}

//================================================================
/*! release memory

  @param  ptr	Return value of mrbc_raw_alloc()
*/
__GURU__ void
mrbc_free(void *ptr)
{
	MUTEX_LOCK(_mutex_mem);

    free_block *target = (free_block *)BLOCKHEAD(ptr);	// get block header
    free_block *prev   = (free_block *)PREV(target);

    _merge_with_next(target);

    if (prev && prev->free==FLAG_FREE_BLOCK) {			// merge with previous, needed?
    	_remove_index(prev);
    	_merge_blocks(prev, target);
    	target = prev;
    }
    _mark_free(target);

    MUTEX_FREE(_mutex_mem);
}

//================================================================
/*! release memory, vm used.

  @param  vm	pointer to VM.
*/
__GURU__ void
mrbc_free_all()
{
    used_block *p = (used_block *)_memory_pool;
    while (1) {
    	if (p->free==FLAG_USED_BLOCK) {
    		mrbc_free(BLOCKDATA(p));
    	}
    	if (p->tail==FLAG_TAIL_BLOCK) break;
    	p = (used_block *)NEXT(p);
    }
}

#ifdef GURU_DEBUG
//================================================================
/*! statistics

  @param  *total	returns total memory.
  @param  *used		returns used memory.
  @param  *free		returns free memory.
  @param  *fragment	returns memory fragmentation
*/
__GPU__ void
_alloc_stat(int v[])
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	int total=0, nfree=0, free=0, nused=0, used=0, nblk=0, nfrag=0;

	used_block *p = (used_block *)_memory_pool;

	int flag = p->free;
	while (1) {
		if (flag != p->free) {       // supposed to be merged
			nfrag++;
			flag = p->free;
		}
		total += p->size;
		nblk  += 1;
		if (p->free==FLAG_FREE_BLOCK) {
			nfree += 1;
			free  += p->size;
		}
		if (p->free==FLAG_USED_BLOCK) {
			nused += 1;
			used  += p->size;
		}
		if (p->tail==FLAG_TAIL_BLOCK) break;
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
guru_memory_init(void *ptr, unsigned int sz)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	_init_mmu(ptr, sz);
}

__HOST__ void *
guru_malloc(size_t sz, int type)
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
guru_malloc_stat(int stat[])
{
	int *v;
	cudaMallocManaged(&v, 8*sizeof(int));

	_alloc_stat<<<1,1>>>(v);
	cudaDeviceSynchronize();

	for (int i=0; i<8; i++) {
		stat[i] = v[i];
	}
	cudaFree(v);
}

__HOST__ void
guru_dump_alloc_stat(void)
{
	int s[8];
	guru_malloc_stat(s);

	printf("\tmem=%d(0x%x): free=%d(%d), used=%d(%d), nblk=%d, nfrag=%d, %d%% allocated\n",
			s[0], s[0], s[1], s[2], s[3], s[4], s[5], s[6], (int)(100*(s[4]+1)/s[0]));
}
#endif
