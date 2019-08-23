/*! @file
  @brief
  guru 32-bit memory management.
*/
#include <stdio.h>
#include <assert.h>
#include "alloc.h"

// TLSF: Two-Level Segregated Fit allocator with O(1) time complexity.
// Layer 1st(f), 2nd(s) model, smallest block 16-bytes, 16-byte alignment
// TODO: multiple-pool, thread-safe
/*
FLI = min(log2(memory_pool_size), 31)
void mapping(size_t size, unsigned *fl, unsigned *sl) {
	// fls() => Find_Last_Set bitmap function
 	*fl = fls(size);
 	*sl = ((size ˆ (1<<fl)) >> (*fl - SLI));
}
void *malloc(size){
	int fl, sl, fl2, sl2;
	void *found_block, *remaining_block;

	mapping(size, &fl, &sl);
	found_block=search_suitable_block(size,fl,sl);
	remove(found_block);

	if (sizeof(found_block)>size) {
		remaining_block = split(found_block, size);
		mapping(sizeof(remaining_block),&fl2,&sl2);
		insert(remaining_block, fl2, sl2);
	}
	remove(found_block);
	return found_block;
}
void free(block){
	int fl, sl;
	void *big_free_block;
	big_free_block = merge(block);
	mapping(sizeof(big_free_block), &fl, &sl);
	insert(big_free_block, fl, sl);
}
*/

#ifndef L1_BITS			// 00000000 00000000 00000000 00000000
#define L1_BITS 	24	// ~~~~~~~~ ~~~~~~~~ ^^^^^^^^           // 16+8 levels
#endif
#ifndef L2_BITS			// 00000000 00000000 00000000 00000000
#define L2_BITS 	4	// ~~~~~~~~ ~~~~~~~~          ^^^^      // 16 entries
#define L2_MASK 	((1<<L2_BITS)-1)
#endif
#ifndef MN_BITS			// 00000000 00000000 00000000 00000000  // smallest blocksize
#define MN_BITS 	4	// ~~~~~~~~ ~~~~~~~~              ^^^^  // 16-bytes
#define MN_BLOCK	(1 << MN_BITS)
#define BASE_BITS   (L2_BITS+MN_BITS)
#endif

#define L1(i) 			((i) >> L2_BITS)
#define L2(i) 			((i) & L2_MASK)
#define MSB_BIT 		31                                      // 32-bit MMU
#define FL_SLOTS		(L1_BITS * (1 << L2_BITS))				// slots for free_list pointers (24 * 16 entries)

#define NEXT(p) 		((uint8_t *)(p) + (p)->size)
#define PREV(p) 		((uint8_t *)(p) - (p)->poff)
#define OFF(p0,p1) 		((uint8_t *)(p1) - (uint8_t *)(p0))

// semaphore
__GURU__ volatile int 	_mutex_mem;

// memory pool
__GURU__ unsigned int 	_memory_pool_size;
__GURU__ uint8_t     	*_memory_pool;

// free memory bitmap
__GURU__ uint32_t 		_l1_map;								// use lower 24 bits
__GURU__ uint16_t 		_l2_map[L1_BITS];						// use all 16 bits
__GURU__ free_block 	*_free_list[FL_SLOTS];

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
__GURU__ __INLINE__ uint32_t
__fls(uint32_t x)
{
	int n;
	asm("bfind.u32 %0, %1;\n\t" : "=r"(n) : "r"(x));
	return n;
}

// least significant bit that is set
__GURU__ __INLINE__ uint32_t
__ffs(uint32_t x)
{
	int n;
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
__GURU__ int
__idx(unsigned int alloc_size, int *l1, int *l2)
{
	int v = __fls(alloc_size);
	int x = __ffs(alloc_size);

    *l1 = v<BASE_BITS ? 0 : v - BASE_BITS;			// 1st level index
    *l2 = (alloc_size >> (v - MN_BITS)) & L2_MASK;  // 2nd level index (with lower bits)

    return INDEX(*l1, *l2);
}

//================================================================
/*! wipe the free_block *target from linked list

  @param  target	pointer to target block.
*/
__GURU__ void
__release(free_block *target)
{
    if (target->prev==NULL) {			// head of linked list?
    	int l1, l2;
        int index = __idx(target->size, &l1, &l2);

        if ((_free_list[index]=target->next)==NULL) {
            CLEAR_MAP(index);			// mark as unallocated
        }
    }
    else {								// down link
        target->prev->next = target->next;
    }
    if (target->next != NULL) {			// up link
        target->next->prev = target->prev;
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
        next->poff = OFF(p0, next);
    }
#ifdef GURU_DEBUG
    *((uint64_t *)p1) = 0xeeeeeeeeeeeeeeee;
#endif
}

__GURU__ free_block*
_merge_with_next(free_block *target)
{
	if (target->tail) return target;

	free_block *next = (free_block *)NEXT(target);

	if (!next->free) return target;

	__release(next);
	__merge(target, next);

	return target;
}

__GURU__ free_block*
_merge_with_prev(free_block *target)
{
	free_block *prev = (free_block *)PREV(target);

	if (prev && prev->free) {			// merge with previous, needed?
		__release(prev);
		__merge(prev, target);
		target = prev;
	}
	return target;
}

//================================================================
/*! Mark that block free and register it in the free index table.

  @param  target	Pointer to target block.

  TODO: check thread safety
*/
__GURU__ void
_mark_free(free_block *target)
{
	int l1, l2;
    int index = __idx(target->size, &l1, &l2);

    int l1x= L1(index);
    int l2x= L2(index);
    int t1 = TIC(l1x);
    int t2 = TIC(l2x);
    uint32_t m1 = L1_MAP(index);
    uint16_t m2 = L2_MAP(index);

    SET_MAP(index);							// set free block available ticks

    uint32_t m1x = L1_MAP(index);
    uint16_t m2x = L2_MAP(index);

    free_block *head = _free_list[index];

    target->free = 1;
    target->next = head;					// setup linked list
    target->prev = NULL;
    if (head) {								// non-end block, add backward link
    	head->prev = target;
    }

    _free_list[index] = target;				// new head of the linked list
}

__GURU__ free_block*
_mark_used(int index)
{
    free_block *target = _free_list[index];

    assert(target!=NULL);

    if (target->next==NULL) {					// top of linked list
        int l1x= L1(index);
        int l2x= L2(index);
        int t1 = TIC(l1x);
        int t2 = TIC(l2x);
        uint32_t m1 = L1_MAP(index);
        uint16_t m2 = L2_MAP(index);

        CLEAR_MAP(index);						// release the index

        uint32_t m1x = L1_MAP(index);
        uint16_t m2x = L2_MAP(index);

        if (L1_MAP(index)==0 && L2_MAP(index)==0) {
        	_free_list[index] = NULL;
        }
    }
    else {
        _free_list[index] = target->next;		// follow the linked list
    	target->next->prev = target->prev;		// 20190819 CC: is this necessary?
    }
    target->free = 0;

    return target;
}

//================================================================
/*! Find index to a free block

  @param  size	size
  @retval -1	not found
  @retval index to available _free_list
*/
__GURU__ int
_find_free_block(unsigned int alloc_size)
{
	int l1, l2;
    int index = __idx(alloc_size, &l1, &l2);	// find free_list index by size

    if (_free_list[index]) return index;		// free block available, use it

    // no previous block exist, create a new one
    int avl = _l2_map[l1];			    // check any 2nd level available
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

  @param  target	pointer to target block
  @param  size	size
  @retval NULL	no split.
  @retval FREE_BLOCK *	pointer to splitted free block.
*/
__GURU__ void
_split_free_block(free_block *target, unsigned int size)
{
    if (target->size < (size + sizeof(free_block) + MN_BLOCK)) return; // too small to split 											// too small to split

    // split block, free
    free_block *free = (free_block *)((uint8_t *)target + size);	// future next block
    free_block *next = (free_block *)NEXT(target);					// current next

    free->size   = target->size - size;								// carve out the block
    free->poff   = OFF(target, free);
    free->tail   = target->tail;
    free->free   = 1;

    if (!free->tail) {
        next->poff = OFF(free, next);
    }
    _mark_free(free);												// add to free_list

    target->size = size;											// reduce size
    target->tail = 0;
}

//================================================================
/*! initialize

  @param  ptr	pointer to free memory block.
  @param  size	size. (max 64KB. see mrbc_memsize_t)
*/
__GURU__ void
_init_mmu(void *mem, unsigned int size)
{
    assert(size > 0);

    _mutex_mem		  = 0;
    _memory_pool      = (uint8_t *)mem;
    _memory_pool_size = size;

    // initialize entire memory pool as the first block
    free_block *block  = (free_block *)_memory_pool;
    block->tail = 1;
    block->free = 1;
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
    //  (1 << (L1_BITS + L2_BITS + MN_BITS)) - alpha
    unsigned int alloc_size = size + sizeof(used_block);

#if GURU_REQUIRE_64BIT_ALIGNMENT
    alloc_size += ((8 - alloc_size) & 7);	// 8-byte align
#endif
    // check minimum alloc size. if need.
#ifdef GURU_DEBUG
    assert(alloc_size >= MN_BLOCK);
#else
    if (alloc_size < MN_BLOCK) {
        alloc_size = MN_BLOCK;
    }
#endif

	MUTEX_LOCK(_mutex_mem);

	int index 			= _find_free_block(alloc_size);
	free_block *target 	= _mark_used(index);

#ifdef GURU_DEBUG
    uint32_t *p = (uint32_t*)BLOCKDATA(target);
    for (int i=0; i < (alloc_size - sizeof(used_block))>>2; i++) *p++ = 0xaaaaaaaa;
#endif
	_split_free_block(target, alloc_size);

	MUTEX_FREE(_mutex_mem);

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
        _split_free_block((free_block *)target, alloc_size);
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

    target = _merge_with_next(target);
    target = _merge_with_prev(target);

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
    	if (!p->free) {
    		mrbc_free(BLOCKDATA(p));
    	}
    	if (p->tail) break;
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
		if (p->free) {
			nfree += 1;
			free  += p->size;
		}
		if (!p->free) {
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
