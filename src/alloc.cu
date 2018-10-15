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
#include <assert.h>
#include "value.h"
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

#ifndef FLI_BITS		// 0000 0000 0000 0000
#define FLI_BITS 	9	// ~~~~~~~~~~~
#define FLI_MASK 	((1<<FLI_BITS)-1)
#endif
#ifndef SLI_BITS		// 0000 0000 0000 0000
#define SLI_BITS 	3	//            ~~~
#define SLI_MASK 	((1<<SLI_BITS)-1)
#endif
#ifndef LSB_BITS		// 0000 0000 0000 0000
#define LSB_BITS 	4	//                ~~~~
#define LSB_BLOCK	(1 << LSB_BITS)
#endif

#define FLI(i) 			(((i) >> SLI_BITS) & FLI_MASK)
#define SLI(i) 			((i) & SLI_MASK)
#define MSB_BIT 		0x8000
#define FLI_MAP(i)      (MSB_BIT >> FLI(i))
#define SLI_MAP(i)		(MSB_BIT >> SLI(i))

// free memory block index
#define SIZE_FREE_BLOCKS ((FLI_BITS + 1) * (1 << SLI_BITS))

#define NEXT(p) 		((uint8_t *)(p) + (p)->size)
#define PREV(p) 		((uint8_t *)(p) - (p)->offset)
#define OFF(p1,p2) 		((uint8_t *)(p2) - (uint8_t *)(p1))

// memory pool
__GURU__ unsigned int 	memory_pool_size;
__GURU__ uint8_t     	*memory_pool;

__GURU__ free_block 	*free_list[SIZE_FREE_BLOCKS + 1];

// free memory bitmap
__GURU__ uint16_t 		fli_bitmap;
__GURU__ uint16_t 		sli_bitmap[FLI_BITS + 2]; // + sentinel

#define FLI_USED(i)		(fli_bitmap & (FLI_MAP(i) - 1))
#define SLI_USED(i)		(sli_bitmap[FLI(i)] & (SLI_MAP(i) - 1))
#define MARK_FREE(i)	{ int fli = FLI(i); \
							sli_bitmap[fli] &= ~SLI_MAP(i); \
							if (sli_bitmap[fli]==0) { fli_bitmap &= ~FLI_MAP(i); } }
//================================================================
/*! Number of leading zeros.

  @param  x	target (16bit unsined)
  @retval int	nlz value
*/
__GURU__ __forceinline__
int _nlz16(uint16_t x)
{
    if (x==0) return 16;

    int n = 1;
    if ((x>> 8)==0) { n+=8; x<<=8; }
    if ((x>>12)==0) { n+=4; x<<=4; }
    if ((x>>14)==0) { n+=2; x<<=2; }

    return n - (x>>15);
}

//================================================================
/*! calc f and s, and returns fli,sli of free_blocks

  @param  alloc_size	alloc size
  @retval int		index of free_blocks
*/
__GURU__
int _get_index(unsigned int alloc_size)
{
    if ((alloc_size >> (FLI_BITS+SLI_BITS+LSB_BITS)) != 0) {		// overflow check
        return SIZE_FREE_BLOCKS;
    }
    int fli   = 16 - _nlz16(alloc_size >> (SLI_BITS + LSB_BITS));	// 1st level index
    int shift =	fli + LSB_BITS - ((fli==0) ? 0 : 1);
    int sli   = (alloc_size >> shift) & SLI_MASK;					// 2nd level index

    assert(fli >= 0);
    assert(fli <= FLI_BITS);
    assert(sli >= 0);
    assert(sli <= SLI_MASK);

    return (fli << SLI_BITS) + sli;
}

//================================================================
/*! Mark that block free and register it in the free index table.

  @param  target	Pointer to target block.
*/
__GURU__
void _add_free_block(free_block *target)
{
    target->f = FLAG_FREE_BLOCK;

    int index = _get_index(target->size) - 1;
    int fli   = FLI(index);  // debug: (index>>3) & ((1<<9)-1)
    int sli   = SLI(index);  // debug: (index & ((1<<3)-1)

    fli_bitmap      |= FLI_MAP(index);
    sli_bitmap[fli] |= SLI_MAP(index);

    target->prev = NULL;
    target->next = free_list[index];
    if (target->next != NULL) {
        target->next->prev = target;
    }
    free_list[index] = target;

#ifdef MRBC_DEBUG
    MEMSET((uint8_t *)(target + sizeof(free_block)), 0xff, target->size - sizeof(free_block));
#endif
}

//================================================================
/*! just remove the free_block *target from index

  @param  target	pointer to target block.
*/
__GURU__
void _remove_index(free_block *target)
{
    if (target->prev==NULL) {	// head of linked list?
        int index = _get_index(target->size) - 1;

        if ((free_list[index]=target->next)==NULL) {
            MARK_FREE(index);
        }
    }
    else {	// link previous to next
        target->prev->next = target->next;
    }
    if (target->next != NULL) {	// reverse link
        target->next->prev = target->prev;
    }
}

//================================================================
/*! Split block by size

  @param  target	pointer to target block
  @param  size	size
  @retval NULL	no split.
  @retval FREE_BLOCK *	pointer to splitted free block.
*/
__GURU__
free_block *_split_block(free_block *target, unsigned int size)
{
    if (target->size < (size + sizeof(free_block) + LSB_BLOCK)) {
    	return NULL;
    }

    // split block, free
    free_block *split = (free_block *)((uint8_t *)target + size);
    free_block *next  = (free_block *)NEXT(target);

    split->size  	= target->size - size;
    split->offset	= OFF(target, split);
    split->t     	= target->t;

    target->size 	= size;
    target->t    	= FLAG_NOT_TAIL_BLOCK;

    if (split->t==FLAG_NOT_TAIL_BLOCK) {
        next->offset = OFF(split, next);
    }
    return split;
}

//================================================================
/*! merge ptr1 and ptr2 block.
  ptr2 will disappear

  @param  ptr1	pointer to free block 1
  @param  ptr2	pointer to free block 2
*/
__GURU__
void _merge_block(free_block *ptr1, free_block *ptr2)
{
    assert(ptr1 < ptr2);

    // merge ptr1 and ptr2
    ptr1->t     = ptr2->t;
    ptr1->size += ptr2->size;

    // update block info
    if (ptr1->t==FLAG_NOT_TAIL_BLOCK) {
        free_block *next = (free_block *)NEXT(ptr1);
        next->offset = OFF(ptr1, next);
    }
}

__GURU__
void _merge_with_next(free_block *target)
{
	if (target->t!=FLAG_NOT_TAIL_BLOCK) return;

	free_block *next = (free_block *)NEXT(target);

	if (next->f!=FLAG_FREE_BLOCK) return;

	_remove_index(next);
	_merge_block(target, next);
}

__GURU__
free_block *_merge_with_prev(free_block *target)
{
    free_block *prev = (free_block *)PREV(target);

    if (prev==NULL || prev->f!=FLAG_FREE_BLOCK) return target; 	// no change

    _remove_index(prev);
    _merge_block(prev, target);

    return prev;
}

__GURU__
int _get_free_index(unsigned int alloc_size)
{
    int index = _get_index(alloc_size);	// find free memory block

    if (free_list[index] != NULL) {
    	return index;					// allocated before, keep using the same block
    }

    // no previous block exist, create a new one
    int fli = FLI(index);
    int sli = SLI(index);

    uint16_t used = SLI_USED(index);
    if (used != 0) {					// check any 2nd level available
    	sli = _nlz16(used);
    }
    else {								// go up to 1st level
    	used = FLI_USED(index);
        if (used == 0) {				// out of memory
        	return -1;
        }
        else {							// allocate new 1st & 2nd level indices
        	fli = _nlz16(used);
            sli = _nlz16(sli_bitmap[fli]);
        }
    }
    assert(fli >= 0);
    assert(fli <= FLI_BITS);
    assert(sli >= 0);
    assert(sli <= (1 << SLI_BITS) - 1);

    return (fli << SLI_BITS) + sli;		// new index
}

/*
 * TODO: refactor into _remove_index()
 */
__GURU__
free_block *_mark_used(int index)
{
    free_block *target = free_list[index];

    assert(target!=NULL);

    // remove free_blocks index
    target->f        = FLAG_USED_BLOCK;
    free_list[index] = target->next;

    if (target->next==NULL) {			// end of linked list?
        MARK_FREE(index);
    }
    else {
        target->next->prev = NULL;		// is this right?
    }
    return target;
}

//================================================================
/*! initialize

  @param  ptr	pointer to free memory block.
  @param  size	size. (max 64KB. see mrbc_memsize_t)
*/
__GURU__
void mrbc_init_alloc(void *ptr, unsigned int size)
{
    assert(size != 0);
    assert(size <= (mrbc_memsize_t)(~0));

    memory_pool      = (uint8_t *)ptr;
    memory_pool_size = size;

    // initialize entire memory pool as the first block
    free_block *block  = (free_block *)memory_pool;
    block->t      = FLAG_TAIL_BLOCK;
    block->f      = FLAG_FREE_BLOCK;
    block->size   = memory_pool_size;
    block->offset = 0;

    _add_free_block(block);
}

//================================================================
/*! allocate memory

  @param  size	request size.
  @return void * pointer to allocated memory.
  @retval NULL	error.
*/
__GURU__
void *mrbc_alloc(unsigned int size)
{
    // TODO: maximum alloc size
    //  (1 << (FLI_BITS + SLI_BITS + LSB_BITS)) - alpha
    unsigned int alloc_size = size + sizeof(free_block);

    // align 4 byte
    alloc_size += ((4 - alloc_size) & 3);

    // check minimum alloc size. if need.
#if 0
    if (alloc_size < LSB_BLOCK) {
        alloc_size = LSB_BLOCK;
    }
#else
    assert(alloc_size >= LSB_BLOCK);
#endif

	int index = _get_free_index(alloc_size);
    free_block *target = _mark_used(index);

    assert(target->size >= alloc_size);

    // split the allocated block
    free_block *release = _split_block(target, alloc_size);
    if (release != NULL) {
        _add_free_block(release);
    }

#ifdef MRBC_DEBUG
    MEMSET((uint8_t *)target + sizeof(used_block), 0xaa, target->size - sizeof(used_block));
#endif

    return (uint8_t *)target + sizeof(used_block);
}

//================================================================
/*! re-allocate memory

  @param  ptr	Return value of mrbc_raw_alloc()
  @param  size	request size
  @return void * pointer to allocated memory.
  @retval NULL	error.
*/
__GURU__
void * mrbc_realloc(void *ptr, unsigned int size)
{
    used_block *target      = (used_block *)((uint8_t *)ptr - sizeof(used_block));
    unsigned int alloc_size = size + sizeof(free_block);

    // align 4 byte
    alloc_size += ((4 - alloc_size) & 3);

    // expand part1.
    // next phys block is free and check enough size?
    if (alloc_size > target->size) {
    	_merge_with_next((free_block *)target);
    }
    if (alloc_size==target->size) {		// is the size the same now?
        return ptr;
    }
    if (alloc_size < target->size) {	// need to split
        free_block *release = _split_block((free_block *)target, alloc_size);
        if (release != NULL) {
            _merge_with_next(release);
            _add_free_block(release);
        }
        return ptr;
    }

    // expand part2.
    // new alloc and deep copy
    uint8_t *new_ptr = (uint8_t *)mrbc_alloc(size);
    if (new_ptr==NULL) return NULL;  // ENOMEM

    MEMCPY(new_ptr, (uint8_t *)ptr, (size_t)(target->size - sizeof(used_block)));
    mrbc_free(ptr);

    return (void *)new_ptr;
}

//================================================================
/*! release memory

  @param  ptr	Return value of mrbc_raw_alloc()
*/
__GURU__
void mrbc_free(void *ptr)
{
    // get target block
    free_block *target = (free_block *)((uint8_t *)ptr - sizeof(used_block));

    _merge_with_next(target);
    _add_free_block(_merge_with_prev(target));	// target, add to index
}

//================================================================
/*! release memory, vm used.

  @param  vm	pointer to VM.
*/
__GURU__
void mrbc_free_all()
{
    used_block *ptr = (used_block *)memory_pool;
    void *free_target = NULL;
    int flag_loop = 1;

    while (flag_loop) {
        if (ptr->t==FLAG_TAIL_BLOCK) flag_loop = 0;
        if (ptr->f==FLAG_USED_BLOCK) {
            if (free_target) {
                mrbc_free(free_target);
            }
            free_target = (uint8_t *)ptr + sizeof(used_block);
        }
        ptr = (used_block *)NEXT(ptr);
    }
    if (free_target) {
        mrbc_free(free_target);
    }
}

#ifdef MRBC_DEBUG
//================================================================
/*! statistics

  @param  *total	returns total memory.
  @param  *used		returns used memory.
  @param  *free		returns free memory.
  @param  *fragment	returns memory fragmentation
*/
__GURU__
void mrbc_alloc_statistics(int *total, int *used, int *free, int *fragmentation)
{
    *total = memory_pool_size;
    *used = 0;
    *free = 0;
    *fragmentation = 0;

    used_block *ptr = (used_block *)memory_pool;
    int flag_used_free = ptr->f;
    while (1) {
        if (ptr->f) {
            *free += ptr->size;
        } else {
            *used += ptr->size;
        }
        if (flag_used_free != ptr->f) {
            (*fragmentation)++;
            flag_used_free = ptr->f;
        }

        if (ptr->t==FLAG_TAIL_BLOCK) break;

        ptr = (used_block *)NEXT(ptr);
    }
}

//================================================================
/*! statistics

  @param  vm_id		vm_id
  @return int		total used memory size
*/
__GURU__
int mrbc_alloc_used()
{
    used_block *ptr = (used_block *)memory_pool;
    int total = 0;

    while (1) {
        if (!ptr->f) {
            total += ptr->size;
        }
        if (ptr->t==FLAG_TAIL_BLOCK) break;

        ptr = (used_block *)NEXT(ptr);
    }
    return total;
}

__global__ void guru_init_alloc(void *ptr, unsigned int sz)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	mrbc_init_alloc(ptr, sz);
}

void *guru_malloc(size_t sz)
{
	void *mem;

    cudaMallocManaged(&mem, sz);			// allocate managed memory
    if (cudaSuccess != cudaGetLastError()) return NULL;

    return mem;
}
#endif
