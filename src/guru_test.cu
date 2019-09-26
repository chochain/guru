/*! @file
  @brief
  GURU Unit Test module

  <pre>
  Copyright (C) 2019 GreenII.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in GURU.

  </pre>
*/
#include <stdio.h>
#include "alloc.h"
#include "gurux.h"
#include "vmx.h"

extern "C" __GPU__  void guru_memory_init(void *ptr, U32 sz);
extern "C" __GPU__  void guru_global_init(void);
extern "C" __GPU__  void guru_class_init(void);
extern "C" __GPU__  void guru_console_init(U8P buf, U32 sz);
extern "C" __HOST__ int  vm_pool_init(U32 step);

U8P _guru_mem;				// guru global memory
U8P _guru_out;				// guru output stream
guru_ses *_ses_list = NULL; // session linked-list

__GPU__ void
_mmu_alloc(U8 **b, U32 sz)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	U8 *p = (U8*)guru_alloc(sz);
	*b = p;
}

__GPU__ void
_mmu_free(U8 *b)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	guru_free(b);
}

__HOST__ int
guru_setup(int step, int trace)
{
	U8P mem = _guru_mem = (U8P)cuda_malloc(BLOCK_MEMORY_SIZE, 1);
	if (!_guru_mem) {
		fprintf(stderr, "ERROR: failed to allocate device main memory block!\n");
		return -1;
	}
	U8P out = _guru_out = (U8P)cuda_malloc(MAX_BUFFER_SIZE, 1);	// allocate output buffer
	if (!_guru_out) {
		fprintf(stderr, "ERROR: output buffer allocation error!\n");
		return -2;
	}
	if (vm_pool_init(step)) {
		fprintf(stderr, "ERROR: VM memory block allocation error!\n");
		return -3;
	}
	_ses_list = NULL;

	guru_memory_init<<<1,1>>>(mem, BLOCK_MEMORY_SIZE);			// setup memory management
//	guru_global_init<<<1,1>>>();								// setup static objects
//	guru_class_init<<<1,1>>>();									// setup basic classes
//	guru_console_init<<<1,1>>>(out, MAX_BUFFER_SIZE);			// initialize output buffer

	U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	if (trace) {
		printf("guru system initialized[defaultStackSize %d => %d]\n", sz0, sz1);
		guru_dump_alloc_stat(trace);
	}
	return 0;
}

__HOST__ int
guru_load(char *rite_fname)
{
	return 0;
}

__HOST__ int
guru_run(int trace)
{
	U32 a[] = { 0x28, 0x8, 0x10, 0x38, 0x8, 0x10 };
	U32 f[] = { 4, 1, 2 };
	U32 b[] = { 0x18, 0x10 };
	U8 *p   = (U8 *)cuda_malloc(12*sizeof(U8*), 1);
	U8 **x  = (U8**)p;

	printf("mmu_test starts here....");
	for (U32 i=0; i<sizeof(a)>>2; i++) {
		printf("\nalloc %d=>0x%02x", i, a[i]);
		_mmu_alloc<<<1,1>>>(&x[i], a[i]);
		guru_dump_alloc_stat(2);
		printf("\t=>%p", x[i]);
	}
	for (U32 i=0, j=f[0]; i<sizeof(f)>>2; j=f[++i]) {
		printf("\nfree %d=>%p", j, x[j]);
		_mmu_free<<<1,1>>>(x[j]);
		guru_dump_alloc_stat(2);
	}
	for (U32 i=0; i<sizeof(b)>>2; i++) {
		printf("\nalloc %d=>0x%02x", i, b[i]);
		_mmu_alloc<<<1,1>>>(&x[i+6], b[i]);
		guru_dump_alloc_stat(2);
		printf("\t=>%p", x[i+6]);
	}
	printf("\nmmu_test done!!!!!");

	return 0;
}
