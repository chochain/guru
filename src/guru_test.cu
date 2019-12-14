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
#include "mmu.h"
#include "gurux.h"
#include "vmx.h"
#include "symbol.h"

extern "C" __GPU__  void guru_mmu_init(void *ptr, U32 sz);
extern "C" __GPU__  void guru_class_init(void);
extern "C" __GPU__  void guru_console_init(U8 *buf, U32 sz);
extern "C" __HOST__ int  vm_pool_init(U32 step);

U8 *_guru_mem;				// guru global memory
U8 *_guru_out;				// guru output stream
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
	U8 *mem = _guru_mem = (U8*)cuda_malloc(BLOCK_MEMORY_SIZE, 1);
	if (!_guru_mem) {
		fprintf(stderr, "ERROR: failed to allocate device main memory block!\n");
		return -1;
	}
	U8 *out = _guru_out = (U8*)cuda_malloc(MAX_BUFFER_SIZE, 1);	// allocate output buffer
	if (!_guru_out) {
		fprintf(stderr, "ERROR: output buffer allocation error!\n");
		return -2;
	}
	if (vm_pool_init(step)) {
		fprintf(stderr, "ERROR: VM memory block allocation error!\n");
		return -3;
	}
	_ses_list = NULL;

	guru_mmu_init<<<1,1>>>(mem, BLOCK_MEMORY_SIZE);				// setup memory management
	guru_class_init<<<1,1>>>();									// setup basic classes
//	guru_console_init<<<1,1>>>(out, MAX_BUFFER_SIZE);			// initialize output buffer

	U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	if (trace) {
		printf("guru system initialized[defaultStackSize %d => %d]\n", sz0, sz1);
		show_mmu_stat(trace);
	}
	return 0;
}

__HOST__ int
guru_load(char *rite_fname)
{
	return 0;
}


__HOST__ void
_time(const char *fname, int ncycle, void (*fp)(cudaStream_t))
{
	printf("%s starts here....\n", fname);

	cudaStream_t hst;
	cudaStreamCreateWithFlags(&hst, cudaStreamNonBlocking);

	cudaEvent_t _event_t0, _event_t1;
	float ms = 0;

	cudaEventCreate(&_event_t0);
	cudaEventCreate(&_event_t1);
	cudaEventRecord(_event_t0);

	for (int i=0; i<ncycle; i++) {
		fp(hst);
	}

	cudaEventRecord(_event_t1);
	cudaEventSynchronize(_event_t1);
	cudaEventElapsedTime(&ms, _event_t0, _event_t1);
	cudaEventDestroy(_event_t1);
	cudaEventDestroy(_event_t0);

	cudaStreamDestroy(hst);

	printf("\n%s done in %f ms.\n", fname, ms);
}

__HOST__ void
guru_mem_test()
{
	U32 a[] = { 0x28, 0x8, 0x10, 0x38, 0x8, 0x10 };
	U32 f[] = { 4, 1, 2 };
	U32 b[] = { 0x18, 0x10 };
	U8 *p   = (U8 *)cuda_malloc(12*sizeof(U8*), 1);
	U8 **x  = (U8**)p;

	for (U32 i=0; i<sizeof(a)>>2; i++) {
		printf("\nalloc %d=>0x%02x", i, a[i]);
		_mmu_alloc<<<1,1>>>(&x[i], a[i]);
		show_mmu_stat(2);
		printf("\t=>%p", x[i]);
	}
	for (U32 i=0, j=f[0]; i<sizeof(f)>>2; j=f[++i]) {
		printf("\nfree %d=>%p", j, x[j]);
		_mmu_free<<<1,1>>>(x[j]);
		show_mmu_stat(2);
	}
	for (U32 i=0; i<sizeof(b)>>2; i++) {
		printf("\nalloc %d=>0x%02x", i, b[i]);
		_mmu_alloc<<<1,1>>>(&x[i+6], b[i]);
		show_mmu_stat(2);
		printf("\t=>%p", x[i+6]);
	}
}

__GURU__ const char *slist[] = {
    "initialize",
    "private",
    "!",
    "!=",
    "<=>",
    "===",
    "class",
    "include",
    "extend",
    "attr_reader",
    "attr_accessor",
    "lambda",
    "is_a?",
    "kind_of?",
    "puts",
    "print",
    "to_s",
    "inspect",
    "p",
    "sprintf",
    "printf"
};

__GPU__ void
_hash_test()
{
	if (threadIdx.x!=0) return;

	cudaStream_t dst;
	cudaStreamCreateWithFlags(&dst, cudaStreamNonBlocking);	// device stream [create,destroy] overhead =~ 0.060ms

	U32 sid;
	for (int i=0; i<sizeof(slist)/sizeof(char *); i++) {
		sid = name2id_s((U8*)slist[i], dst);
//		sid = name2id((U8*)slist[i]);
	}
	cudaStreamDestroy(dst);
}

__HOST__ void
guru_hash_test(cudaStream_t hst)
{
	_hash_test<<<1,1, 0, hst>>>();
	cudaStreamSynchronize(hst);
}

__HOST__ int
guru_run(int trace)
{
	//_time("mem_test", 10, &guru_mem_test);
	_time("hash_test", 1000, &guru_hash_test);

	cudaDeviceReset();
	return 0;
}
