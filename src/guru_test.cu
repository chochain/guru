/*! @file
  @brief
  GURU Unit Test module

  <pre>
  Copyright (C) 2019 GreenII.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in GURU.

  </pre>
*/
#include "helper_cuda.h"
#include "guru.h"
#include "gurux.h"
#include "mmu.h"
#include "vmx.h"
#include "symbol.h"

extern "C" __GPU__  void guru_core_init(void);
extern "C" __GPU__  void guru_console_init(U8 *buf, U32 sz);
extern "C" __HOST__ int  vm_pool_init(int step);

U8 *_guru_out;						// guru output stream
guru_ses *_ses_list = NULL; 		// session linked-list

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

__HOST__ void
_device_setup()
{
	cudaDeviceReset();

	int dv = findCudaDevice(1, NULL);
	cudaGetDevice(&dv);

	cudaDeviceProp dp;
	cudaGetDeviceProperties(&dp, dv);

	fprintf(stderr, "> Detected Compute SM %d.%d hardware with %d multi-processors (concurrent=%d)\n",
		dp.major, dp.minor, dp.multiProcessorCount, dp.concurrentKernels);
}

__HOST__ int
guru_setup(int step, int trace)
{
	_device_setup();

	U8 *mem = guru_host_heap = (U8*)cuda_malloc(GURU_HEAP_SIZE, 1);
	if (!mem) {
		fprintf(stderr, "ERROR: failed to allocate device main memory block!\n");
		return -1;
	}
	U8 *out = _guru_out = (U8*)cuda_malloc(OUTPUT_BUF_SIZE, 1);	// allocate output buffer
	if (!_guru_out) {
		fprintf(stderr, "ERROR: output buffer allocation error!\n");
		return -2;
	}
	if (vm_pool_init(step)) {
		fprintf(stderr, "ERROR: VM memory block allocation error!\n");
		return -3;
	}
	_ses_list = NULL;

	guru_mmu_init<<<1,1>>>(mem, GURU_HEAP_SIZE);				// setup memory management
//	guru_core_init<<<1,1>>>();									// setup basic classes
//	guru_console_init<<<1,1>>>(out, OUTPUT_BUF_SIZE);			// initialize output buffer

	U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	if (trace) {
		printf("guru system initialized[defaultStackSize %d => %d]\n", sz0, sz1);
		guru_mmu_check(trace);
	}
	return 0;
}

__HOST__ int
guru_load(char *rite_fname)
{
	return 0;
}


__HOST__ void
guru_mem_test(int ncycle)
{
	U32 a[] = { 0x1200, 0x80, 0x78, 0x8, 0x10, 0x1000, 0x100 };
	U32 f[] = { 4, 2, 1, 5 };
	U32 b[] = { 0x100, 0x10, 0x1010 };
	U32 asz = sizeof(a)/sizeof(U32);
	U32 fsz = sizeof(f)/sizeof(U32);
	U32 bsz = sizeof(b)/sizeof(U32);
	U8 *p   = (U8 *)cuda_malloc(128*sizeof(U8*), 1);
	U8 **x  = (U8**)p;

	for (int i=0; i<asz; i++) {
		_mmu_alloc<<<1,1>>>(&x[i], a[i]);
		guru_mmu_check(2);
		printf("alloc %d:%04x=>%p\n", i, a[i], x[i]);
	}
	for (int i=0, j=f[0]; i<fsz; j=f[++i]) {
		printf("free %d=>%p\n", j, x[j]);
		_mmu_free<<<1,1>>>(x[j]);
		guru_mmu_check(2);
	}
	for (int i=0; i<bsz; i++) {
		_mmu_alloc<<<1,1>>>(&x[i+asz], b[i]);
		guru_mmu_check(2);
		printf("alloc %d:%04x=>%p\n", i, b[i], x[i+6]);
	}
}

__GURU__ const char *slist[] = {
	"p",
	"sprintf",
	"printf",
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
    "inspect"
};

const int LIST_SIZE = sizeof(slist)/sizeof(char*);

__GPU__ void
_hash_test(U32 *sid, U32 ncycle)
{
	int x = threadIdx.x;

	for (int n=0; x<LIST_SIZE && n<ncycle; n++) {
		sid[x] = name2id((U8*)slist[x]);
	}
}


__HOST__ void
guru_hash_test(int ncycle)
{
	const int N=1;				// number of streams
	U32 *sid = (U32*)cuda_malloc(sizeof(U32)*N*LIST_SIZE, 1);

	cudaStream_t hst[N];												// host streams
	for (int i=0; i<N; i++)
		cudaStreamCreateWithFlags(&hst[i], cudaStreamNonBlocking);		// overhead ~= 0.013ms


	for (int i=0; i<N; i++) {
		_hash_test<<<1, 32, 0, hst[i]>>>(&sid[i*LIST_SIZE], ncycle);	// using default sync stream
	}
	cudaDeviceSynchronize();

	for (int i=0; i<N; i++)
		cudaStreamDestroy(hst[i]);

	for (int i=0; i<N; i++) {
		for (int j=0; j<LIST_SIZE; j++) {
			U32 x = j+i*LIST_SIZE;
			printf("%2d:%08x %d\n", x, sid[x], sid[x]);
		}
	}

	cuda_free(sid);
}

__HOST__ void
_time(const char *fname, int ncycle, void (*fp)(int))
{
	printf("%s starts here....\n", fname);

	cudaEvent_t _event_t0, _event_t1;
	float ms = 0;

	cudaEventCreate(&_event_t0);
	cudaEventCreate(&_event_t1);
	cudaEventRecord(_event_t0);

	fp(ncycle);

	cudaEventRecord(_event_t1);
	cudaEventSynchronize(_event_t1);
	cudaEventElapsedTime(&ms, _event_t0, _event_t1);
	cudaEventDestroy(_event_t1);
	cudaEventDestroy(_event_t0);

	printf("\n%s done in %f ms.\n", fname, ms);
}

__HOST__ int
guru_run()
{
	_time("mem_test", 1, &guru_mem_test);
	//_time("hash_test", 1000, &guru_hash_test);

	cudaDeviceReset();

	return 0;
}

__HOST__ void
guru_teardown(int sig) {}
