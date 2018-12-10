/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2018- Greenii
  </pre>
*/
#include <stdio.h>
#include "guru.h"
#include "alloc.h"
#include "console.h"

// forward declaration for implementation
extern "C" __GURU__   void mrbc_init_global();							// global.cu
extern "C" __GURU__   void mrbc_init_class();							// class.cu

extern "C" cudaError_t guru_vm_init(guru_ses *ses);						// vm.cu
extern "C" cudaError_t guru_vm_run(guru_ses *ses);						// vm.cu
extern "C" cudaError_t guru_vm_release(guru_ses *ses);					// vm.cu

__GPU__ void
guru_static_init(void)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	mrbc_init_global();
	mrbc_init_class();
}

__HOST__ uint8_t*
_load_bytecode(const char *rite_fname)
{
  FILE *fp = fopen(rite_fname, "rb");

  if (!fp) {
    fprintf(stderr, "File not found\n");
    return NULL;
  }

  // get filesize
  fseek(fp, 0, SEEK_END);
  size_t sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  uint8_t *req = (uint8_t *)guru_malloc(sz, 1);	// allocate bytecode storage

  if (req) {
	  fread(req, sizeof(char), sz, fp);
  }
  fclose(fp);

  return req;
}

__HOST__ int
session_init(guru_ses *ses, const char *rite_fname)
{
	void *mem = guru_malloc(BLOCK_MEMORY_SIZE, 1);
	if (!mem) {
		fprintf(stderr, "ERROR: failed to allocate device main memory block!\n");
		return -1;
	}
	uint8_t *req = ses->req = _load_bytecode(rite_fname);
	if (!req) {
		fprintf(stderr, "ERROR: bytecode request allocation error!\n");
		return -2;
	}
	uint8_t *res = ses->res = (uint8_t *)guru_malloc(MAX_BUFFER_SIZE, 1);	// allocate output buffer
	if (!res) {
		fprintf(stderr, "ERROR: output buffer allocation error!\n");
		return -3;
	}
    guru_memory_init<<<1,1>>>(mem, BLOCK_MEMORY_SIZE);			// setup memory management
	guru_static_init<<<1,1>>>();								// setup static objects
    guru_console_init<<<1,1>>>(ses->res, MAX_BUFFER_SIZE);		// initialize output buffer

	int sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	if (ses->debug > 0) {
		printf("guru session initialized[defaultStackSize %d => %d]\n", sz0, sz1);
		guru_dump_alloc_stat();
	}
	return 0;
}

__HOST__ int
session_start(guru_ses *ses)
{
	cudaError_t rst = guru_vm_init(ses);
	if (cudaSuccess != rst) {
		fprintf(stderr, "ERROR: virtual memory block allocation error!\n");
		return -1;
	}
	if (ses->debug > 0) {
		printf("guru bytecode loaded\n");
		guru_dump_alloc_stat();
	}

	rst = guru_vm_run(ses);
    if (cudaSuccess != rst) {
    	fprintf(stderr, "\nERR> %s\n", cudaGetErrorString(rst));
    }

	if (ses->debug > 0) {
		printf("guru_session completed\n");
		guru_dump_alloc_stat();
	}
	rst = guru_vm_release(ses);
	return cudaSuccess!=rst;
}

    
