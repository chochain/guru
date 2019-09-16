/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2019- GreenII
  </pre>
*/
#include <stdio.h>
#include "gurux.h"
#include "vmx.h"
#include "alloc.h"				// guru_malloc

// forward declaration for implementation
extern "C" __GPU__ void guru_memory_init(void *ptr, U32 sz);
extern "C" __GPU__ void guru_global_init(void);
extern "C" __GPU__ void guru_class_init(void);
extern "C" __GPU__ void guru_console_init(U8P buf, U32 sz);

U8P _guru_mem;				// guru global memory
U8P _guru_out;				// guru output stream
guru_ses *_ses_list = NULL; // session linked-list

__HOST__ U8P
_fetch_bytecode(const U8P rite_fname)
{
  FILE *fp = fopen((const char *)rite_fname, "rb");

  if (!fp) {
    fprintf(stderr, "File not found\n");
    return NULL;
  }

  // get filesize
  fseek(fp, 0, SEEK_END);
  size_t sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  U8P req = (U8P)cuda_malloc(sz, 1);			// allocate bytecode storage

  if (req) {
	  fread(req, sizeof(char), sz, fp);
  }
  fclose(fp);

  return req;
}

__HOST__ int
guru_setup(int trace)
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
	_ses_list = NULL;

	guru_memory_init<<<1,1>>>(mem, BLOCK_MEMORY_SIZE);			// setup memory management
	guru_global_init<<<1,1>>>();								// setup static objects
	guru_class_init<<<1,1>>>();									// setup basic classes
	guru_console_init<<<1,1>>>(out, MAX_BUFFER_SIZE);			// initialize output buffer

	U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	if (trace) {
		printf("guru session initialized[defaultStackSize %d => %d]\n", sz0, sz1);
		guru_dump_alloc_stat(trace);
	}
	return 0;
}

__HOST__ int
guru_load(char *rite_name, int step, int trace)
{
	guru_ses *ses = (guru_ses *)malloc(sizeof(guru_ses));
	if (!ses) return -1;		// memory allocation error

	ses->trace = trace;
	ses->out   = _guru_out;

	U8P in = ses->in = _fetch_bytecode((U8P)rite_name);
	if (!in) {
		fprintf(stderr, "ERROR: bytecode request allocation error!\n");
		return -2;
	}
	cudaError_t rst = guru_vm_setup(ses, step);
	if (cudaSuccess != rst) {
		fprintf(stderr, "ERROR: virtual memory block allocation error!\n");
		return -3;
	}
	if (trace) {
		guru_dump_alloc_stat(trace);
	}
	ses->next = _ses_list;		// add to linked-list
	_ses_list = ses;

	return 0;
}

__HOST__ int
guru_run(int trace)
{
	cudaError_t rst = guru_vm_run(_ses_list);
    if (cudaSuccess != rst) {
    	fprintf(stderr, "\nERR> %s\n", cudaGetErrorString(rst));
    }
	if (trace) {
		printf("guru_session completed\n");
		guru_dump_alloc_stat(trace);
	}
	rst = guru_vm_release(_ses_list);

	return cudaSuccess != rst;
}


