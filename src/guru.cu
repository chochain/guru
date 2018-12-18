/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2018- Greenii
  </pre>
*/
#include <stdio.h>
#include "gurux.h"
#include "vmx.h"
#include "alloc.h"				// guru_malloc

// forward declaration for implementation
extern "C" __GPU__ void guru_memory_init(void *ptr, unsigned int sz);
extern "C" __GPU__ void guru_global_init(void);
extern "C" __GPU__ void guru_class_init(void);
extern "C" __GPU__ void guru_console_init(uint8_t *buf, unsigned int sz);

uint8_t  *_guru_mem;			// guru global memory
uint8_t  *_guru_out;			// guru output stream
guru_ses *_ses_list;			// session linked-list

__HOST__ uint8_t*
_get_request_bytecode(const char *rite_fname)
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
guru_system_setup(int trace)
{
	uint8_t *mem = _guru_mem = (uint8_t *)guru_malloc(BLOCK_MEMORY_SIZE, 1);
	if (!_guru_mem) {
		fprintf(stderr, "ERROR: failed to allocate device main memory block!\n");
		return -1;
	}
	uint8_t *out = _guru_out = (uint8_t *)guru_malloc(MAX_BUFFER_SIZE, 1);	// allocate output buffer
	if (!_guru_out) {
		fprintf(stderr, "ERROR: output buffer allocation error!\n");
		return -2;
	}
	_ses_list = NULL;

	guru_memory_init<<<1,1>>>(mem, BLOCK_MEMORY_SIZE);			// setup memory management
	guru_global_init<<<1,1>>>();								// setup static objects
	guru_class_init<<<1,1>>>();									// setup basic classes
    guru_console_init<<<1,1>>>(out, MAX_BUFFER_SIZE);			// initialize output buffer

    int sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	if (trace) {
		printf("guru session initialized[defaultStackSize %d => %d]\n", sz0, sz1);
		guru_dump_alloc_stat();
	}
	return 0;
}

__HOST__ int
guru_system_run(int trace)
{
	cudaError_t rst = guru_vm_run(_ses_list, trace);
    if (cudaSuccess != rst) {
    	fprintf(stderr, "\nERR> %s\n", cudaGetErrorString(rst));
    }
	if (trace) {
		printf("guru_session completed\n");
		guru_dump_alloc_stat();
	}
	rst = guru_vm_release(_ses_list, trace);

	return cudaSuccess != rst;
}

__HOST__ int
session_add(guru_ses *ses, const char *rite_fname, int trace)
{
	ses->trace = trace;
	ses->out   = _guru_out;

	uint8_t *in = ses->in = _get_request_bytecode(rite_fname);
	if (!in) {
		fprintf(stderr, "ERROR: bytecode request allocation error!\n");
		return -3;
	}
	cudaError_t rst = guru_vm_setup(ses, trace);
	if (cudaSuccess != rst) {
		fprintf(stderr, "ERROR: virtual memory block allocation error!\n");
		return -1;
	}
	if (trace) guru_dump_alloc_stat();

	ses->next = _ses_list;		// add to linked-list
	_ses_list = ses;

	return rst;
}
