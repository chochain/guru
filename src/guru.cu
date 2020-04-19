/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2019- GreenII
  </pre>
*/
#include <stdio.h>
#include "guru.h"
#include "gurux.h"
#include "mmu.h"				// guru_malloc
#include "vmx.h"
#include "debug.h"

// forward declaration for implementation
extern "C" __GPU__  void guru_mmu_init(void *ptr, U32 sz);
extern "C" __GPU__  void guru_core_init(void);
extern "C" __GPU__  void guru_console_init(U8 *buf, U32 sz);

U8 *_guru_mem;					// guru global memory
U8 *_guru_out;					// guru output stream
guru_ses *_ses_list = NULL; 	// session linked-list

//
// _fetch_bytecode:
//     	read raw bytecode from input file (or stream) into CUDA managed memory
//		for later CUDA IREP image building
//
__HOST__ U8 *
_fetch_bytecode(const U8 *rite_fname)
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

  U8 *req = (U8*)cuda_malloc(sz, 1);			// allocate bytecode storage

  if (req) {
	  fread(req, sizeof(char), sz, fp);
  }
  fclose(fp);

  return req;
}

__HOST__ int
guru_setup(int step, int trace)
{
	cudaDeviceReset();

	debug_init(trace);												// initialize logger
	debug_log("guru initializing...");

	U8 *mem = _guru_mem = (U8*)cuda_malloc(BLOCK_MEMORY_SIZE, 1);	// allocate main block (i.e. RAM)
	if (!_guru_mem) {
		fprintf(stderr, "ERROR: failed to allocate device main memory block!\n");
		return -1;
	}
	U8 *out = _guru_out = (U8*)cuda_malloc(MAX_BUFFER_SIZE, 1);		// allocate output buffer
	if (!_guru_out) {
		fprintf(stderr, "ERROR: output buffer allocation error!\n");
		return -2;
	}
	if (vm_pool_init(step)) {										// allocate VM pool
		fprintf(stderr, "ERROR: VM memory block allocation error!\n");
		return -3;
	}
	_ses_list = NULL;

	guru_mmu_init<<<1,1>>>(mem, BLOCK_MEMORY_SIZE);			// setup memory management
	guru_core_init<<<1,1>>>();								// setup basic classes	(TODO: => ROM)
#if GURU_USE_CONSOLE
	guru_console_init<<<1,1>>>(out, MAX_BUFFER_SIZE);		// initialize output buffer
#endif

    U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	debug_log("guru initialized, ready to go...");

	return 0;
}

__HOST__ int
guru_load(char *rite_name)
{
	debug_log("guru loading RITE image into ses->stdin memory...");

	guru_ses *ses = (guru_ses *)malloc(sizeof(guru_ses));
	if (!ses) return -1;		// memory allocation error

	ses->stdout = _guru_out;

	U8 *ins = ses->stdin = _fetch_bytecode((U8*)rite_name);
	if (!ins) {
		fprintf(stderr, "ERROR: bytecode request allocation error!\n");
		return -2;
	}
	ses->next = _ses_list;		// add to linked-list
	_ses_list = ses;

	return 0;
}

__HOST__ int
guru_run()
{
	debug_log("guru session starting...");
	debug_mmu_stat();

	// parse BITE code into each vm
	// TODO: work producer (enqueue)
	for (guru_ses *ses=_ses_list; ses!=NULL; ses=ses->next) {
		int x = ses->id = vm_get(ses->stdin);
		if      (x==-1) fprintf(stderr, "ERROR: bytecode parsing error!\n");
		else if (x==-2) fprintf(stderr, "ERROR: No more VM available!\n");
		else if (vm_ready(x)) {
			fprintf(stderr, "ERROR: VM state failed to transit!\n");
		}
	}
	// kick up main loop until all VM are done
	vm_main_start();

	debug_mmu_stat();
	debug_log("guru session completed.");

	return 0;
}

__HOST__ void
guru_teardown(int sig)
{
	cudaDeviceReset();

	guru_ses *tmp, *ses = _ses_list;
	while (ses) {
		tmp = ses;
		ses = ses->next;
		free(tmp);
	}
}
