/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2019- GreenII
  </pre>
*/
#include <stdio.h>
#include "gurux.h"
#include "mmu.h"				// guru_malloc
#include "load.h"				// guru_parse_bytecode
#include "vmx.h"

// forward declaration for implementation
extern "C" __GPU__  void guru_mmu_init(void *ptr, U32 sz);
extern "C" __GPU__  void guru_global_init(void);
extern "C" __GPU__  void guru_class_init(void);
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

	guru_mmu_init<<<1,1>>>(mem, BLOCK_MEMORY_SIZE);		// setup memory management
	guru_global_init<<<1,1>>>();							// setup static objects (TODO: => dynamic?)
	guru_class_init<<<1,1>>>();								// setup basic classes	(TODO: => ROM)
	guru_console_init<<<1,1>>>(out, MAX_BUFFER_SIZE);		// initialize output buffer

	U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	if (trace) {
		printf("guru system initialized[defaultStackSize %d => %d]\n", sz0, sz1);
		guru_mmu_stat(trace);
	}
	return 0;
}

__HOST__ int
guru_load(char *rite_name)
{
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
guru_run(int trace)
{
	for (guru_ses *ses=_ses_list; ses!=NULL; ses=ses->next) {
		U8 *irep_img = guru_parse_bytecode(ses->stdin);		// build CUDA IREP image (in Managed Mem)

		if (irep_img) {
			int vm_id = ses->id = vm_get(irep_img, trace);	// acquire a VM for the session
			vm_run(vm_id);
		}
		else {
			fprintf(stderr, "ERROR: bytecode parsing error!\n");
		}
	}
	// kick up main loop until all VM are done
	// TODO: become a server which responses to IREP requests
	vm_main_start(trace);

	return 0;
}
