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

U8 *guru_host_heap;				// guru global memory

class Guru::Impl
{
	VM_Pool  *_vm_pool  = NULL;
	guru_ses *_ses_list = NULL;
	U8       *_guru_out = NULL;		// guru output buffer
	//
	// _fetch_bytecode:
	//     	read raw bytecode from input file (or stream) into CUDA managed memory
	//		for later CUDA IREP image building
	//
	__HOST__ char *
	_fetch_bytecode(const char *rite_fname)
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

		char *req = (char*)cuda_malloc(sz, 1);			// allocate bytecode storage

		if (req) {
			fread(req, sizeof(char), sz, fp);
		}
		fclose(fp);

		return req;
	}

public:
	__HOST__ Impl() {
		cudaDeviceReset();
	}

	__HOST__ ~Impl() {
		cudaDeviceReset();

		guru_ses *tmp, *ses = _ses_list;
		while (ses) {
			tmp = ses;
			ses = ses->next;
			free(tmp);
		}
	}

	__HOST__ int
	init(int step)
	{
		U8 *mem = guru_host_heap = (U8*)cuda_malloc(GURU_HEAP_SIZE, 1);	// allocate main block (i.e. RAM)
		if (!mem) 		return -1;

		U8 *out = _guru_out = (U8*)cuda_malloc(OUTPUT_BUF_SIZE, 1);		// allocate output buffer
		if (!_guru_out) return -2;

		_vm_pool = new VM_Pool(step);
		if (!_vm_pool) 	return -3;

		guru_mmu_init<<<1,1>>>(mem, GURU_HEAP_SIZE);			// setup memory management
		guru_core_init<<<1,1>>>();								// setup basic classes	(TODO: => ROM)
	#if GURU_USE_CONSOLE
		guru_console_init<<<1,1>>>(out, OUTPUT_BUF_SIZE);		// initialize output buffer
	#endif
		GPU_SYNC();

		return 0;
	}

	__HOST__ int
	get_ses(char *rite_name)
	{
		guru_ses *ses = (guru_ses *)malloc(sizeof(guru_ses));
		if (!ses) return -3;

		char *ins = _fetch_bytecode(rite_name);
		if (!ins) return -4;

		int id = ses->id = _vm_pool->get(ins);
		cuda_free(ins);

		if (id>=0) {
			ses->stdout = _guru_out;		// assign session output buffer
			ses->next   = _ses_list;		// add to linked-list
			_ses_list   = ses;
		}
		return id;
	}

	__HOST__ int
	run()
	{
		debug_log("guru session starting...");
		debug_mmu_stat();

		// parse BITE code into each vm
		// TODO: work producer (enqueue)
		_vm_pool->start();

		debug_mmu_stat();
		debug_log("guru session completed.");

		return 0;
	}
};

__HOST__
Guru::Guru(int step, int trace) : _impl(new Impl())
{
	debug_init(trace);												// initialize logger
	debug_log("guru initializing...");

	int rst = _impl->init(step);

	switch (rst) {
	case -1: fprintf(stderr, "ERROR: failed to allocate device main memory block!\n"); 	break;
	case -2: fprintf(stderr, "ERROR: output buffer allocation error!\n"); 				break;
	case -3: fprintf(stderr, "ERROR: VM memory block allocation error!\n");				break;
	default: break;
	}

	if (rst) {
		debug_log("guru initialized failed, bailing out...");
		exit(-1);
	}
	else {
		U32 sz0, sz1;
		cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
		cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
		cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

		debug_log("guru initialized, ready to go...");
	}
}

__HOST__
Guru::~Guru() = default;

__HOST__ int
Guru::load(char *rite_name)
{
	debug_log("guru loading RITE image into ses->stdin memory...");

	int id = _impl->get_ses(rite_name);

	if (id>=0) return 0;

	switch (id) {
	case -1: fprintf(stderr, "ERROR: bytecode parsing error!\n");				break;
	case -2: fprintf(stderr, "ERROR: No more VM available!\n");					break;
	case -3: fprintf(stderr, "ERROR: session memory allocation error!\n");		break;
	case -4: fprintf(stderr, "ERROR: bytecode memory allocation error!\n"); 	break;
	default: break;
	}
	return 1;
}

__HOST__ int
Guru::run()
{
	return _impl->run();
}
