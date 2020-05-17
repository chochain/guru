/*! @file
  @brief
  GURU VM pool implementation

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  initialize VM
  	  allocate vm cuda memory
  execute
      start main loop
         invoke every VM (in serial or parallel)
  </pre>
*/
#include <pthread.h>

#include "guru.h"
#include "util.h"
#include "mmu.h"
#include "vm.h"
#include "vmx.h"
#include "load.h"
#include "debug.h"

#define _LOCK		(pthread_mutex_lock(&_mutex))
#define _UNLOCK		(pthread_mutex_unlock(&_mutex))

class VM_Pool::Impl
{
	guru_vm *_vm_pool = NULL;
	U32		_vm_cnt   = 0;

	pthread_mutex_t _mutex;

	int
	_has_job()
	{
    	guru_vm *vm = _vm_pool;
    	for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
    		if (vm->run==VM_STATUS_RUN && !vm->err) return 1;
    	}
    	return 0;
	}

	__HOST__ int
	_set_status(U32 mid, U32 new_status, U32 status_flag)
	{
		guru_vm *vm = &_vm_pool[mid];
		if (!(vm->run & status_flag)) return -1;		// transition state machine

		_LOCK;
		vm->run = new_status;
		_UNLOCK;

		return 0;
	}

	__HOST__ int vm_ready(U32 mid) { return _set_status(mid, VM_STATUS_RUN,  VM_STATUS_READY); }
	__HOST__ int vm_hold(U32 mid)  { return _set_status(mid, VM_STATUS_HOLD, VM_STATUS_RUN);   }
	__HOST__ int vm_stop(U32 mid)  { return _set_status(mid, VM_STATUS_STOP, VM_STATUS_RUN);   }

public:
	Impl(U32 step)
	{
		guru_vm *vm = _vm_pool = (guru_vm*)cuda_malloc(sizeof(guru_vm) * MIN_VM_COUNT, 1);

		for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
			vm->id    = i;
			vm->step  = step;
			vm->depth = vm->err  = 0;
			vm->run   = VM_STATUS_FREE;					// VM not allocated
			cudaStreamCreateWithFlags(&vm->st, cudaStreamNonBlocking);
		}
	}

	__HOST__ S32
	vm_main_start()
	{
		// TODO: spin off as a server thread
		do {
			guru_vm *vm = (guru_vm*)_vm_pool;
			for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {	// TODO: parallel
				if (!vm->state || vm->run!=VM_STATUS_RUN) continue;
				// add pre-hook here
				if (debug_disasm(vm)) {
					vm->err = 1;						// stop a run-away loop
				}
				else {
					//_exec<<<1,1,0,vm->st>>>(vm);		// guru -x to run without single-stepping
				}
				cudaError_t e = cudaGetLastError();
				if (e) {
					printf("CUDA ERROR: %s, bailing\n", cudaGetErrorString(e));
					vm->err = 1;
				}
				// add post-hook here
			}
			GPU_SYNC();									// TODO: cooperative thread group
#if GURU_USE_CONSOLE
			guru_console_flush(ses->out, ses->trace);	// dump output buffer
#endif  // GURU_USE_CONSOLE
		} while (_has_job());							// join()

		return 0;
	}

	__HOST__ S32
	vm_get(U8 *ibuf)
	{
		if (!_vm_pool) 				return -1;
		if (_vm_cnt>=MIN_VM_COUNT) 	return -1;

		guru_vm *vm = &_vm_pool[_vm_cnt];

#if GURU_HOST_GRIT_IMAGE
		GRIT *gr = (GRIT*)parse_bytecode(ibuf);
		if (!gr) return -2;

		//_prep<<<1,1,0,vm->st>>>(vm, gr);
#else
		_prep<<<1,1,0,vm->st>>>(vm, ibuf);				// acquire VM, vm status will changed
#endif // GURU_HOST_GRIT_IMAGE
		GPU_SYNC();

		debug_vm_irep(vm);

		return _vm_cnt++;
	}
};

__HOST__ VM_Pool::VM_Pool(int step) : _impl(new Impl((U32)step)) {}
__HOST__ VM_Pool::~VM_Pool() = default;

__HOST__ int
VM_Pool::start()
{
	return _impl->vm_main_start();
}

__HOST__ int
VM_Pool::get(char *ibuf)
{
	return _impl->vm_get((U8*)ibuf);
}
