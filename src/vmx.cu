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
#include "ucode.h"
#include "state.h"
#include "load.h"
#include "debug.h"

#define _LOCK		(pthread_mutex_trylock(&_mutex))
#define _UNLOCK		(pthread_mutex_unlock(&_mutex))

__GPU__ void _vm_init(VM *vm, int i, int step)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	vm->init(i, step);
}

__GPU__ void _vm_exec(VM *vm)
{
	static Ucode *_uc_pool[MIN_VM_COUNT] = { NULL, NULL };

	int b = blockIdx.x;
	if (b || threadIdx.x!=0) return;

	extern __shared__ Ucode uc[];							// TODO: move VM into shared memory, too

	if (!_uc_pool[b]) {										// lazy allocation
		_uc_pool[b] = new Ucode(vm);						// microcode execution unit
	}
	uc[b] = *_uc_pool[b];									// copy into shared memory
	__syncthreads();

	if (uc[b].run()) {										// whether my VM is completed
		StateMgr *sm = new StateMgr(vm);					// needs a helper
		sm->free_states();
	}
}

__GPU__ void _vm_load_grit(VM *vm, U8 *u8_gr)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	vm->load_grit(u8_gr);
}

class VM_Pool::Impl
{
	VM 	*_pool = NULL;
	U32	_idx   = 0;

	pthread_mutex_t _mutex;
	cudaStream_t    _st[MIN_VM_COUNT];						// a stream per each VM

	int
	_has_job()
	{
    	VM *vm = _pool;
    	for (int i=0; i<MIN_VM_COUNT; i++, vm++) {
    		if (vm->state && vm->run==VM_STATUS_RUN && !vm->err) return 1;
    	}
    	return 0;
	}

	__HOST__ int
	_set_status(U32 mid, U32 new_status, U32 status_flag)
	{
		VM *vm = &_pool[mid];
		if (!(vm->run & status_flag)) return -1;		// transition state machine

		if (_LOCK) {
			vm->run = new_status;
			_UNLOCK;
		}
		else {
			fprintf(stderr, "ERROR: set_status failed, thread do not have the lock\n");
			return -1;
		}
		return 0;
	}

	__HOST__ int ready(U32 mid) { return _set_status(mid, VM_STATUS_RUN,  VM_STATUS_READY); }
	__HOST__ int hold(U32 mid)  { return _set_status(mid, VM_STATUS_HOLD, VM_STATUS_RUN);   }
	__HOST__ int stop(U32 mid)  { return _set_status(mid, VM_STATUS_STOP, VM_STATUS_RUN);   }

public:
	Impl(U32 step)
	{
		VM *vm = _pool = (VM*)cuda_malloc(sizeof(VM) * MIN_VM_COUNT, 1);

		for (int i=0; vm && i<MIN_VM_COUNT; i++, vm++) {
			cudaStreamCreateWithFlags(&_st[i], cudaStreamNonBlocking);
			_vm_init<<<1,1, 0, _st[i]>>>(vm, i, step);
		}
		GPU_SYNC();
	}

	~Impl()
	{
		for (int i=0; i<MIN_VM_COUNT; i++) {
			cudaStreamDestroy(_st[i]);
		}
	}

	__HOST__ S32
	vm_main_loop()
	{
		// TODO: spin off as a server thread
		do {
			VM *vm = _pool;
			for (int i=0; i<MIN_VM_COUNT; i++, vm++) {		// TODO: parallel
				if (!vm->state || vm->run!=VM_STATUS_RUN) continue;
				// add pre-hook here
				debug_disasm((guru_vm*)vm);

				U32 bsz = sizeof(Ucode)*MIN_VM_COUNT;
				_vm_exec<<<1,1,bsz,_st[i]>>>(vm);			// guru -x to run without single-stepping

				cudaError_t e = cudaGetLastError();
				if (e) {
					printf("CUDA ERROR: %s, bailing\n", cudaGetErrorString(e));
					vm->err = 1;
				}
				// add post-hook here
			}
			GPU_SYNC();										// TODO: cooperative thread group
#if GURU_USE_CONSOLE
			guru_console_flush(ses->out, ses->trace);		// dump output buffer
#endif  // GURU_USE_CONSOLE
		} while (_has_job());								// join()

		return 0;
	}

	__HOST__ S32
	vm_get(U8 *ibuf)
	{
		if (!_pool) 				return -1;
		if (_idx>=MIN_VM_COUNT) 	return -1;

		VM *vm = &_pool[_idx];

#if GURU_HOST_GRIT_IMAGE
		U8 *gr = (U8*)parse_bytecode(ibuf);
		if (!gr) 					return -2;

		_vm_load_grit<<<1,1>>>(vm, gr);
#else
		_load_grit<<<1,1,0,vm->st>>>(vm, ibuf);				// acquire VM, vm status will changed
#endif // GURU_HOST_GRIT_IMAGE
		GPU_SYNC();

		debug_vm_irep((guru_vm*)vm);
		if (ready(_idx)) 			return -3;

		return _idx++;
	}
};

__HOST__ VM_Pool::VM_Pool(int step) : _impl(new Impl((U32)step)) {}
__HOST__ VM_Pool::~VM_Pool() = default;

__HOST__ int
VM_Pool::start()
{
	return _impl->vm_main_loop();
}

__HOST__ int
VM_Pool::get(char *ibuf)
{
	return _impl->vm_get((U8*)ibuf);
}
