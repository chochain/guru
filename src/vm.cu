/*! @file
  @brief
  GURU instruction unit
    1. guru VM, host or cuda image, constructor and dispatcher
    2. dumpers for regfile and irep tree

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  initialize VM
  	  allocate vm cuda memory
  	  parse mruby bytecodee
  	  dump irep tree (optional)
  execute VM
  	  opcode execution loop (on GPU)
  	    ucode_prefetch
  	    ucode_exec
  	    flush output per step (optional)
  </pre>
*/
#include <pthread.h>

#include "guru.h"
#include "util.h"
#include "mmu.h"
#include "symbol.h"
#include "c_string.h"

#include "class.h"
#include "state.h"
#include "vmx.h"
#include "debug.h"
#include "load.h"

#include "ucode.h"

guru_vm *_vm_pool;
U32      _vm_cnt = 0;

pthread_mutex_t 	_mutex_pool;
#define _LOCK		(pthread_mutex_lock(&_mutex_pool))
#define _UNLOCK		(pthread_mutex_unlock(&_mutex_pool))

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__GURU__ void
__ready(guru_vm *vm, GRIT *gr)
{
	GV *r = vm->regfile;
	for (U32 i=0; i<MAX_REGFILE_SIZE; i++, r++) {					// wipe register
		r->gt  = (i>0) ? GT_EMPTY : GT_CLASS;						// reg[0] is "self"
		r->cls = (i>0) ? NULL     : guru_rom_get_class(GT_OBJ);
		r->acl = 0;
	}
    vm->state = NULL;
    vm->run   = VM_STATUS_READY;
    vm->depth = vm->err = 0;

    vm_state_push(vm, gr->reps, 0, vm->regfile, 0);
}

__GURU__ void
__free(guru_vm *vm)
{
	if (vm->run!=VM_STATUS_STOP) return;

	while (vm->state) {								// pop off call stack
		vm_state_pop(vm, vm->state->regs[1]);
	}
	vm->run   = VM_STATUS_FREE;						// release the vm
	vm->state = NULL;								// redundant?
}

//================================================================
// Transcode Pooled objects and Symbol table recursively
// from source memory pointers to GV[] (i.e. regfile)
//
__GURU__ void
__transcode(GRIT *gr)
{
	GV *v = gr->pool;
	for (U32 i=0; i < gr->psz; i++, v++) {			// symbol table
		switch (v->gt) {
		case GT_SYM: guru_sym_rom(v);	break;
		case GT_STR: guru_str_rom(v);	break;		// instantiate the string
		default:
			// do nothing
		}
	}
}

//================================================================
// Fetch a VM for operation
// Note: thread 0 is the master controller, no other thread can
//       modify the VM status
//
#if GURU_HOST_IMAGE
__GPU__ void
_get(guru_vm *vm, guru_irep *irep)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;	// singleton thread

	if (vm->run==VM_STATUS_FREE) {
		__transcode(irep);
		__ready(vm, irep);
	}
}
#else
__GPU__ void
_get(guru_vm *vm, U8 *ibuf)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;	// singleton thread

	if (vm->run==VM_STATUS_FREE) {
		GRIT *gr = parse_bytecode(ibuf);
		__transcode(gr);
		__ready(vm, gr);
	}
}
#endif // GURU_HOST_IMAGE

//================================================================
/*!@brief
  execute one ISEQ instruction for each VM

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GPU__ void
_exec(guru_vm *vm)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;	// TODO: single thread for now

#if GURU_CXX_CODEBASE
 	Ucode uc(vm);

	if (uc.run()) {
		__free(vm);
	}
	return;
#else
	// start up instruction and dispatcher unit
	while (vm->run==VM_STATUS_RUN) {				// run my (i.e. blockIdx.x) VM
		// add before_fetch hooks here
		ucode_prefetch(vm);
		// add before_exec hooks here
		ucode_step(vm);
		// add after_exec hooks here
		if (vm->step) break;
	}
	if (vm->run==VM_STATUS_STOP) {					// whether my VM is completed
		__free(vm);									// free up my vm_state, return VM to free pool
	}
#endif // GURU_USE_CXX
}

__HOST__ int
vm_pool_init(U32 step)
{
	guru_vm *vm = _vm_pool = (guru_vm *)cuda_malloc(sizeof(guru_vm) * MIN_VM_COUNT, 1);
	if (!vm) return -1;

	for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
		vm->id    = i;
		vm->step  = step;
		vm->depth = vm->err  = 0;
		vm->run   = VM_STATUS_FREE;		// VM not allocated
		cudaStreamCreateWithFlags(&vm->st, cudaStreamNonBlocking);
	}
	return 0;
}

__HOST__ int
_has_job() {
	guru_vm *vm = _vm_pool;
	for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
		if (vm->run==VM_STATUS_RUN) return 1;
	}
	return 0;
}

__HOST__ int
vm_main_start()
{
	// TODO: spin off as a server thread
	do {
		guru_vm *vm = _vm_pool;
		for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
			if (!vm->state) continue;
			// add pre-hook here
			if (debug_disasm(vm)) break;
			_exec<<<1,1,sizeof(guru_vm),vm->st>>>(vm);	// guru -x to run without single-stepping
			// add post-hook here
		}
		GPU_SYNC();											// TODO: cooperative thread group
#if GURU_USE_CONSOLE
		guru_console_flush(ses->out, ses->trace);		// dump output buffer
#endif  // GURU_USE_CONSOLE
	} while (_has_job());								// join()

	return 0;
}

__HOST__ int
vm_get(U8 *ibuf)
{
	if (!_vm_pool) 				return -1;
	if (_vm_cnt>=MIN_VM_COUNT) 	return -1;

	guru_vm *vm = &_vm_pool[_vm_cnt];

#if GURU_HOST_IMAGE
	guru_irep *irep = (guru_irep *)parse_bytecode(ibuf);
	if (!irep) return -2;

	_get<<<1,1,0,vm->st>>>(vm, irep);
#else
	_get<<<1,1,0,vm->st>>>(vm, ibuf);			// acquire VM, vm status will changed
#endif // GURU_HOST_IMAGE
	GPU_SYNC();
	debug_vm_irep(vm);

	return _vm_cnt++;
}

__HOST__ int
_set_status(U32 mid, U32 new_status, U32 status_flag)
{
	guru_vm *vm = &_vm_pool[mid];
	if (!(vm->run & status_flag)) return -1;	// transition state machine

	_LOCK;
	vm->run = new_status;
	_UNLOCK;

	return 0;
}

__HOST__ int vm_ready(U32 mid) { return _set_status(mid, VM_STATUS_RUN,  VM_STATUS_READY); }
__HOST__ int vm_hold(U32 mid)  { return _set_status(mid, VM_STATUS_HOLD, VM_STATUS_RUN);   }
__HOST__ int vm_stop(U32 mid)  { return _set_status(mid, VM_STATUS_STOP, VM_STATUS_RUN);   }
