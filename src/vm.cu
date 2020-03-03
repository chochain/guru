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
#include <stdio.h>
#include <pthread.h>

#include "mmu.h"
#include "class.h"
#include "state.h"
#include "symbol.h"
#include "vmx.h"
#include "vm_debug.h"

#include "ucode.h"
#include "c_string.h"

guru_vm *_vm_pool;

pthread_mutex_t 	_mutex_pool;
#define _LOCK		(pthread_mutex_lock(&_mutex_pool))
#define _UNLOCK		(pthread_mutex_unlock(&_mutex_pool))

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__GURU__ void
_ready(guru_vm *vm, guru_irep *irep)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;		// single threaded

	GV *r = vm->regfile;
	for (U32 i=0; i<MAX_REGFILE_SIZE; i++, r++) {		// wipe register
		r->gt  = (i>0) ? GT_EMPTY : GT_CLASS;			// reg[0] is "self"
		r->cls = (i>0) ? NULL     : guru_rom_get_class(GT_OBJ);
		r->acl = 0;
	}
    vm->state = NULL;
    vm->run   = VM_STATUS_READY;
    vm->depth = vm->err = 0;

    vm_state_push(vm, irep, 0, vm->regfile, 0);
}

__GURU__ void
_free(guru_vm *vm)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;		// single threaded
	if (vm->run!=VM_STATUS_STOP) 		 return;

	while (vm->state) {									// pop off call stack
		vm_state_pop(vm, vm->state->regs[1]);
	}
	vm->run = VM_STATUS_FREE;							// release the vm
}

//================================================================
// Transcode Pooled objects and Symbol table recursively
// from source memory pointers to GV[] (i.e. regfile)
//
__GURU__ void
__transcode(guru_irep *irep)
{
	GV *p = irep->pool;
	for (U32 i=0; i < irep->s; i++, p++) {		// symbol table
		*p = guru_sym_new(p->raw);
		p->acl |= ACL_READ_ONLY;				// rom-based
	}
	for (U32 i=0; i < irep->p; i++, p++) {		// pooled objects
		if (p->gt==GT_STR) {
			*p = guru_str_new(p->raw);
		}
		p->acl |= ACL_READ_ONLY;				// rom-based
	}
	// use tail recursion (i.e. no call stack, so compiler optimization becomes possible)
	for (U32 i=0; i < irep->r; i++) {
		__transcode(irep->reps[i]);
	}
}

//================================================================
// Fetch a VM for operation
// Note: thread 0 is the master controller, no other thread can
//       modify the VM status
//
__GPU__ void
_fetch(guru_vm *pool, guru_irep *irep, int *vid)
{
	*vid = -1;

	if (blockIdx.x!=0 || threadIdx.x!=0) return;	// single threaded
	if (!pool) return;								// not initialized yet

	guru_vm *vm = pool;
	int idx;
	for (idx=0; idx<MIN_VM_COUNT; idx++, vm++) {
		if (vm->run==VM_STATUS_FREE) {
			vm->run = VM_STATUS_READY;		// reserve the VM
			break;
		}
	}
	if (idx>=MIN_VM_COUNT) return;

	__transcode(irep);		// recursively transcode Pooled objects and Symbol table
	_ready(vm, irep);

	*vid = idx;
}

//================================================================
/*!@brief
  execute one ISEQ instruction for each VM

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GPU__ void
_step(guru_vm *vm)
{
	if (threadIdx.x!=0) return;						// TODO: single thread for now

	// start up instruction and dispatcher unit
	while (vm->run==VM_STATUS_RUN) {				// run my (i.e. blockIdx.x) VM
		// add before_fetch hooks here
		ucode_prefetch(vm);
		// add before_exec hooks here
		ucode_exec(vm);
		// add after_exec hooks here
		if (vm->step) break;
	}
	if (vm->run==VM_STATUS_STOP) {					// whether my VM is completed
		_free(vm);									// free up my vm_state, return VM to free pool
	}
}

#if !GURU_HOST_IMAGE
//================================================================
/*!@brief
  release mrbc_irep holds memory
*/
__GURU__ void
_mrbc_irep_free(mrbc_irep *irep)
{
    // release pool.
    for (U32 i=0; i < irep->plen; i++) {
        guru_free(irep->pool[i]);
    }
    if (irep->plen) guru_free(irep->pool);

    // release all child ireps.
    for (U32 i=0; i < irep->rlen; i++) {
        _mrbc_irep_free(irep->list[i]);
    }
    if (irep->rlen) guru_free(irep->list);
    guru_free(irep);
}
#endif

__HOST__ int
vm_pool_init(U32 step)
{
#if GURU_HOST_IMAGE
	guru_vm *vm = _vm_pool = (guru_vm *)cuda_malloc(sizeof(guru_vm) * MIN_VM_COUNT, 1);
	if (!vm) return -1;

	for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
		vm->id    = i;
		vm->step  = step;
		vm->depth = vm->err  = 0;
		vm->run   = VM_STATUS_FREE;		// VM not allocated
		cudaStreamCreateWithFlags(&vm->st, cudaStreamNonBlocking);
	}
#else
	mrbc_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);
	if (!vm) return cudaErrorMemoryAllocation;

	mrbc_parse_bytecode<<<1,1>>>(vm, ses->in);
	SYNC();

	if (ses->trace) {
		printf("  vm[%d]: %p\n", vm->id, (void *)vm);
		mrbc_show_irep(vm->irep);
	}
#endif
	return 0;
}

__HOST__ int
_join() {
	U32 i;
	guru_vm *vm = _vm_pool;

	_LOCK;											// TODO: make it a per-VM (i.e. per-blockIdx) control
	for (i=0; i<MIN_VM_COUNT; i++, vm++) {
		if (vm->run==VM_STATUS_RUN) break;
	}
	_UNLOCK;

	return (i<MIN_VM_COUNT) ? 1 : 0;
}

__HOST__ int
vm_main_start()
{
	do {
		// add pre-hook here
		guru_vm *vm = _vm_pool;
		for (int i=0; i<MIN_VM_COUNT; i++, vm++) {
			if (vm->run != VM_STATUS_RUN) continue;

			debug_disasm(i);
			_step<<<1,1,0,vm->st>>>(vm);			// guru -x to run without single-stepping
		}
		SYNC();										// TODO: cooperative thread group

#if GURU_USE_CONSOLE
		guru_console_flush(ses->out, ses->trace);	// dump output buffer
#endif  // GURU_USE_CONSOLE
		// add post-hook here
	} while(_join());								// GPU device barrier + HOST pthread guard

	return 0;
}

__HOST__ int
vm_get(U8 *irep_img)
{
	guru_irep *irep = (guru_irep *)irep_img;
	void *vm_id;
	int  vid;

	cudaMallocManaged(&vm_id, sizeof(int));			// allocate device memory, auto synchronize

	_LOCK;
	_fetch<<<1,1>>>(_vm_pool, irep, (int*)vm_id);	// use default stream, vm status will changed
	SYNC();
	vid = *(int*)vm_id;
	_UNLOCK;

	cudaFree(vm_id);								// free memory, auto synchronize

	debug_vm_irep(vid);

	return vid;
}

__HOST__ int
_set_status(U32 vid, U32 new_status, U32 status_flag)
{
	guru_vm *vm = _vm_pool + vid;
	if (!(vm->run & status_flag)) return -1;		// state machine

	_LOCK;
	vm->run = new_status;
	_UNLOCK;

	return 0;
}

__HOST__ int vm_ready(U32 vid) { return _set_status(vid, VM_STATUS_RUN,  VM_STATUS_READY); }
__HOST__ int vm_hold(U32 vid)  { return _set_status(vid, VM_STATUS_HOLD, VM_STATUS_RUN);   }
__HOST__ int vm_stop(U32 vid)  { return _set_status(vid, VM_STATUS_STOP, VM_STATUS_RUN);   }

