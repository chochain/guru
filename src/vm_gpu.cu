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

#if !GURU_HOST_IMAGE
__GURU__ guru_vm *_vm_pool;
__GURU__ U32 _vm_cnt = 0;

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__GURU__ void
__ready(guru_vm *vm, guru_irep *irep)
{
	GV *r = vm->regfile;
	for (U32 i=0; i<MAX_REGFILE_SIZE; i++, r++) {	// wipe register
		r->gt  = (i>0) ? GT_EMPTY : GT_CLASS;		// reg[0] is "self"
		r->cls = (i>0) ? NULL     : guru_rom_get_class(GT_OBJ);
		r->acl = 0;
	}
    vm->state = NULL;
    vm->run   = VM_STATUS_READY;
    vm->depth = vm->err = 0;

    vm_state_push(vm, irep, 0, vm->regfile, 0);
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
__transcode(guru_irep *irep)
{
	GV *p = irep->pool;
	for (U32 i=0; i < irep->s; i++, p++) {			// symbol table
		*p = guru_sym_new(p->raw);
		p->acl |= ACL_READ_ONLY;					// rom-based
	}
	for (U32 i=0; i < irep->p; i++, p++) {			// pooled objects
		if (p->gt==GT_STR) {
			*p = guru_str_new(p->raw);
		}
		p->acl |= ACL_READ_ONLY;					// rom-based
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
_get(guru_vm *vm, guru_irep *irep)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;	// singleton thread

	if (vm->run==VM_STATUS_FREE) {
		__transcode(irep);							// recursively transcode Pooled objects and Symbol table
		__ready(vm, irep);
	}
}
//================================================================
/*!@brief
  execute one ISEQ instruction for each VM

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GURU__ void
_exec(guru_vm *vm)
{
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
}

__GPU__ void
_init(guru_vm *pool, U32 step)
{
	U32 i=threadIdx.x;
	if (i>=MIN_VM_COUNT) return;

	guru_vm *vm = &pool[i];

	vm->id    = i;
	vm->step  = step;
	vm->depth = vm->err  = 0;
	vm->run   = VM_STATUS_FREE;		// VM not allocated

	cudaStreamCreateWithFlags(&vm->st, cudaStreamNonBlocking);
}

__GPU__ void
_has_job(U32 *rst) {
	U32 i = threadIdx.x;
	if (i<MIN_VM_COUNT) {
		guru_vm *vm = &_vm_pool[i];
		if (vm->run==VM_STATUS_RUN) *rst = 1;
	}
	__syncthreads();
}

__GPU__ void
_main_start()
{
	U32 i = threadIdx.x;
	guru_vm *vm = &_vm_pool[i];

	if (i<MIN_VM_COUNT && vm->state) {
		// add pre-hook here
		//debug_disasm(vm);
		_exec(vm);									// guru -x to run without single-stepping
		// add post-hook here
	}
	__syncthreads();

#if GURU_USE_CONSOLE
	guru_console_flush(ses->out, ses->trace);	// dump output buffer
#endif  // GURU_USE_CONSOLE
}

__GPU__ void
vm_get(U8 *ibuf)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;
	if (_vm_cnt<MIN_VM_COUNT) {
		guru_vm *vm = &_vm_pool[_vm_cnt++];
		vm->state->irep = (guru_irep *)parse_bytecode(ibuf);
	}
	__syncthreads();
}

__HOST__ int
vm_pool_init(U32 step)
{
	guru_vm *pool = (guru_vm *)cuda_malloc(sizeof(guru_vm) * MIN_VM_COUNT, 1);
	_init<<<1,1>>>(pool, step);
	cudaDeviceSynchronize();

	return 0;
}


__HOST__ int
vm_main_start()
{
	U32 *x = (U32*)cuda_malloc(sizeof(U32), 1);
	*x = 0;
	while (x) {
		_main_start<<<1,1>>>();
		cudaDeviceSynchronize();
		_has_job<<<1,1>>>(x);
		cudaDeviceSynchronize();
	}
	cuda_free(x);

	return 0;
}

__GPU__ void
_set_status(U32 vid, U32 new_status, U32 status_flag)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	guru_vm *vm = &_vm_pool[vid];
	ASSERT(vm->run & status_flag);			// transition state machine
	vm->run = new_status;
}

__HOST__ int vm_ready(U32 vid) {
	_set_status<<<1,1>>>(vid, VM_STATUS_RUN,  VM_STATUS_READY); cudaDeviceSynchronize(); return 0;
}
__HOST__ int vm_hold(U32 vid)  {
	_set_status<<<1,1>>>(vid, VM_STATUS_HOLD, VM_STATUS_RUN);   cudaDeviceSynchronize(); return 0;
}
__HOST__ int vm_stop(U32 vid)  {
	_set_status<<<1,1>>>(vid, VM_STATUS_STOP, VM_STATUS_RUN);   cudaDeviceSynchronize(); return 0;
}

#endif // !GURU_HOST_IMAGE
