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
cudaStream_t	_st_pool[MIN_VM_COUNT];

pthread_mutex_t 	_mutex_pool;
#define _LOCK		(pthread_mutex_lock(&_mutex_pool))
#define _UNLOCK		(pthread_mutex_unlock(&_mutex_pool))

#if GURU_CXX_CODEBASE
__GURU__ Ucode *_uc_pool[MIN_VM_COUNT] = { NULL };
#endif // GURU_CXX_CODEBASE
//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__GURU__ void
__ready(guru_vm *vm, GP irep)
{
	GR *r0 = (GR*)MEMPTR(vm->regfile);
	GR *r  = r0;
	for (U32 i=0; i<VM_REGFILE_SIZE; i++, r++) {	// wipe register
		r->gt  = (i==0) ? GT_CLASS : GT_EMPTY;		// reg[0] is "self"
		r->acl = 0;
		r->off = (i==0) ? guru_rom_get_class(GT_OBJ) : 0;
	}
    vm->state = NULL;
    vm->run   = VM_STATUS_READY;
    vm->depth = vm->err = 0;

    vm_state_push(vm, irep, 0, r0, 0);
}

__GURU__ void
__free(guru_vm *vm)
{
	if (vm->run!=VM_STATUS_STOP) return;

	while (vm->state) {								// pop off call stack
		vm_state_pop(vm, _REGS(VM_STATE(vm))[1]);	// passing value of regs[1]
	}
	vm->run   = VM_STATUS_FREE;						// release the vm
	vm->state = NULL;								// redundant?
}

//================================================================
// Transcode Pooled objects and Symbol table recursively
// from source memory pointers to GR[] (i.e. regfile)
//
__GURU__ void
__transcode(U8 *u8_gr)
{
	GRIT *gr = (GRIT*)u8_gr;
	GR   *r  = (GR*)U8PADD(gr, gr->pool);
	for (U32 i=0; i < gr->psz; i++, r++) {			// symbol table
		switch (r->gt) {
		case GT_SYM: guru_sym_rom(r);	break;
		case GT_STR: guru_str_rom(r);	break;		// instantiate the string
		default:
			// do nothing
			break;
		}
	}
}

//================================================================
// Fetch a VM for operation
// Note: thread 0 is the master controller, no other thread can
//       modify the VM status
//
#if GURU_HOST_GRIT_IMAGE
__GPU__ void
_prep(guru_vm *vm,  U8 *u8_gr)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;	// singleton thread

	if (vm->run==VM_STATUS_FREE) {
	    GP irep = MEMOFF(U8PADD(u8_gr, ((GRIT*)u8_gr)->reps));
		__transcode(u8_gr);
		__ready(vm, irep);

#if GURU_CXX_CODEBASE
		if (!_uc_pool[vm->id]) {
			_uc_pool[vm->id] = new Ucode(vm);
		}
#endif // GURU_CXX_CODEBASE
	}
}
#else
__GPU__ void
_prep(guru_vm *vm, U8 *ibuf)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;	// singleton thread

	if (vm->run==VM_STATUS_FREE) {
		U8 *u8_gr = parse_bytecode(ibuf);
	    GP irep   = MEMOFF(U8PADD(gr, gr->reps));
		__transcode(u8_gr);
		__ready(vm, gr);
	}
}
#endif // GURU_HOST_GRIT_IMAGE

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
	extern __shared__ Ucode uc[];

 	uc[blockIdx.x] = *_uc_pool[vm->id];
 	__syncthreads();

	if (uc[blockIdx.x].run()) {
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
		if (vm->step || vm->err) break;
	}
	if (vm->run==VM_STATUS_STOP) {					// whether my VM is completed
		__free(vm);									// free up my vm_state, return VM to free pool
	}
#endif // GURU_USE_CXX
}

__HOST__ int
vm_pool_init(int step)
{
	guru_vm *vm = _vm_pool = (guru_vm *)cuda_malloc(sizeof(guru_vm) * MIN_VM_COUNT, 1);
	GR      *rf = (GR*)cuda_malloc(sizeof(GR) * VM_REGFILE_SIZE * MIN_VM_COUNT, 1);
	if (!vm || !rf) return -1;

	for (U32 i=0; i<MIN_VM_COUNT; i++, vm++, rf+=VM_REGFILE_SIZE) {
		vm->id      = i;
		vm->step    = step;
		vm->depth   = vm->err     = 0;
		vm->run     = VM_STATUS_FREE;					// VM not allocated
		vm->regfile = U8POFF(rf, guru_host_heap);

		cudaStreamCreateWithFlags(&_st_pool[i], cudaStreamNonBlocking);
	}
	return 0;
}

__HOST__ int
_has_job() {
	guru_vm *vm = _vm_pool;
	for (int i=0; i<MIN_VM_COUNT; i++, vm++) {
		if (vm->run==VM_STATUS_RUN && !vm->err) return 1;
	}
	return 0;
}

__HOST__ int
vm_main_start()
{
	// TODO: spin off as a server thread
	do {
		guru_vm *vm = _vm_pool;
		for (int i=0; i<MIN_VM_COUNT; i++, vm++) {
			if (!vm->state || vm->run!=VM_STATUS_RUN) continue;
			// add pre-hook here
			if (debug_disasm(vm)) {
				vm->err = 1;						// stop a run-away loop
			}
			else {
#if GURU_CXX_CODEBASE
				U32 bsz = sizeof(Ucode)*MIN_VM_COUNT;
#else
				U32 bsz = 0;
#endif // GURU_CXX_CODEBASE
				_exec<<<1,1, bsz,_st_pool[i]>>>(vm);		// guru -x to run without single-stepping
			}
			cudaError_t e = cudaGetLastError();
			if (e) {
				printf("CUDA ERROR: %s, bailing\n", cudaGetErrorString(e));
				vm->err = 1;
			}
			// add post-hook here
		}
		GPU_SYNC();								// TODO: cooperative thread group
#if GURU_USE_CONSOLE
		guru_console_flush(ses->out, ses->trace);	// dump output buffer
#endif  // GURU_USE_CONSOLE
	} while (_has_job());							// join()

	return 0;
}

__HOST__ int
vm_get(char *ibuf)
{
	if (!_vm_pool) 				return -1;
	if (_vm_cnt>=MIN_VM_COUNT) 	return -1;

	guru_vm *vm = &_vm_pool[_vm_cnt];

#if GURU_HOST_GRIT_IMAGE
	U8 *gr = parse_bytecode((U8*)ibuf);
	if (!gr) return -2;

	_prep<<<1,1,0,_st_pool[_vm_cnt]>>>(vm, gr);
#else
	_prep<<<1,1,0,_sp_pool[_vm_cnt]>>>(vm, ibuf);				// acquire VM, vm status will changed
#endif // GURU_HOST_GRIT_IMAGE
	GPU_SYNC();

	debug_vm_irep(vm);

	return _vm_cnt++;
}

__HOST__ int
_set_status(int mid, int new_status, int status_flag)
{
	guru_vm *vm = &_vm_pool[mid];
	if (!(vm->run & status_flag)) return -1;		// transition state machine

	_LOCK;
	vm->run = new_status;
	_UNLOCK;

	return 0;
}

__HOST__ int vm_ready(int mid) { return _set_status(mid, VM_STATUS_RUN,  VM_STATUS_READY); }
__HOST__ int vm_hold(int mid)  { return _set_status(mid, VM_STATUS_HOLD, VM_STATUS_RUN);   }
__HOST__ int vm_stop(int mid)  { return _set_status(mid, VM_STATUS_STOP, VM_STATUS_RUN);   }
