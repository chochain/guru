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

#include "alloc.h"
#include "static.h"
#include "ucode.h"
#include "load.h"
#include "state.h"
#include "vmx.h"
#include "vm.h"

#include "puts.h"

guru_vm *_vm_pool;

pthread_mutex_t 	_mutex_pool;
#define _LOCK		(pthread_mutex_lock(&_mutex_pool))
#define _UNLOCK		(cudaDeviceSynchronize(), pthread_mutex_unlock(&_mutex_pool))

__HOST__ void _show_irep(guru_irep *irep, U32 ioff, char level, char *idx);		// forward declaration
__HOST__ void _trace(U32 level);												// forward declaration

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__GURU__ void
_ready(guru_vm *vm, guru_irep *irep)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;		// single threaded

	MEMSET(vm->regfile, 0, sizeof(vm->regfile));		// clean up registers

	vm->regfile[0].gt  = GT_CLASS;						// regfile[0] is self
    vm->regfile[0].cls = guru_class_object;				// root class

    vm->state = NULL;
    vm->run   = VM_STATUS_READY;
    vm->err   = 0;

    vm_state_push(vm, irep, vm->regfile, 0);
}

__GURU__ void
_free(guru_vm *vm)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;		// single threaded
	if (vm->run!=VM_STATUS_STOP) 		 return;

	while (vm->state) {									// pop off call stack
		vm_state_pop(vm, vm->state->regs[1], 0);
	}
	vm->run = VM_STATUS_FREE;							// release the vm
}

__GPU__ void
_fetch(guru_vm *pool, guru_irep *irep, int *vid)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;		// single threaded

	guru_vm *vm = pool;
	int idx;

	for (idx=0; idx<MIN_VM_COUNT; idx++, vm++) {
		if (vm->run==VM_STATUS_FREE) {
			vm->run = VM_STATUS_READY;				// reserve the VM
			break;
		}
	}
	if (idx<MIN_VM_COUNT) _ready(vm, irep);
	else 				  idx = -1;

	*vid = idx;
}

//================================================================
/*!@brief
  execute one ISEQ instruction for each VM

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GPU__ void
_step(guru_vm *pool)
{
	if (threadIdx.x != 0) return;					// TODO: single thread for now

	guru_vm *vm = pool+blockIdx.x;					// start up all VMs (with different blockIdx

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
	__syncthreads();								// sync all cooperating threads (to shared data)
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
		vm->temp16= 0xeeee;
		vm->temp32= 0xeeeeeeee;
		vm->run   = VM_STATUS_FREE;		// VM not allocated
	}
#else
	mrbc_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);
	if (!vm) return cudaErrorMemoryAllocation;

	mrbc_parse_bytecode<<<1,1>>>(vm, ses->in);
	cudaDeviceSynchronize();
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

	_LOCK;
	for (i=0; i<MIN_VM_COUNT; i++, vm++) {
		if (vm->run==VM_STATUS_RUN) break;
	}
	_UNLOCK;

	return (i<MIN_VM_COUNT) ? 1 : 0;
}

__HOST__ int
vm_main_start(U32 trace)
{
	if (trace) printf("guru_session starting...\n");

	// TODO: pthread
	do {
		_trace(trace);
		cudaDeviceSynchronize();

		_step<<<MIN_VM_COUNT, 1>>>(_vm_pool);
		cudaDeviceSynchronize();

	// add host hook here
#if GURU_USE_CONSOLE
		guru_console_flush(ses->out, ses->trace);	// dump output buffer
#endif
	} while (_join());

	if (trace) {
		printf("guru_session completed\n");
		guru_dump_alloc_stat(trace);
	}
	return 0;
}

__HOST__ int
vm_get(U8 *irep_img, U32 trace)
{
	guru_irep *irep = (guru_irep *)irep_img;
	int *vid = (int *)cuda_malloc(sizeof(int), 1);

	_LOCK;
	_fetch<<<1,1>>>(_vm_pool, irep, vid);
	_UNLOCK;

	if (trace) {
		if (vid<0) {
			printf("ERROR: no vm available!");
		}
		else {
			printf("  vm[%d]:\n", *vid);
			char c = 'a';
			_show_irep(irep, 0, 'A', &c);
		}
	}
	return *vid;		// CUDA memory leak?
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

__HOST__ int vm_run(U32 vid)  { return _set_status(vid, VM_STATUS_RUN,  VM_STATUS_READY); }
__HOST__ int vm_hold(U32 vid) { return _set_status(vid, VM_STATUS_HOLD, VM_STATUS_RUN);   }
__HOST__ int vm_stop(U32 vid) { return _set_status(vid, VM_STATUS_STOP, VM_STATUS_RUN);   }

//========================================================================================
// the following code is for debugging purpose, turn off GURU_DEBUG for release
//========================================================================================
#if GURU_DEBUG

static const char *_vtype[] = {
	"___","nil","f  ","t  ","num","flt","sym","cls",	// 0x0
	"prc","","","","","","","",							// 0x8
	"obj","ary","str","rng","hsh"					    // 0x10
};

static const char *_opcode[] = {
    "NOP ",	"MOVE",	"LOADL","LOADI","LOADSYM","LOADNIL","LOADSLF","LOADT",
    "LOADF","GETGBL","SETGBL","","","GETIV","SETIV","",
    "","GETCONS","SETCONS","","","GETUVAR","SETUVAR","JMP ",
    "JMPIF","JMPNOT","","","","","","",
    "SEND","SENDB","","CALL","","","ENTER","",
    "","RETURN","","BLKPUSH","ADD ","ADDI","SUB ","SUBI",
    "MUL ","DIV ","EQ  ","LT  ","LE  ","GT  ","GE  ","ARRAY",
    "","","","","","STRING","STRCAT","HASH",
    "LAMBDA","RANGE","","CLASS","","EXEC","METHOD","",
    "TCLASS","","STOP","","","","","",
    "ABORT"
};

__HOST__ int
_match_irep(guru_irep *irep0, guru_irep *irep1, U8P idx)
{
	if (irep0==irep1) return 1;

	U8P  base = (U8P)irep0;
	U32P off  = (U32P)U8PADD(base, irep0->reps);		// child irep offset array
	for (U32 i=0; i<irep0->c; i++) {
		*idx += 1;
		if (_match_irep((guru_irep *)(base + off[i]), irep1, idx)) return 1;
	}
	return 0;		// not found
}

__HOST__ void
_show_irep(guru_irep *irep, U32 ioff, char level, char *idx)
{
	printf("\tirep[%c]=%c%04x: size=%d, nreg=%d, nlocal=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			*idx, level, ioff,
			irep->size, irep->nr, irep->nv, irep->p, irep->s, irep->c, irep->i);

	// dump all children ireps
	U8  *base = (U8 *)irep;
	U32 *off  = (U32 *)U8PADD(base, irep->reps);		// pointer to irep offset array
	for (U32 i=0; i<irep->c; i++) {
		*idx += 1;
		_show_irep((guru_irep *)(base + off[i]), off[i], level+1, idx);
	}
}

__HOST__ void
_show_regfile(guru_vm *vm, U32 lvl)
{
	U32 n;
	GV  *v = &vm->regfile[MAX_REGS_SIZE-1];
	for (n=MAX_REGS_SIZE-1; n>0; n--, v--) {
		if (v->gt!=GT_EMPTY) break;
	}

	v = vm->regfile;
	printf("[");
	for (U32 i=0; i<=n; i++, v++) {
		const char *t = _vtype[v->gt];
		U8 c = i==lvl ? '|' : ' ';
		if (v->gt & GT_HAS_REF)	printf("%s%d%c", t, v->self->rc, c);
		else					printf("%s %c",  t, c);
    }
	printf("]");
}

#define bin2u32(x) ((x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24))

__HOST__ void
_show_ucode(guru_vm *vm)
{
	U16  pc    = vm->state->pc;						// program counter
	U32  *iseq = (U32*)VM_ISEQ(vm);
	U32  code  = bin2u32(*(iseq + pc));				// convert to big endian
	U16  op    = code & 0x7f;       				// in HOST mode, GET_OPCODE() is DEVICE code
	U8P  opc   = (U8P)_opcode[GET_OP(op)];

	guru_state *st    = vm->state;
	guru_irep  *irep1 = st->irep;

	U8 idx = 'a';
	if (st->prev) {
		if (!_match_irep(st->prev->irep, irep1, &idx)) idx='?';
	}
	printf("%1d%c%-4d%-8s", vm->id, idx, pc, opc);

	U32 lvl=0;
	while (st->prev != NULL) {
		st = st->prev;
		lvl += 2 + st->argc;
	}
	_show_regfile(vm, lvl);

	if (op==OP_SEND || op==OP_SENDB) {				// display function name
		U32 rb = (code >> 14) & 0x1ff;
		printf(" #%s", VM_SYM(vm, rb));
	}
}

__HOST__ void
_trace(U32 level)
{
	if (level==0) return;

	guru_vm *vm = _vm_pool;
	for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
		if (vm->run==VM_STATUS_RUN && vm->step) {
			guru_state *st = vm->state;
			while (st->prev) {
				printf("  ");
				st = st->prev;
			}
			_show_ucode(vm);
			printf("\n");
		}
	}
	if (level>1) guru_dump_alloc_stat(level);
}

#else
__HOST__ void _show_irep(guru_irep *irep, U32 ioff, char level, char *idx) {}
__HOST__ void _trace(U32 level) {}
#endif 	// GURU_DEBUG

