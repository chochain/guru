/*! @file
  @brief
  guru instruction unit
    1. guru VM, host or cuda image, constructor and dispatcher
    2. dumpers for regfile and irep tree

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  initialize VM
  	  allocate vm cuda memory
  	  parse mruby bytecodee
  	  dump irep tree (optional)
  execute VM
  	  opcode execution loop (on GPU)
  	  flush output per single-step (optional)
  </pre>
*/
#include <stdio.h>

#include "alloc.h"
#include "static.h"
#include "opcode.h"
#include "load.h"
#include "vmx.h"
#include "vm.h"

#include "puts.h"

__HOST__ cudaError_t _vm_trace(U32 level);		// forward

U32     _vm_pool_ok = 0;
guru_vm *_vm_pool;

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__GPU__ void
_vm_begin(guru_vm *pool)
{
	guru_vm *vm = pool+blockIdx.x;

	if (threadIdx.x!=0 || vm->id==0) return;	// bail if vm not allocated

	MEMSET((U8P)vm->regfile, 0, sizeof(vm->regfile));	// clean up registers

    vm->regfile[0].tt  	= GURU_TT_CLASS;		// regfile[0] is self
    vm->regfile[0].cls 	= mrbc_class_object;	// root class

    guru_state *st = (guru_state *)mrbc_alloc(sizeof(guru_state));

    st->pc 	  = 0;								// starting IP
    st->klass = mrbc_class_object;				// target class
    st->reg   = vm->regfile;					// point to reg[0]
    st->irep  = vm->irep;						// root of irep tree
    st->argc  = 0;
    st->prev  = NULL;							// state linked-list (i.e. call stack)

    vm->state = st;
    vm->run   = 1;
    vm->err   = 0;
}

//================================================================
/*!@brief
  VM finalizer.

  @param  vm  Pointer to VM
*/
__GPU__ void
_vm_end(guru_vm *pool)
{
	guru_vm *vm = pool+blockIdx.x;

	if (threadIdx.x!=0 || vm->id==0) return;		// bail if vm not allocated

#ifndef GURU_DEBUG
	// clean up register file?						// CC: moved from mrbc_op 20181102
	mrbc_value *p = vm->regfile;
	for (U32 i=0; i < MAX_REGS_SIZE; i++, p++) {
		mrbc_release(p);
	}
    mrbc_free_all();
#endif
}

//================================================================
/*!@brief
  Fetch a bytecode and execute

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GPU__ void
_vm_exec(guru_vm *pool)
{
	guru_vm *vm = pool+blockIdx.x;
	if (vm->id==0 || !vm->run) return;		// not allocated yet, or completed

	while (guru_op(vm)==0 && vm->step==0) {	// multi-threading in guru_op
		// add cuda hook here
	}
}

#if !GURU_HOST_IMAGE
//================================================================
/*!@brief
  release mrbc_irep holds memory
*/
__GURU__ void
_mrbc_free_irep(mrbc_irep *irep)
{
    // release pool.
    for (U32 i=0; i < irep->plen; i++) {
        mrbc_free(irep->pool[i]);
    }
    if (irep->plen) mrbc_free(irep->pool);

    // release all child ireps.
    for (U32 i=0; i < irep->rlen; i++) {
        _mrbc_free_irep(irep->list[i]);
    }
    if (irep->rlen) mrbc_free(irep->list);
    mrbc_free(irep);
}
#endif

__HOST__ int
_vm_join(void)
{
	guru_vm *vm = _vm_pool;
	for (U32 i=1; i<=MIN_VM_COUNT; i++, vm++) {
		if (vm->id != 0 && vm->run) return 1;
	}
	return 0;
}

__HOST__ int
_vm_pool_init(void)
{
	guru_vm *vm = _vm_pool = (guru_vm *)guru_malloc(sizeof(guru_vm) * MIN_VM_COUNT, 1);
	if (!vm) return 0;

	for (U32 i=1; i<=MIN_VM_COUNT; i++, vm++) {
		vm->id = vm->step = vm->run = vm->err = 0;
	}
	return MIN_VM_COUNT;
}

__HOST__ cudaError_t
guru_vm_setup(guru_ses *ses, U32 step)
{
#if GURU_HOST_IMAGE
	if (!_vm_pool_ok) {
		_vm_pool_ok = _vm_pool_init();
		if (!_vm_pool_ok) return cudaErrorMemoryAllocation;
	}
	guru_vm *vm = _vm_pool;
	U32 i;
	for (i=1; i<=MIN_VM_COUNT; i++, vm++) {
		if (vm->id == 0) {			// whether vm is unallocated
			vm->id = ses->id = i;	// found, assign vm to session
			vm->step = step;
			break;
		}
	}
	if (i>MIN_VM_COUNT) return cudaErrorMemoryAllocation;

	guru_parse_bytecode(vm, ses->in);
	if (ses->trace) {
		printf("  vm[%d]: %p\n", vm->id, (void *)vm);
		guru_show_irep(vm->irep);
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
	return cudaSuccess;
}

__HOST__ cudaError_t
guru_vm_run(guru_ses *ses)
{
    _vm_begin<<<MIN_VM_COUNT, 1>>>(_vm_pool);
	cudaDeviceSynchronize();

	do {	// TODO: flip session/vm centric view into app-server style main loop
		_vm_trace(ses->trace);

		_vm_exec<<<MIN_VM_COUNT, 1>>>(_vm_pool);
		cudaDeviceSynchronize();

		// add host hook here
		guru_console_flush(ses->out, ses->trace);	// dump output buffer
	} while (_vm_join());

    _vm_end<<<MIN_VM_COUNT, 1>>>(_vm_pool);
	cudaDeviceSynchronize();

	return cudaSuccess;
}

__HOST__ cudaError_t
guru_vm_release(guru_ses *ses)
{
	// TODO: release vm back to pool
	return cudaSuccess;
}

//========================================================================================
// the following code is for debugging purpose, turn off GURU_DEBUG for release
//========================================================================================
#ifdef GURU_DEBUG
static const char *_vtype[] = {
	"___","nil","f  ","t  ","num","flt","sym","cls",
	"","","","","","","","",
	"","","","","obj","prc","ary","str",
	"rng","hsh"
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
    "CLASS","","STOP","","","","","",
    "ABORT"
};

__HOST__ int
_find_irep(guru_irep *irep0, guru_irep *irep1, U8P idx)
{
	if (irep0==irep1) return 1;

	U8P  base = (U8P)irep0;
	U32P off  = (U32P)(base + irep0->list);		// child irep offset array
	for (U32 i=0; i<irep0->rlen; i++) {
		*idx += 1;
		if (_find_irep((guru_irep *)(base + off[i]), irep1, idx)) return 1;
	}
	return 0;		// not found
}

__HOST__ void
_show_decoder(guru_vm *vm)
{
	U16  pc    = vm->state->pc;
	U32  *iseq = (U32*)VM_ISEQ(vm);
	U16  opid  = (*(iseq + pc) >> 24) & 0x7f;		// in HOST mode, GET_OPCODE() is DEVICE code
	U8P  opc   = (U8P)_opcode[GET_OPCODE(opid)];

	U8 idx = 'a';
	if (!_find_irep(vm->irep, vm->state->irep, &idx)) idx='?';
	printf("%1d%c%-4d%-8s", vm->id, idx, pc, opc);

	U32 lvl=0;
	guru_state *st = vm->state;
	while (st->prev != NULL) {
		st = st->prev;
		lvl += 2 + st->argc;
	}

	mrbc_value *v = vm->regfile;				// scan
	U32 last = 0;
	for (U32 i=0; i<MAX_REGS_SIZE; i++, v++) {
		if (v->tt==GURU_TT_EMPTY) continue;
		last=i;
	}

	v = vm->regfile;							// rewind
	printf("[");
	for (U32 i=0; i<=last; i++, v++) {
		printf("%s",_vtype[v->tt]);
		if (v->tt >= GURU_TT_OBJECT) printf("%d", v->self->refc);
		else 						 printf(" ");
	    printf("%c", i==lvl ? '|' : ' ');
    }
	printf("]\n");
}

__HOST__ cudaError_t
_vm_trace(U32 level)
{
	if (level==0) return cudaSuccess;

	guru_vm *vm = _vm_pool;
	for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
		if (vm->id > 0 && vm->run && vm->step) {
			guru_state *st = vm->state;
			while (st->prev) {
				printf("%p <-- ", st);
				st = st->prev;
			}
			_show_decoder(vm);
		}
	}
	if (level>1) guru_dump_alloc_stat(level);

	return cudaSuccess;
}
#else
__HOST__ cudaError_t _vm_trace(U32 level) { return cudaSuccess; }
#endif 	// GURU_DEBUG



