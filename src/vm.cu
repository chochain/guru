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
#include "console.h"
#include "opcode.h"
#include "load.h"
#include "vm.h"

int     _vm_pool_ok = 0;
guru_vm **_vm_pool;

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__GPU__ void
_vm_begin(guru_vm *pool[])
{
	guru_vm *vm = pool[blockIdx.x];

	if (threadIdx.x!=0 || vm->free) return;

    MEMSET((uint8_t *)vm->regfile, 0, sizeof(vm->regfile));	// clean up registers

    vm->regfile[0].tt  	= GURU_TT_CLASS;		// regfile[0] is self
    vm->regfile[0].cls 	= mrbc_class_object;	// root class

    guru_state *st = (guru_state *)mrbc_alloc(sizeof(guru_state));

    st->pc 	  = 0;								// starting IP
    st->klass = mrbc_class_object;				// target class
    st->reg   = vm->regfile;					// pointer to reg[0]
    st->irep  = vm->irep;						// root of irep tree
    st->argc  = 0;
    st->prev  = NULL;

    vm->state = st;
    vm->run   = 1;								// TODO: updated by scheduler
    vm->done  = 0;
    vm->err   = 0;
}

//================================================================
/*!@brief
  VM finalizer.

  @param  vm  Pointer to VM
*/
__GPU__ void
_vm_end(guru_vm *pool[])
{
	guru_vm *vm = pool[blockIdx.x];

	if (threadIdx.x!=0 || vm->free) return;

#ifndef GURU_DEBUG
	// clean up register file?						// CC: moved from mrbc_op 20181102
	mrbc_value *p = vm->regfile;
	for(int i = 0; i < MAX_REGS_SIZE; i++, p++) {
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
_vm_exec(guru_vm *pool[], int step)
{
	guru_vm *vm = pool[blockIdx.x];

	if (vm->free) return;

	while (guru_op(vm)==0 && step==0) {		// multi-threading in guru_op
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
    for(int i = 0; i < irep->plen; i++) {
        mrbc_free(irep->pool[i]);
    }
    if (irep->plen) mrbc_free(irep->pool);

    // release all child ireps.
    for(int i = 0; i < irep->rlen; i++) {
        _mrbc_free_irep(irep->list[i]);
    }
    if (irep->rlen) mrbc_free(irep->list);
    mrbc_free(irep);
}
#endif

__HOST__ cudaError_t
_vm_pool_init(int debug)
{
	_vm_pool    = (guru_vm **)guru_malloc(sizeof(guru_vm *) * MIN_VM_COUNT, 1);
	guru_vm *vm = (guru_vm *)guru_malloc(sizeof(guru_vm) * MIN_VM_COUNT, 1);
	if (!_vm_pool || !vm) return cudaErrorMemoryAllocation;

	for (int i=0; i<MIN_VM_COUNT; i++, vm++) {
		vm->free = 1;
		_vm_pool[i] = vm;
	}
	if (debug) printf("\tnumber of VMs allocated: %d\n", MIN_VM_COUNT);

	return cudaSuccess;
}

__HOST__ cudaError_t
guru_vm_init(guru_ses *ses)
{
#if GURU_HOST_IMAGE
	if (!_vm_pool_ok) {
		if (_vm_pool_init(ses->debug)!=cudaSuccess) return cudaErrorMemoryAllocation;
		_vm_pool_ok = 1;
	}

	ses->vm_id = 0;						// assign vm to session
	guru_vm *vm = _vm_pool[0];			// allocate from the pool
	vm->free   = 0;

	guru_parse_bytecode(vm, ses->req);
#else
	mrbc_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);
	if (!vm) return cudaErrorMemoryAllocation;

	guru_parse_bytecode<<<1,1>>>(vm, ses->req);
	cudaDeviceSynchronize();
#endif

	if (ses->debug > 0)	guru_dump_irep(vm->irep);

	return cudaSuccess;
}

__HOST__ cudaError_t
guru_vm_run(guru_ses *ses)
{
	guru_vm *vm = _vm_pool[ses->vm_id];

    _vm_begin<<<MIN_VM_COUNT, 1>>>(_vm_pool);
	cudaDeviceSynchronize();

	do {	// enter the vm loop, potentially with different register files per thread
		guru_dump_regfile(_vm_pool, ses->debug);			// for debugging

		_vm_exec<<<MIN_VM_COUNT, 1>>>(_vm_pool, 1);			// 1: single-step
		cudaDeviceSynchronize();

		// add host hook here
		guru_console_flush(ses->res);						// dump output buffer
	} while (!vm->done && !vm->err);

    _vm_end<<<MIN_VM_COUNT, 1>>>(_vm_pool);
	cudaDeviceSynchronize();

	return cudaSuccess;
}

__HOST__ cudaError_t
guru_vm_release(guru_ses *ses)
{
	// release vm back to pool
	return cudaSuccess;
}

#ifdef GURU_DEBUG
static const char *_vtype[] = {
	"___","nil","f  ","t  ","num","flt","sym","cls",
	"","","","","","","","",
	"","","","","obj","prc","ary","str",
	"rng","hsh"
};

static const char *_opcode[] = {
    "NOP ","MOVE","LOADL","LOADI","LOADSYM","LOADNIL","LOADSLF","LOADT",
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

#if GURU_HOST_IMAGE
__HOST__ void
guru_dump_irep(guru_irep *irep)
{
	printf("\tsize=%d, nreg=%d, nlocal=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			irep->size, irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);

	// dump all children ireps
	uint8_t  *base = (uint8_t *)irep;
	uint32_t *off  = (uint32_t *)(base + irep->list);
	for (int i=0; i<irep->rlen; i++, off++) {
		guru_dump_irep((guru_irep *)(base + *off));
	}
}

__HOST__ void
guru_dump_regfile(guru_vm *pool[], int debug_level)
{
	if (debug_level==0) return;

	guru_vm *vm = pool[0];

	uint16_t    pc      = vm->state->pc;
	uint32_t    *iseq   = VM_ISEQ(vm);
	uint16_t    opid    = (*(iseq + pc) >> 24) & 0x7f;
	const char 	*opcode = _opcode[GET_OPCODE(opid)];
	mrbc_value 	*v 	 	= vm->regfile;

	int last=0;
	for(int i=0; i<MAX_REGS_SIZE; i++, v++) {
		if (v->tt==GURU_TT_EMPTY) continue;
		last=i;
	}
	int lvl=0;
	guru_state *s = vm->state;
	while (s->prev != NULL) {
		s = s->prev;
		lvl++;
	}

	v = vm->regfile;	// rewind
	if (debug_level==1) {
		int s[8];
		guru_malloc_stat(s);
		printf("%c%-4d%-8s%-3d[ ", 'a'+lvl, pc, opcode, s[3]);
	}
	else if (debug_level==2) {
		guru_dump_alloc_stat();
		printf("%c%-4d%-8s[ ", 'a'+lvl, pc, opcode);
	}
	for (int i=0; i<=last; i++, v++) {
		printf("%2d.%s", i, _vtype[v->tt]);
	    if (v->tt >= GURU_TT_OBJECT) printf("_%d", v->self->refc);
	    printf(" ");
    }
	printf("]\n");
}
#else // !GURU_HOST_IMAGE
__HOST__ void
guru_dump_irep(mrbc_irep *irep)
{
	printf("\tnregs=%d, nlocals=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);

	// dump all children ireps
	for (int i=0; i<irep->rlen; i++) {
		guru_dump_irep(irep->list[i]);
	}
}

__HOST__ void
guru_dump_regfile(mrbc_vm *vm, int debug_level)
{
	if (debug_level==0) return;

	uint16_t    pc      = vm->state->pc;
	uint32_t    *iseq   = vm->state->irep->iseq;
	uint16_t    opid    = (*(iseq + pc) >> 24) & 0x7f;
	const char 	*opcode = _opcode[GET_OPCODE(opid)];
	mrbc_value 	*v 	 	= vm->regfile;

	int last=0;
	for(int i=0; i<MAX_REGS_SIZE; i++, v++) {
		if (v->tt==GURU_TT_EMPTY) continue;
		last=i;
	}
	int lvl=0;
	guru_state *s = vm->state;
	while (s->prev != NULL) {
		s = s->prev;
		lvl++;
	}

	v = vm->regfile;	// rewind
	if (debug_level==1) {
		int s[8];
		guru_malloc_stat(s);
		printf("%c%-4d%-8s%-3d[ ", 'a'+lvl, pc, opcode, s[3]);
	}
	else if (debug_level==2) {
		guru_dump_alloc_stat();
		printf("%c%-4d%-8s[ ", 'a'+lvl, pc, opcode);
	}
	for (int i=0; i<=last; i++, v++) {
		printf("%2d.%s", i, _vtype[v->tt]);
	    if (v->tt >= GURU_TT_OBJECT) printf("_%d", v->self->refc);
	    printf(" ");
    }
	printf("]\n");
}
#endif	// GURU_HOST_IMAGE
#endif 	// GURU_DEBUG



