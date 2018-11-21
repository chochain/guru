/*! @file
  @brief
  Guru bytecode executor.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  Fetch mruby VM bytecodes, decode and execute.

  </pre>
*/
#include <stdio.h>

#include "alloc.h"
#include "static.h"
#include "console.h"
#include "opcode.h"
#include "load.h"
#include "vm.h"

//================================================================
/*!@brief
  VM initializer.

  @param  vm  Pointer to VM
*/
__global__ void
_vm_begin(guru_vm *vm)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

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
    vm->err   = 0;
}

//================================================================
/*!@brief
  VM finalizer.

  @param  vm  Pointer to VM
*/
__global__ void
_vm_end(guru_vm *vm)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

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
__global__ void
_vm_exec(guru_vm *vm, int step)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;		// TODO: multi-threading

	while (guru_op(vm)==0 && step==0) {
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

__host__ cudaError_t
guru_vm_init(guru_ses *ses)
{
#if GURU_HOST_IMAGE
	guru_vm *vm = (guru_vm *)guru_malloc(sizeof(guru_vm), 1);
	if (!vm) return cudaErrorMemoryAllocation;

	guru_parse_bytecode(vm, ses->req);
	ses->vm = (uint8_t *)vm;

	if (ses->debug > 0)	guru_dump_irep(vm->irep);
#else
	guru_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);
	if (!vm) return cudaErrorMemoryAllocation;

	guru_parse_bytecode<<<1,1>>>(vm, ses->req);
	cudaDeviceSynchronize();

	ses->vm = (uint8_t *)vm;
	if (ses->debug > 0)	guru_dump_irep1(vm->irep);
#endif
	return cudaSuccess;
}

__host__ cudaError_t
guru_vm_run(guru_ses *ses)
{
    guru_vm *vm = (guru_vm *)ses->vm;
	_vm_begin<<<1,1>>>(vm);
	cudaDeviceSynchronize();

	do {	// enter the vm loop, potentially with different register files per thread
		guru_dump_regfile(vm, ses->debug);					// for debugging

		_vm_exec<<<1,1>>>(vm, 1);							// 1: single-step
		cudaDeviceSynchronize();

		// add host hook here
		guru_console_flush(ses->res);						// dump output buffer
	} while (vm->run && vm->err==0);

    _vm_end<<<1,1>>>(vm);
	cudaDeviceSynchronize();

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
__host__ void
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

__host__ void
guru_dump_regfile(guru_vm *vm, int debug_level)
{
	if (debug_level==0) return;

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
		guru_get_alloc_stat(s);
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
__host__ void
guru_dump_irep(mrbc_irep *irep)
{
	printf("\tnregs=%d, nlocals=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);

	// dump all children ireps
	for (int i=0; i<irep->rlen; i++) {
		guru_dump_irep(irep->list[i]);
	}
}

__host__ void
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
		guru_get_alloc_stat(s);
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



