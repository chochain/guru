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
_vm_begin(mrbc_vm *vm)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

    MEMSET((uint8_t *)vm->regfile, 0, sizeof(vm->regfile));	// clean up registers

    vm->regfile[0].tt  	= GURU_TT_CLASS;		// regfile[0] is self
    vm->regfile[0].cls 	= mrbc_class_object;	// root class

    mrbc_state *ci = (mrbc_state *)mrbc_alloc(sizeof(mrbc_state));

    ci->pc 	  = 0;								// starting IP
    ci->klass = mrbc_class_object;				// target class
    ci->reg   = vm->regfile;					// pointer to reg[0]
    ci->irep  = vm->irep;						// root of irep tree
    ci->argc  = 0;
    ci->prev  = NULL;

    vm->state = ci;
    vm->run   = 1;								// TODO: updated by scheduler
    vm->err   = 0;
}

//================================================================
/*!@brief
  VM finalizer.

  @param  vm  Pointer to VM
*/
__global__ void
_vm_end(mrbc_vm *vm)
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
_vm_exec(mrbc_vm *vm, int step)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;		// TODO: multi-threading

	while (guru_op(vm)==0 && step==0) {
		// add cuda hook here
	}
}

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
        _mrbc_free_irep(irep->ilist[i]);
    }
    if (irep->rlen) mrbc_free(irep->ilist);

    mrbc_free(irep);
}

__host__ cudaError_t
guru_vm_init(guru_ses *ses)
{
	mrbc_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);
	if (!vm) return cudaErrorMemoryAllocation;

	guru_parse_bytecode<<<1,1>>>(vm, ses->req);		// can also be done on host?
	cudaDeviceSynchronize();

	ses->vm = (uint8_t *)vm;

	if (ses->debug > 0)	guru_dump_irep(vm->irep);

	return cudaSuccess;
}

__host__ cudaError_t
guru_vm_run(guru_ses *ses)
{
    mrbc_vm *vm = (mrbc_vm *)ses->vm;
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
__host__ void
guru_dump_irep(mrbc_irep *irep)
{
	printf("\tnregs=%d, nlocals=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);

	// dump all children ireps
	for (int i=0; i<irep->rlen; i++) {
		guru_dump_irep(irep->ilist[i]);
	}
}

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

//(_bin_to_uint32((vm)->irep->iseq + (vm)->pc * 4))
//		((i) & 0x7f)

__host__ void
guru_dump_regfile(mrbc_vm *vm, int debug)
{
	if (debug==0) return;

	uint16_t    opid    = (*(GET_IREP(vm)->iseq + vm->state->pc) >> 24) & 0x7f;
	const char 	*opcode = _opcode[GET_OPCODE(opid)];
	mrbc_value 	*v 	 	= vm->regfile;

	int last=0;
	for(int i=0; i<MAX_REGS_SIZE; i++, v++) {
		if (v->tt==GURU_TT_EMPTY) continue;
		last=i;
	}
	v = vm->regfile;

	if (debug==1) {
		int s[8];
		guru_get_alloc_stat(s);
		printf("%-4d%-8s%-3d[ ", vm->state->pc, opcode, s[3]);
	}
	else if (debug==2) {
		guru_dump_alloc_stat();
		printf("%-4d%-8s[ ", vm->state->pc, opcode);
	}
	for (int i=0; i<=last; i++, v++) {
		printf("%2d.%s", i, _vtype[v->tt]);
	    if (v->tt >= GURU_TT_OBJECT) printf("_%d", v->self->refc);
	    printf(" ");
    }
	printf("]\n");
}
#endif



