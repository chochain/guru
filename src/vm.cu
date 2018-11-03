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

    vm->regfile[0].tt  	= MRBC_TT_CLASS;		// regfile[0] is self
    vm->regfile[0].cls 	= mrbc_class_object;	// root class

    vm->calltop = NULL;							// no call

    vm->pc 		= 0;							// starting IP
    vm->klass 	= mrbc_class_object;			// target class
    vm->reg 	= vm->regfile;					// pointer to reg[0]
    vm->pc_irep = vm->irep;						// root of irep tree
    vm->run   	= 1;							// TODO: updated by scheduler
    vm->err     = 0;
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

#ifndef MRBC_DEBUG
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
	if (threadIdx.x!=0 || blockIdx.x!=0) return;		// single thread for now

	while (guru_op(vm)==0 && step==0);
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
        _mrbc_free_irep(irep->irep_list[i]);
    }
    if (irep->rlen) mrbc_free(irep->irep_list);

    mrbc_free(irep);
}
__host__ int
guru_vm_init(guru_ses *ses)
{
	mrbc_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);
	if (!vm) return -4;

	guru_parse_bytecode<<<1,1>>>(vm, ses->req);		// can also be done on host?
	cudaDeviceSynchronize();

#ifdef MRBC_DEBUG
	printf("guru bytecode loaded:\n");
	guru_dump_irep(vm->irep);
#endif
	ses->vm = (uint8_t *)vm;
	return 0;
}

__host__ int
guru_vm_run(guru_ses *ses)
{
	int sz;
	cudaDeviceGetLimit((size_t *)&sz, cudaLimitStackSize);
	printf("defaultStackSize %d =>", sz);

	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz*4);
	cudaDeviceGetLimit((size_t *)&sz, cudaLimitStackSize);
	printf("%d\n", sz);

    guru_console_init<<<1,1>>>(ses->res, MAX_BUFFER_SIZE);	// initialize output buffer

    mrbc_vm *vm = (mrbc_vm *)ses->vm;
	_vm_begin<<<1,1>>>(vm);
	cudaDeviceSynchronize();

	// enter the vm loop, potentially with different register files per thread
	do {
		_vm_exec<<<1,1>>>(vm, 1);							// 1: single-step
		cudaDeviceSynchronize();

		// add service hook here
		guru_console_flush(ses->res);						// dump output buffer
		guru_dump_regfile(vm);
	} while (vm->run && vm->err==0);
	cudaError rst = cudaGetLastError();
    if (cudaSuccess != rst) {
    	printf("\nERR> %s\n", cudaGetErrorString(rst));
    }

	_vm_end<<<1,1>>>(vm);
	cudaDeviceSynchronize();

	return 0;
}

#ifdef MRBC_DEBUG
__host__ void
guru_dump_irep(mrbc_irep *irep)
{
	printf("\tnregs=%d, nlocals=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);
	// dump all children ireps
	for (int i=0; i<irep->rlen; i++) {
		guru_dump_irep(irep->irep_list[i]);
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
    "LOADF","GETG","SETG","","","GETI","SETI","",
    "","GETC","SETC","","","GETU","SETU","JMP ",
    "JMPIF","JMPNOT","","","","","","",
    "SEND","SENDB","","CALL","","","ENTER","",
    "","RETURN","","BLKPUSH","ADD ","ADDI","SUB ","SUBI",
    "MUL ","DIV ","EQ  ","LT  ","LE  ","GT  ","GE  ","ARRAY",
    "","","","","","STRING","STRCAT","HASH",
    "LAMBDA","RANGE","","CLASS","","EXEC","METHOD","",
    "CLASS","","STOP","","","","","",
    "ABORT"
};

__host__ void
guru_dump_regfile(mrbc_vm *vm)
{
	mrbc_value *v = vm->regfile;

	int last=0;
	for(int i=0; i<MAX_REGS_SIZE; i++, v++) {
		if (v->tt==MRBC_TT_EMPTY) continue;
		last=i;
	}
	v = vm->regfile;

	printf("%s\t[ ", _opcode[vm->opcode]);
	for (int i=0; i<last; i++, v++) {
		printf("%2d.%s", i, _vtype[v->tt]);
	    if (v->tt >= MRBC_TT_OBJECT) printf("_%d", v->self->refc);
	    printf(" ");
    }
	printf("]\n");
}
#endif



