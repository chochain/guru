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
_vm_exec(mrbc_vm *vm, mrbc_value *regfile)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;
	//
	// create a vm loop, potentially with different register files per vm
	while (guru_op(vm, regfile)==0) {
		// add service hook here
		// i.g. flush output buffer
	}
	__syncthreads();
}

//================================================================
/*!@brief
  release mrbc_irep holds memory
*/
__GURU__
void _mrbc_free_irep(mrbc_irep *irep)
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

int guru_vm_init(guru_ses *ses)
{
	mrbc_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);
	if (!vm) return -4;

	guru_parse_bytecode<<<1,1>>>(vm, ses->req);		// can also be done on host?
	cudaDeviceSynchronize();

#ifdef MRBC_DEBUG
	printf("guru bytecode loaded:\n");
	dump_irep(vm->irep);
#endif
	ses->vm = (uint8_t *)vm;
	return 0;
}

int guru_vm_run(guru_ses *ses)
{
	int sz;
	cudaDeviceGetLimit((size_t *)&sz, cudaLimitStackSize);
	printf("defaultStackSize %d =>", sz);

	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz*4);
	cudaDeviceGetLimit((size_t *)&sz, cudaLimitStackSize);
	printf("%d\n", sz);

	mrbc_vm *vm = (mrbc_vm *)ses->vm;
	_vm_begin<<<1,1>>>(vm);
	_vm_exec<<<1,1>>>(vm, vm->regfile);
	_vm_end<<<1,1>>>(vm);
	cudaDeviceSynchronize();

	return 0;
}

#ifdef MRBC_DEBUG
void dump_irep(mrbc_irep *irep)
{
	printf("\tnregs=%d, nlocals=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			irep->nreg, irep->nlv, irep->plen, irep->slen, irep->rlen, irep->ilen);
	// dump all children ireps
	for (int i=0; i<irep->rlen; i++) {
		dump_irep(irep->irep_list[i]);
	}
}
#endif

