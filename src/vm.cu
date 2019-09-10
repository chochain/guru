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

	MEMSET(vm->regfile, 0, sizeof(vm->regfile));	// clean up registers

	vm->regfile[0].gt  = GT_CLASS;				// regfile[0] is self
    vm->regfile[0].cls = guru_class_object;		// root class

    guru_state *st = (guru_state *)guru_alloc(sizeof(guru_state));

    st->pc 	  = 0;								// starting IP
    st->klass = guru_class_object;				// target class
    st->regs  = vm->regfile;					// point to reg[0]
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

#if !GURU_DEBUG
	// clean up register file
	GV *p = vm->regfile;
	for (U32 i=0; i < MAX_REGS_SIZE; i++, p++) {
		ref_clr(p);
	}
    guru_memory_clr();
#endif
}

//================================================================
/*!@brief
  GURU Instruction Unit - Prefetcher (fetch bytecode and decode)

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GURU__ void
_vm_prefetch(guru_vm *vm)
{
   vm->bytecode = VM_BYTECODE(vm);

   vm->op  = vm->bytecode & 0x7f;          // opcode
   vm->opn = vm->bytecode >> 7;            // operands
   vm->ar  = (GAR *)&(vm)->opn;            // operands struct/union

   vm->state->pc++;							// advance program counter (ready for next fetch)
}

//================================================================
/*!@brief
  GURU Instruction dispatcher

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GPU__ void
_vm_exec(guru_vm *pool)
{
	guru_vm *vm = pool+blockIdx.x;
	if (vm->id==0 || !vm->run) return;			// not allocated yet, or completed

	// start up instruction and dispatcher unit
	U32 ret;
	do {
		// add before_fetch hooks here
		_vm_prefetch(vm);
		// add before_exec hooks here
		ret = guru_op(vm);
		// add after_exec hooks here
		ret |= vm->step;						// single stepping?
	} while (ret==0);
	__syncthreads();							// sync all cooperating threads (to share data)
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
	guru_vm *vm = _vm_pool = (guru_vm *)cuda_malloc(sizeof(guru_vm) * MIN_VM_COUNT, 1);
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
#if GURU_USE_CONSOLE
		guru_console_flush(ses->out, ses->trace);	// dump output buffer
#endif
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
#if GURU_DEBUG
static const char *_vtype[] = {
	"___","nil","f  ","t  ","num","flt","sym","cls",	// 0x0
	"","","","","","","","",							// 0x8
	"obj","prc","ary","str","rng","hsh"					// 0x10
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
	U32P off  = (U32P)U8PADD(base, irep0->reps);		// child irep offset array
	for (U32 i=0; i<irep0->c; i++) {
		*idx += 1;
		if (_find_irep((guru_irep *)(base + off[i]), irep1, idx)) return 1;
	}
	return 0;		// not found
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
		if (v->gt & GT_HAS_REF)	printf("%s%d%c", t, v->rc, c);
		else					printf("%s %c",  t, c);
    }
	printf("]");
}

#define bin2u32(x) ((x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24))

__HOST__ void
_show_decoder(guru_vm *vm)
{
	U16  pc    = vm->state->pc;						// program counter
	U32  *iseq = (U32*)VM_ISEQ(vm);
	U32  code  = bin2u32(*(iseq + pc));				// convert to big endian
	U16  op    = code & 0x7f;       				// in HOST mode, GET_OPCODE() is DEVICE code
	U8P  opc   = (U8P)_opcode[GET_OPCODE(op)];

	U8 idx = 'a';
	if (!_find_irep(vm->irep, vm->state->irep, &idx)) idx='?';
	printf("%1d%c%-4d%-8s", vm->id, idx, pc, opc);

	U32 lvl=0;
	guru_state *st = vm->state;
	while (st->prev != NULL) {
		st = st->prev;
		lvl += 2 + st->argc;
	}
	_show_regfile(vm, lvl);

	if (op==OP_SEND) {								// display function name
		U32 rb = (code >> 14) & 0x1ff;
		printf(" #%s", VM_SYM(vm, rb));
	}
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
				printf("  ");
				st = st->prev;
			}
			_show_decoder(vm);
			printf("\n");
		}
	}
	if (level>1) guru_dump_alloc_stat(level);

	return cudaSuccess;
}
#else
__HOST__ cudaError_t _vm_trace(U32 level) { return cudaSuccess; }
#endif 	// GURU_DEBUG

