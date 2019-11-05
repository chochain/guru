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

#include "mmu.h"
#include "static.h"
#include "state.h"
#include "symbol.h"
#include "vmx.h"
#include "vm.h"

#include "ucode.h"
#include "c_string.h"

guru_vm *_vm_pool;

pthread_mutex_t 	_mutex_pool;
#define _LOCK		(pthread_mutex_lock(&_mutex_pool))
#define _UNLOCK		(cudaDeviceSynchronize(), pthread_mutex_unlock(&_mutex_pool))

__HOST__ void _show_irep(guru_irep *irep, char level, char *idx);		// forward declaration
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

	GV *r = vm->regfile;
	for (U32 i=0; i<MAX_REGFILE_SIZE; i++, r++) {		// wipe register
		r->gt  = (i>0) ? GT_EMPTY : GT_CLASS;			// reg[0] is "self"
		r->cls = (i>0) ? NULL     : guru_class_object;
		r->acl = 0;
	}
    vm->state = NULL;
    vm->run   = VM_STATUS_READY;
    vm->depth = vm->err = 0;

    vm_state_push(vm, irep, 0, vm->regfile, 0);
}

__GURU__ void
_free(guru_vm *vm)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;		// single threaded
	if (vm->run!=VM_STATUS_STOP) 		 return;

	while (vm->state) {									// pop off call stack
		vm_state_pop(vm, vm->state->regs[1]);
	}
	vm->run = VM_STATUS_FREE;							// release the vm
}

//================================================================
// Transcode Pooled objects and Symbol table recursively
// from source memory pointers to GV[] (i.e. regfile)
//
__GURU__ void
__transcode(guru_irep *irep)
{
	GV *p = irep->pool;
	for (U32 i=0; i < irep->s; i++, p++) {		// symbol table
		*p = guru_sym_new(p->sym);
		p->acl |= ACL_READ_ONLY;
	}
	for (U32 i=0; i < irep->p; i++, p++) {		// pooled objects
		if (p->gt==GT_STR) {
			*p = guru_str_new(p->sym);
		}
		p->acl |= ACL_READ_ONLY;
	}
	// use tail recursion (i.e. no call stack, so compiler optimization becomes possible)
	for (U32 i=0; i < irep->r; i++) {
		__transcode(irep->reps[i]);
	}
}

__GPU__ void
_fetch(guru_vm *pool, guru_irep *irep, int *vid)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;	// single threaded

	guru_vm *vm = pool;
	int idx;

	for (idx=0; idx<MIN_VM_COUNT; idx++, vm++) {
		if (vm->run==VM_STATUS_FREE) {
			vm->run = VM_STATUS_READY;		// reserve the VM
			break;
		}
	}
	if (idx<MIN_VM_COUNT) {
		__transcode(irep);					// recursively transcode Pooled objects and Symbol table
		_ready(vm, irep);
	}
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
#endif // GURU_USE_CONSOLE
	} while (_join());

	if (trace) {
		printf("guru_session completed\n");
		show_mmu_stat(trace);
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
			_show_irep(irep, 'A', &c);
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
	"___","nil","f  ","t  ","num","flt","sym","",		// 0x0
	"cls","prc","","","","","","",						// 0x8
	"obj","ary","str","rng","hsh","itr"					// 0x10
};

static const char *_opcode[] = {
    "NOP ",	"MOVE",	"LOADL","LOADI","LOADSYM","LOADNIL","LOADSLF","LOADT",
    "LOADF","GETGBL","SETGBL","GETSPC","SETSPC","GETIV","SETIV","GETCV",
    "SETCV","GETCONS","SETCONS","","","GETUVAR","SETUVAR","JMP ",
    "JMPIF","JMPNOT","ONERR","RESCUE","POPERR","RAISE","EPUSH","EPOP",
    "SEND","SENDB","","CALL","","","ENTER","",
    "","RETURN","","BLKPUSH","ADD ","ADDI","SUB ","SUBI",
    "MUL ","DIV ","EQ  ","LT  ","LE  ","GT  ","GE  ","ARRAY",
    "","","","","","STRING","STRCAT","HASH",
    "LAMBDA","RANGE","","CLASS","MODULE","EXEC","METHOD","SCLASS",
    "TCLASS","","STOP","","","","","",
    "ABORT"
};

static const int _op_bru[] = {
	OP_LOADNIL, OP_LOADSELF, OP_LOADT, OP_LOADF,
    OP_POPERR, OP_RAISE,
    OP_CALL, OP_RETURN, OP_BLKPUSH,
    OP_LAMBDA, OP_TCLASS, OP_STOP
};
#define SZ_BRU 	(sizeof(_op_bru)/sizeof(int))

static const int _op_alu[] = {
	OP_ADD, OP_SUB, OP_MUL, OP_DIV,
	OP_EQ, OP_LT, OP_LE, OP_GT, OP_GE,
	OP_STRCAT, OP_RANGE
};
#define SZ_ALU	(sizeof(_op_alu)/sizeof(int))

static const int _op_jmp[] = {
	OP_JMP, OP_JMPIF, OP_JMPNOT, OP_ONERR
};
#define SZ_JMP	(sizeof(_op_jmp)/sizeof(int))

static const int _op_sym[] = {
	OP_LOADSYM,
	OP_GETGLOBAL, OP_SETGLOBAL, OP_GETCONST, OP_SETCONST,
	OP_GETIV, OP_SETIV, OP_SETCV, OP_GETCV
};
#define SZ_SYM	(sizeof(_op_sym)/sizeof(int))

static const int _op_exe[] = {
	OP_SEND, OP_SENDB, OP_CLASS, OP_MODULE, OP_METHOD
};
#define SZ_EXE	(sizeof(_op_exe)/sizeof(int))

#define OUTBUF_SIZE	256
U8 *outbuf = NULL;

#define bin2u32(x) ((x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24))
__HOST__ int
_find_op(const int *lst, int op, int n)
{
	for (U32 i=0; i<n; i++) {
		if (op==lst[i]) return i;
	}
	return -1;
}

__HOST__ int
_find_irep(guru_irep *irep0, guru_irep *irep1, U8 *idx)
{
	if (irep0==irep1) return 1;
	for (U32 i=0; i<irep0->r; i++) {
		*idx += 1;
		if (_find_irep(irep0->reps[i], irep1, idx)) return 1;
	}
	return 0;		// not found
}

__HOST__ void
_show_irep(guru_irep *irep, char level, char *n)
{
	U32 a = (U32A)irep;
	printf("\t%c irep[%c]=%08x: size=0x%x, nreg=%d, nlocal=%d, pools=%d, syms=%d, reps=%d, ilen=%d\n",
			level, *n, a,
			irep->size, irep->nr, irep->nv, irep->p, irep->s, irep->r, irep->i);

	// dump all children ireps
	for (U32 i=0; i<irep->r; i++) {
		*n += (*n=='z') ? -57 : 1;		// a-z, A-Z
		_show_irep(irep->reps[i], level+1, n);
	}
}

__HOST__ void
_show_regs(GV *v, U32 vi)
{
	for (U32 i=0; i<vi; i++, v++) {
		const char *t = _vtype[v->gt];
		U8 c = (i==0) ? '|' : ' ';
		if (IS_READ_ONLY(v)) 	printf("%s.%c",  t, c);
		else if (HAS_REF(v))	printf("%s%d%c", t, v->self->rc, c);
		else					printf("%s %c", t, c);
	}
}

__HOST__ void
_show_state_regs(guru_state *st, U32 lvl)
{
	if (st->prev) _show_state_regs(st->prev, lvl+1);		// back tracing recursion
	U32 n = st->nv;											// depth of current stack frame
	if (lvl==0) {											// top most
		for (n=st->irep->nr; n>0 && st->regs[n].gt==GT_EMPTY; n--);
		n++;
	}
	_show_regs((st->flag & STATE_LOOP) ? st->regs-2 : st->regs, n);
}

__HOST__ void
_show_decode(guru_vm *vm, U32 code)
{
	U16  op = code & 0x7f;
	U32  n  = code>>7;
	GAR  ar = *((GAR*)&n);
	U32  a  = ar.a;
	U32  up = (ar.c+1)<<(IN_LAMBDA(vm->state) ? 0 : 1);

	switch (op) {
	case OP_MOVE: 		printf(" r%-2d =r%-17d", a, ar.b);							return;
	case OP_STRING:		printf(" r%-2d ='%-17s", a, VM_STR(vm, ar.bx).str->raw);	return;
	case OP_LOADI:		printf(" r%-2d =%-18d",  a, ar.bx - MAX_sBx);				return;
	case OP_LOADL:		printf(" r%-2d =%-18g",  a, VM_VAR(vm, ar.bx).f);			return;
	case OP_ADDI:
	case OP_SUBI:		printf(" r%-2d ~%-18d",  a, ar.c);							return;
	case OP_EXEC:		printf(" r%-2d +I%-17d", a, ar.bx+1);						return;
	case OP_GETUPVAR:   printf(" r%-2d =r^%-2d+%-13d",  a, up, ar.b);				return;
	case OP_SETUPVAR:	printf(" r^%-2d+%-2d =r%-13d",  up, ar.b, a);				return;
	case OP_ARRAY:
	case OP_HASH: {
		if (ar.c<1)		printf(" r%-2d < %-17s", a, op==OP_ARRAY ? "[]" : "{}");
		else			printf(" r%-2d <r%-2d..r%-12d", a, ar.b, ar.b+ar.c-1);
		return;
	} break;
	case OP_SCLASS:		printf(" r%-22d", ar.b);									return;
	case OP_ENTER:		printf(" @%-22d", 1 + vm->state->argc - (n>>18));			return;
	case OP_RESCUE:
		printf(" r%-2d =%1d?r%-15d", (ar.c ? a+1 : a), ar.c, (ar.c ? a : a+1));		return;
	}
	if (_find_op(_op_bru, op, SZ_BRU) >= 0) {
		printf(" r%-22d", a);							return;
	}
	if (_find_op(_op_alu, op, SZ_ALU) >= 0) {
		printf(" r%-2d ,r%-17d", a, a+1);				return;
	}
	if (_find_op(_op_jmp, op, SZ_JMP) >= 0) {
		printf(" r%-2d @%-18d", a, vm->state->pc+(ar.bx - MAX_sBx));	return;
	}

	if (outbuf==NULL) outbuf = (U8*)cuda_malloc(OUTBUF_SIZE, 1);	// lazy alloc
	if (_find_op(_op_sym, op, SZ_SYM) >= 0) {
		GS  sid = VM_SYM(vm, ar.bx);
		id2name_host(sid, outbuf);
		printf(" r%-2d :%-18s", a, outbuf);				return;
	}
	if (_find_op(_op_exe, op, SZ_EXE) >= 0) {
		GS  sid = VM_SYM(vm, ar.b);
		id2name_host(sid, outbuf);
		printf(" r%-2d #%-18s", a, outbuf);				return;
	}
}

__HOST__ void
_disasm(guru_vm *vm, U32 level)
{
	U16  pc    = vm->state->pc;						// program counter
	U32  *iseq = VM_ISEQ(vm);
	U32  code  = bin2u32(*(iseq + pc));				// convert to big endian
	U16  op    = code & 0x7f;       				// in HOST mode, GET_OPCODE() is DEVICE code
	U8   *opc  = (U8*)_opcode[GET_OP(op)];

	guru_state *st    = vm->state;
	guru_irep  *irep1 = st->irep;

	U8 idx = 'a';
	if (st->prev) {
		if (!_find_irep(st->prev->irep, irep1, &idx)) idx='?';
	}
	printf("%1d%c%-4d%-8s", vm->id, idx, pc, opc);

	_show_decode(vm, code);
	printf("[");
	_show_state_regs(vm->state, 0);
	printf("]\n");
}

__HOST__ void
_trace(U32 level)
{
	if (level==0) return;

	guru_vm *vm = _vm_pool;
	for (U32 i=0; i<MIN_VM_COUNT; i++, vm++) {
		if (vm->run==VM_STATUS_RUN && vm->step) {
			for (guru_state *st=vm->state; st->prev; st=st->prev) {
				printf("  ");
			}
			_disasm(vm, level);
		}
	}
	if (level>1) show_mmu_stat(level);
}

#else
__HOST__ void _show_irep(guru_irep *irep, U32 ioff, char level, char *idx) {}
__HOST__ void _trace(U32 level) {}
#endif 	// GURU_DEBUG

