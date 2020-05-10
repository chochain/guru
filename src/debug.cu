/*! @file
  @brief
  GURU instruction debugger (single-step)
    1. guru VM, host or cuda image, constructor and dispatcher
    2. dumpers for regfile and irep tree

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include "guru.h"
#include "util.h"
#include "mmu.h"
#include "symbol.h"

#include "state.h"
#include "ucode.h"

#include "debug.h"

cudaEvent_t _event_t0, _event_t1;

__GPU__ void
__id2str(GS sid, U8 *str)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	U8 *s = id2name(sid);
	STRCPY(str, s);
}

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
U8 *_outbuf = NULL;
U32 _debug  = 0;

__HOST__ void
_id2name(GS sid, U8 *str)
{
	__id2str<<<1,1>>>(sid, str);
	GPU_SYNC();
}

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
		if (_find_irep(IREP_REPS(irep0)+i, irep1, idx)) return 1;
	}
	return 0;		// not found
}

__HOST__ __INLINE__ RObj*
__gr_obj(GR *r)
{
	switch(r->gt) {
	case GT_STR:
	case GT_RANGE: 	return (RObj*)(guru_host_heap + r->off);
	default: 		return r->self;
	}
}

__HOST__ void
_show_regs(GR *r, U32 ri)
{
	for (U32 i=0; i<ri; i++, r++) {
		const char *t = _vtype[r->gt];
		U8 c = (i==0) ? '|' : ' ';
		RObj *o = __gr_obj(r);
		if (HAS_REF(r))			printf("%s%d%c", t, o->rc, c);
		else if (r->gt==GT_STR) printf("%s.%c", t, c);
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
	case OP_STRING: {
		guru_str *s0 = (guru_str*)(guru_host_heap + VM_STR(vm, ar.bx)->str);
		printf(" r%-2d ='%-17s", a, guru_host_heap + s0->raw);
		return;
	}
	case OP_LOADI:		printf(" r%-2d =%-18d",  a, ar.bx - MAX_sBx);				return;
	case OP_LOADL:		printf(" r%-2d =%-18g",  a, VM_VAR(vm, ar.bx)->f);			return;
	case OP_ADDI:
	case OP_SUBI:		printf(" r%-2d ~%-18d",  a, ar.c);							return;
	case OP_EXEC:		printf(" r%-2d +I%-17d", a, ar.bx+1);						return;
	case OP_GETUPVAR:   printf(" r%-2d =r^%-2d+%-13d",  a, up, ar.b);				return;
	case OP_SETUPVAR:	printf(" r^%-2d+%-2d =r%-13d",  up, ar.b, a);				return;
	case OP_ARRAY:
	case OP_HASH:
		if (ar.c<1)		printf(" r%-2d < %-17s", a, op==OP_ARRAY ? "[]" : "{}");
		else			printf(" r%-2d <r%-2d..r%-12d", a, ar.b, ar.b+ar.c-1);
		return;
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

	if (_outbuf==NULL) _outbuf = (U8*)cuda_malloc(OUTBUF_SIZE, 1);	// lazy alloc
	if (_find_op(_op_sym, op, SZ_SYM) >= 0) {
		GS sid = VM_SYM(vm, ar.bx);
		_id2name(sid, _outbuf);
		printf(" r%-2d :%-18s", a, _outbuf);			return;
	}
	if (_find_op(_op_exe, op, SZ_EXE) >= 0) {
		GS sid = VM_SYM(vm, ar.b);
		_id2name(sid, _outbuf);
		printf(" r%-2d #%-18s", a, _outbuf);			return;
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
_show_irep(guru_irep *irep, char level, char *n)
{
	U32 a = (U32A)irep;
	printf("\t%c irep[%c]=%08x: nreg=%d, nlocal=%d, pools=%d, syms=%d, reps=%d\n",
				level, *n, a, irep->nr, irep->nv, irep->p, irep->s, irep->r);
	// dump all children ireps
	guru_irep *r0 = IREP_REPS(irep);
	for (U32 i=0; i<irep->r; i++, r0++) {
		*n += (*n=='z') ? -57 : 1;		// a-z, A-Z
		_show_irep(r0, level+1, n);
	}
}

__HOST__ void
debug_init(U32 flag)
{
	_debug = flag;

	cudaEventCreate(&_event_t0);
	cudaEventCreate(&_event_t1);
	cudaEventRecord(_event_t0);
}

__HOST__ void
debug_mmu_stat()
{
	guru_mmu_check(_debug);
}

__HOST__ void
debug_vm_irep(guru_vm *vm)
{
	if (_debug<1 || !vm->state) return;

	printf("  vm[%d]:\n", vm->id);
	char c = 'a';
	_show_irep(vm->state->irep, '0'+vm->id, &c);
}

__HOST__ int
debug_disasm(guru_vm *vm)
{
	if (_debug<1 || !vm->state) return 0;

	for (guru_state *st=vm->state; st->prev; st=st->prev) printf("  ");

	_disasm(vm, _debug);

	return (_debug>1) ? guru_mmu_check(_debug) : 0;
}

__HOST__ void
debug_log(const char *msg)
{
	float ms;

	cudaEventRecord(_event_t1);
	cudaEventSynchronize(_event_t1);
	cudaEventElapsedTime(&ms, _event_t0, _event_t1);

	if (_debug) printf("%.3f> %s\n", ms, msg);
}

#else	// GURU_DEBUG
__HOST__ void debug_show_irep(guru_irep *irep, U32 ioff, char level, char *idx) {}
__HOST__ void debug_trace() {}
#endif 	// GURU_DEBUG

