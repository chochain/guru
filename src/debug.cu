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
#include "static.h"
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

	U8 *s = _RAW(sid);
	STRCPY(str, s);
}

//========================================================================================
// the following code is for debugging purpose, turn off GURU_DEBUG for release
//========================================================================================
#if GURU_DEBUG
static const char *_vtype[] = {
	"___","nil","f  ","t  ","num","flt","sym","sys",	// 0x0
	"cls","prc","obj","ary","str","rng","hsh","itr"		// 0x8
};

static const char *_errcode[] = {
	"", "ERROR", "Method Not Found", "CUDA"
};

static const char *_opcode[] = {
    "NOP ",	"MOVE",	"LOADL","LOADI","LOADSYM","LOADNIL","LOADSLF","LOADT",
    "LOADF","GETGBL","SETGBL","GETSPC","SETSPC","GETIV","SETIV","GETCV",
    "SETCV","GETCONS","SETCONS","","","GETUVAR","SETUVAR","JMP ",
    "JMPIF","JMPNOT","ONERR","RESCUE","POPERR","RAISE","EPUSH","EPOP",
    "SEND","SENDB","","CALL","","","ENTER","",
    "","RETURN","","BLKPUSH","ADD ","ADDI","SUB ","SUBI",
    "MUL ","DIV ","EQ  ","LT  ","LE  ","GT  ","GE  ","ARRAY",
    "ARYCAT","ARYPUSH","AREF","ASET","APOST","STRING","STRCAT","HASH",
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

#define h_STATE(off)  	((guru_state*)(guru_host_heap + (off)))
#define h_OBJ(off)		((guru_obj*)(guru_host_heap + (off)))
#define h_STR(off)		((guru_str*)(guru_host_heap + (off)))

#define ST_IREP(st)		((guru_irep*)(guru_host_heap + (st)->irep))
#define ST_REGS(st)		((GR*)(guru_host_heap + (st)->regs))
#define ST_ISEQ(st)	 	((U32*)IREP_ISEQ(ST_IREP(st)))
#define ST_STR(st,n)	(&IREP_POOL(ST_IREP(st))[(n)])
#define ST_VAR(st,n)	(&IREP_POOL(ST_IREP(st))[(n)])
#define ST_SYM(st,n)    ((IREP_POOL(ST_IREP(st))[ST_IREP(st)->p+(n)]).i)

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
	for (int i=0; i<n; i++) {
		if (op==lst[i]) return i;
	}
	return -1;
}

__HOST__ int
_match_irep(guru_irep *ix, guru_irep *ix0, U8 *n)
{
	if (ix==ix0) return 1;

	// search into children recursively
	guru_irep *ix1 = IREP_REPS(ix);
	for (int i=0; i<ix->r; i++, ix1++) {
		*n += (*n=='z') ? -57 : 1;		// a-z, A-Z
		if (_match_irep(ix1, ix0, n)) return 1;
	}
	return 0;		// not found
}

__HOST__ U8
_get_irep_id(guru_state *st)
{
	guru_state *sx = st;
	while (sx->prev) sx = h_STATE(sx->prev);			// find the root IREP

	guru_irep  *ix0 = ST_IREP(st);						// current IREP
	U8 idx = 'a';
	if (!_match_irep(ST_IREP(sx), ix0, &idx)) idx='?';	// recursively find the id

	return idx;
}

__HOST__ void
_show_regs(GR *r, U32 ri)
{
	for (int i=0; i<ri; i++, r++) {
		const char *t = _vtype[r->gt];
		U8  c  = (i==0) ? '|' : ' ';
		if (HAS_REF(r)) {
			U32 rc = h_OBJ(r->off)->rc;
			printf("%s%d%c", t, rc, c);
		}
		else if (r->gt==GT_STR) printf("%s.%c", t, c);
		else					printf("%s %c", t, c);
	}
}

__HOST__ void
_show_state_regs(guru_state *st, U32 lvl)
{
	if (st->prev) {
		guru_state *st1 = h_STATE(st->prev);
		_show_state_regs(st1, lvl+1);						// back tracing recursion
	}
	U32 n  = st->nv;										// depth of current stack frame
	GR  *r = ST_REGS(st);
	if (lvl==0) { 											// top most
		guru_irep *irep = ST_IREP(st);
		r = ST_REGS(st) + irep->nr;
		for (n=irep->nr; n>0 && r->gt==GT_EMPTY; n--, r--);
		n++;
	}
	r = ST_REGS(st);
	if (IS_LOOP(st)) _show_regs(r-3, n+1);					// alloc for looper
	else			 _show_regs(r,   n);
}

__HOST__ void
_show_decode(guru_state *st, GAR ar)
{
	U32  op = ar.op;
	U32  a  = ar.a;
	U32  in_lambda = st->prev && (h_STATE(st->prev)->flag & STATE_LAMBDA);
	U32  up = (ar.c+1)<<(in_lambda ? 0 : 1);

	switch (op) {
	case OP_MOVE: 		printf(" r%-2d =r%-17d", a, ar.b);							return;
	case OP_STRING: {
		guru_str *s0 = h_STR(ST_STR(st, ar.bx)->off);
		printf(" r%-2d ='%-17s", a, guru_host_heap + s0->raw);
		return;
	}
	case OP_LOADI:		printf(" r%-2d =%-18d",  a, ar.bx - MAX_sBx);				return;
	case OP_LOADL:		printf(" r%-2d =%-18g",  a, ST_VAR(st, ar.bx)->f);			return;
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
	case OP_AREF:		printf(" r%-2d =r%-2d+%-2d%12s", a, ar.b, ar.c, "");		return;
	case OP_ASET:		printf(" r%-2d+%-2d =r%-12d", ar.b, ar.c, a);				return;
	case OP_SCLASS:		printf(" r%-22d", ar.b);									return;
	case OP_ENTER:		printf(" @%-22d", 1 + st->argc - (ar.ax>>18)&0x1f);			return;
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
		printf(" r%-2d @%-18d", a, st->pc+(ar.bx - MAX_sBx));	return;
	}

	if (_outbuf==NULL) _outbuf = (U8*)cuda_malloc(OUTBUF_SIZE, 1);	// lazy alloc
	if (_find_op(_op_sym, op, SZ_SYM) >= 0) {
		GS sid = ST_SYM(st, ar.bx);
		_id2name(sid, _outbuf);
		printf(" r%-2d :%-18s", a, _outbuf);			return;
	}
	if (_find_op(_op_exe, op, SZ_EXE) >= 0) {
		GS sid = ST_SYM(st, ar.b);
		_id2name(sid, _outbuf);
		printf(" r%-2d #%-18s", a, _outbuf);			return;
	}
}

__HOST__ void
_disasm(guru_vm *vm, U32 level)
{
	guru_state *st = h_STATE(vm->state);

	U16  pc    = st->pc;							// program counter
	U32  *iseq = ST_ISEQ(st);
	U32  code  = bin2u32(*(iseq + pc));				// convert to big endian
	U32  rcode = ((code & 0x7f)<<25) | (code>>7);
	GAR  ar  { rcode };								// fit into RITE decoder template
	U8   op    = ar.op;
	U8   *opc  = (U8*)_opcode[GET_OP(op)];			// opcode text

	if (op >= OP_MAX) {
		printf("ERROR: st=%p, opcode %d out of range, bailing out...\n", st, op);
	}

	U8 idx = _get_irep_id(st);
	printf("%1d%c%-4d%-8s", vm->id, idx, pc, opc);

	_show_decode(st, ar);
	printf("[");
	_show_state_regs(st, 0);
	printf("]\n");
}

__HOST__ void
_show_irep(guru_irep *ix, int level, char *n)
{
	printf("\tirep[%c] %p", *n, ix);
	for (int i=0; i<=level; i++) {
		printf("  ");
	}
	printf("nreg=%d, nlocal=%d, pools=%d, syms=%d, reps=%d\n",
		ix->nr, ix->nv, ix->p, ix->s, ix->r);
	// dump all children ireps
	guru_irep *ix1 = IREP_REPS(ix);
	for (int i=0; i<ix->r; i++, ix1++) {
		*n += (*n=='z') ? -57 : 1;		// a-z, A-Z
		_show_irep(ix1, level+1, n);
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
	guru_state *st = h_STATE(vm->state);
	guru_irep  *ix = ST_IREP(st);
	_show_irep(ix, 0, &c);
}

__HOST__ void
debug_disasm(guru_vm *vm)
{
	if (_debug<1 || !vm->state) return;

	guru_state *st = h_STATE(vm->state);
	while (st->prev) {
		printf("  ");
		st = h_STATE(st->prev);
	}
	_disasm(vm, _debug);

	if (_debug>1) guru_mmu_check(_debug);
}

__HOST__ void
debug_error(guru_vm *vm)
{
	if (_debug<1 || !vm->err) return;

	printf("ERROR: %s\n", _errcode[vm->err]);
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

#else	// !GURU_DEBUG
__HOST__ void	debug_init(U32 flag) 		{}
__HOST__ void	debug_log(const char *msg) 	{}
__HOST__ void	debug_mmu_stat()			{}
__HOST__ void	debug_disasm(guru_vm *vm)	{}
__HOST__ void	debug_vm_irep(guru_vm *vm)	{}
#endif 	// GURU_DEBUG

