/*! @file
  @brief
  GURU microcode unit

  <pre>
  Copyright (C) 2019 GreenII

  1. a list of opcode (microcode) executor, and
  2. the core opcode dispatcher
  </pre>
*/
#include <assert.h>

#include "alloc.h"
#include "store.h"
#include "static.h"
#include "symbol.h"
#include "global.h"
#include "opcode.h"
#include "object.h"
#include "class.h"
#include "state.h"

#if GURU_USE_STRING
#include "c_string.h"
#endif
#if GURU_USE_ARRAY
#include "c_range.h"
#include "c_array.h"
#include "c_hash.h"
#endif

#include "puts.h"

__GURU__ U32 _mutex_op;
//
// becareful with the following macros, because they release regs[ra] first
// so, make sure value is kept before the release
//
#define _ARG(r)         ((vm->ar->r))
#define _R(r)			((vm)->state->regs[_ARG(r)])
#define _RA(v)      	(ref_dec(&regs[ra]), regs[ra]=(v))
#define _RA_X(r)    	(ref_dec(&regs[ra]), regs[ra]=*(r), ref_inc(r))
#define _RA_T(t, e) 	(ref_dec(&regs[ra]), regs[ra].gt=(t), regs[ra].e)
#define _RA_T2(t,e)     (_R(a).gt=(t), _R(a).e)
#define SKIP(x)	        { guru_na(x); return; }
#define QUIT(x)			{ vm->quit=1; guru_na(x); return; }
//================================================================
/*!@brief
  Execute OP_NOP

  No operation

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_nop(guru_vm *vm)
{
}

//================================================================
/*!@brief
  Execute OP_MOVE

  R(A) := R(B)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_move(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
	U32 ra = GETARG_A(code);
	U32 rb = GETARG_B(code);

    _RA_X(&regs[rb]); 	                	// [ra] <= [rb]
}

//================================================================
/*!@brief
  Execute OP_LOADL

  R(A) := Pool(Bx)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_loadl(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
	U32 ra = GETARG_A(code);
    U32 rb = GETARG_Bx(code);
    U32P p = VM_VAR(vm, rb);
    guru_obj obj;

    if (*p & 1) {
    	obj.gt = GT_FLOAT;
    	obj.f  = *(GF *)p;
    }
    else {
    	obj.gt = GT_INT;
    	obj.i  = *p>>1;
    }
    _RA(obj);
}

//================================================================
/*!@brief
  Execute OP_LOADI

  R(A) := sBx

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_loadi(guru_vm *vm)
{
    GI rb = _ARG(bx) - MAXARG_sBx;		// sBx

    _RA_T2(GT_INT, i=rb);
}


//================================================================
/*!@brief
  Execute OP_LOADSYM

  R(A) := Syms(Bx)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_loadsym(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS  sid = name2id(VM_SYM(vm, rb));

    _RA_T(GT_SYM, i=sid);
}

//================================================================
/*!@brief
  Execute OP_LOADNIL

  R(A) := nil

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_loadnil(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    _RA_T(GT_NIL, i=0);
}

//================================================================
/*!@brief
  Execute OP_LOADSELF

  R(A) := self

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_loadself(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    _RA(regs[0]);                   	// [ra] <= class
}

//================================================================
/*!@brief
  Execute OP_LOADT

  R(A) := true

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_loadt(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    _RA_T(GT_TRUE, i=0);
}

//================================================================
/*!@brief
  Execute OP_LOADF

  R(A) := false

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_loadf(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    _RA_T(GT_FALSE, i=0);
}

//================================================================
/*!@brief
  Execute OP_GETGLOBAL

  R(A) := getglobal(Syms(Bx))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_getglobal(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS sid  = name2id(VM_SYM(vm, rb));

    guru_obj *obj = global_object_get(sid);

    _RA_X(obj);
}

//================================================================
/*!@brief
  Execute OP_SETGLOBAL

  setglobal(Syms(Bx), R(A))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_setglobal(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS  sid = name2id(VM_SYM(vm, rb));

    global_object_add(sid, &regs[ra]);
}

//================================================================
/*!@brief
  Execute OP_GETIV

  R(A) := ivget(Syms(Bx))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_getiv(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_Bx(code);

    U8P name = VM_SYM(vm, rb);
    GS sid   = name2id(name+1);					// skip '@'
    GV ret   = guru_store_get(&regs[0], sid);

    _RA(ret);
}

//================================================================
/*!@brief
  Execute OP_SETIV

  ivset(Syms(Bx),R(A))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_setiv(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_Bx(code);

    U8P name = VM_SYM(vm, rb);
    GS  sid  = name2id(name+1);			// skip '@'

    guru_store_set(&regs[0], sid, &regs[ra]);
}

//================================================================
/*!@brief
  Execute OP_GETCONST

  R(A) := constget(Syms(Bx))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_getconst(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS  sid = name2id(VM_SYM(vm, rb));

    guru_obj *obj = const_object_get(sid);

    _RA_X(obj);
}

//================================================================
/*!@brief
  Execute OP_SETCONST

  constset(Syms(Bx),R(A))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_setconst(guru_vm *vm) {
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS  sid = name2id(VM_SYM(vm, rb));

    const_object_add(sid, &regs[ra]);
}

//================================================================
/*!@brief
  Execute OP_GETUPVAR

  R(A) := uvget(B,C)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_getupvar(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);
    U32 rc = GETARG_C(code);   		// UP

    guru_state *st = vm->state;

    U32 n = (rc+1) << 1;			// depth of call stack
    while (n > 0){					// walk up call stack
        st = st->prev;
        n--;
    }
    GV *uregs = st->regs;			// outer scope register file

    _RA(uregs[rb]);          // ra <= up[rb]
}

//================================================================
/*!@brief
  Execute OP_SETUPVAR

  uvset(B,C,R(A))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_setupvar(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);
    U32 rc = GETARG_C(code);   				// UP level

    guru_state *st = vm->state;

    U32 n = (rc+1) << 1;					// 2 per outer scope level
    while (n > 0){
        st = st->prev;	CHECK_NULL(st);
        n--;
    }
    GV *uregs = st->regs;					// pointer to caller's register file

    ref_dec(&uregs[rb]);
    uregs[rb] = regs[ra];                   // update outer-scope vars
}

//================================================================
/*!@brief
  Execute OP_JMP

  pc += sBx

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_jmp(guru_vm *vm)
{
	U32 code = vm->bytecode;
    vm->state->pc += GETARG_sBx(code) - 1;
}

//================================================================
/*!@brief
  Execute OP_JMPIF

  if R(A) pc += sBx

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_jmpif (guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    if (regs[GETARG_A(code)].gt > GT_FALSE) {
        vm->state->pc += GETARG_sBx(code) - 1;
    }
}

//================================================================
/*!@brief
  Execute OP_JMPNOT

  if not R(A) pc += sBx

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_jmpnot(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    if (regs[GETARG_A(code)].gt <= GT_FALSE) {
        vm->state->pc += GETARG_sBx(code) - 1;
    }
}


//================================================================
/*!@brief
  Execute OP_SEND / OP_SENDB

  OP_SEND   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
  OP_SENDB  R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
_wipe_stack(GV *regs, U32 rc)
{
    for (U32 i=0; i<=rc; i++) {			// sweep block parameters
    	ref_dec(&regs[i]);
    	regs[i].gt = GT_EMPTY;					// clean up for stat dumper
    }
}

__GURU__ void
op_send(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);  						// proc.sid
    U32 rc = GETARG_C(code);  						// number of params

    GV  *obj = &regs[ra];							// message receiver object
	GS  sid  = name2id(VM_SYM(vm, rb)); 			// function sid

	guru_proc *m = (guru_proc *)proc_by_sid(obj, sid);
    if (m==0) {
    	U8P sym = VM_SYM(vm, rb);
    	_wipe_stack(regs+ra+1, rc);
    	SKIP(sym); 									// function not found, bail out
    }

    if (IS_CFUNC(m)) {
    	if (m->func==prc_call) {					// because VM is not passed to dispatcher,
    		vm_proc_call(vm, regs+ra, rc);			// special handling needed for call() and new()
    	}
    	else if (m->func==obj_new) {
        	vm_object_new(vm, regs+ra, rc);
        }
        else {
        	m->func(obj, rc);						// call the C-func
        }
    	_wipe_stack(regs+ra+1, rc);
    }
    else {											// m->func is a Ruby function (aka IREP)
    	vm_state_push(vm, m->irep, regs+ra, rc);	// append callinfo list
    }
}

//================================================================
/*!@brief
  Execute OP_CALL

  R(A) := self.call(frame.argc, frame.argv)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_call(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	guru_irep *irep = regs[0].proc->irep;

	vm_state_push(vm, irep, regs, 0);
}



//================================================================
/*!@brief
  Execute OP_ENTER

  arg setup according to flags (23=5:5:1:5:5:1:1)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_enter(guru_vm *vm)
{
	U32 code  = vm->bytecode;
    U32 param = GETARG_Ax(code);

    U32 arg0 = (param >> 13) & 0x1f;  // default args
    U32 argc = (param >> 18) & 0x1f;  // given args

    if (arg0 > 0){
        vm->state->pc += vm->state->argc - argc;
    }
}

//================================================================
/*!@brief
  Execute OP_RETURN

  return R(A) (B=normal,in-block return/break)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_return(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra   = GETARG_A(code);
    GV  ret  = regs[ra];

    vm_state_pop(vm, ret, ra);		// pass return value
}

//================================================================
/*!@brief
  Execute OP_BLKPUSH (yield implementation)

  R(A) := block (16=6:1:5:4)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_blkpush(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    GV *stack = regs + 1;       			// use regs[1] as the class

    assert(stack[0].gt==GT_PROC);			// ensure

    _RA(*stack);             				// ra <= proc
}

//================================================================
/*!@brief
  Execute OP_ADD

  R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_add(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    GV  *r0 = &regs[ra];
    GV  *r1 = &regs[ra+1];

    if (r0->gt==GT_INT) {
        if      (r1->gt==GT_INT) 	r0->i += r1->i;
#if GURU_USE_FLOAT
        else if (r1->gt==GT_FLOAT) {	// in case of Fixnum, Float
            r0->gt = GT_FLOAT;
            r0->f  = r0->i + r1->f;
        }
        else SKIP("Fixnum + ?");
    }
    else if (r0->gt==GT_FLOAT) {
        if      (r1->gt==GT_INT) 	r0->f += r1->i;
        else if (r1->gt==GT_FLOAT)	r0->f += r1->f;
        else SKIP("Float + ?");
#endif
    }
    else {    	// other case
    	op_send(vm);			// should have already released regs[ra + n], ...
    }
    r1->gt = GT_EMPTY;
}

//================================================================
/*!@brief
  Execute OP_ADDI

  R(A) := R(A)+C (Syms[B]=:+)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_addi(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
	U32 ra = GETARG_A(code);
	U32 rc = GETARG_C(code);

    GV *r0 = &regs[ra];

    if (r0->gt==GT_INT)     	r0->i += rc;
#if GURU_USE_FLOAT
    else if (r0->gt==GT_FLOAT)	r0->f += rc;
#else
    else QUIT("Float class");
#endif
}

//================================================================
/*!@brief
  Execute OP_SUB

  R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_sub(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    GV *r0 = &regs[ra];
    GV *r1 = &regs[ra+1];

    if (r0->gt==GT_INT) {
        if      (r1->gt==GT_INT) 	r0->i -= r1->i;
#if GURU_USE_FLOAT
        else if (r1->gt==GT_FLOAT) {		// in case of Fixnum, Float
            r0->gt = GT_FLOAT;
            r0->f  = r0->i - r1->f;
        }
        else SKIP("Fixnum - ?");
    }
    else if (r0->gt==GT_FLOAT) {
        if      (r1->gt==GT_INT)	r0->f -= r1->i;
        else if (r1->gt==GT_FLOAT)	r0->f -= r1->f;
        else SKIP("Float - ?");
#endif
    }
    else {  // other case
    	op_send(vm);
    }
    ref_clr(r1);
}

//================================================================
/*!@brief
  Execute OP_SUBI

  R(A) := R(A)-C (Syms[B]=:-)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_subi(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rc = GETARG_C(code);

    GV *r0 = &regs[ra];

    if (r0->gt==GT_INT) 		r0->i -= rc;
#if GURU_USE_FLOAT
    else if (r0->gt==GT_FLOAT) 	r0->f -= rc;
#else
    else QUIT("Float class");
#endif
}

//================================================================
/*!@brief
  Execute OP_MUL

  R(A) := R(A)*R(A+1) (Syms[B]=:*)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_mul(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    GV *r0 = &regs[ra];
    GV *r1 = &regs[ra+1];

    if (r0->gt==GT_INT) {
        if      (r1->gt==GT_INT) 	r0->i *= r1->i;
#if GURU_USE_FLOAT
        else if (r1->gt==GT_FLOAT) {	// in case of Fixnum, Float
            r0->gt = GT_FLOAT;
            r0->f  = r0->i * r1->f;
        }
        else SKIP("Fixnum * ?");
    }
    else if (r0->gt==GT_FLOAT) {
        if      (r1->gt==GT_INT) 	r0->f *= r1->i;
        else if (r1->gt==GT_FLOAT)  r0->f *= r1->f;
        else SKIP("Float * ?");
#endif
    }
    else {   // other case
    	op_send(vm);
    }
    ref_clr(r1);
}

//================================================================
/*!@brief
  Execute OP_DIV

  R(A) := R(A)/R(A+1) (Syms[B]=:/)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_div(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    GV *r0 = &regs[ra];
    GV *r1 = &regs[ra+1];

    if (r0->gt==GT_INT) {
        if      (r1->gt==GT_INT) 	r0->i /= r1->i;
#if GURU_USE_FLOAT
        else if (r1->gt==GT_FLOAT) {		// in case of Fixnum, Float
            r0->gt = GT_FLOAT;
            r0->f  = r0->i / r1->f;
        }
        else SKIP("Fixnum / ?");
    }
    else if (r0->gt==GT_FLOAT) {
        if      (r1->gt==GT_INT) 	r0->f /= r1->i;
        else if (r1->gt==GT_FLOAT)	r0->f /= r1->f;
        else SKIP("Float / ?");
#endif
    }
    else {   // other case
    	op_send(vm);
    }
    ref_clr(r1);
}

//================================================================
/*!@brief
  Execute OP_EQ

  R(A) := R(A)==R(A+1)  (Syms[B]=:==,C=1)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_eq(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    GT  tt = GT_BOOL(guru_cmp(&regs[ra], &regs[ra+1])==0);

    regs[ra+1].gt = GT_EMPTY;
    _RA_T(tt, i=0);
}

// macro for comparators
#define ncmp(r0, op, r1)								\
do {													\
	if ((r0)->gt==GT_INT) {								\
		if ((r1)->gt==GT_INT) {							\
			(r0)->gt = GT_BOOL((r0)->i op (r1)->i);		\
		}												\
		else if ((r1)->gt==GT_FLOAT) {					\
			(r0)->gt = GT_BOOL((r0)->i op (r1)->f);		\
		}												\
	}													\
	else if ((r0)->gt==GT_FLOAT) {						\
		if ((r1)->gt==GT_INT) {							\
			(r0)->gt = GT_BOOL((r0)->f op (r1)->i);		\
		}												\
		else if ((r1)->gt==GT_FLOAT) {					\
			(r0)->gt = GT_BOOL((r0)->f op (r1)->f);		\
		}												\
	}													\
	else {												\
		op_send(vm);						\
	}													\
    ref_clr(r1);										\
} while (0)

//================================================================
/*!@brief
  Execute OP_LT

  R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_lt(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
	U32 ra = GETARG_A(code);

	ncmp(&regs[ra], <, &regs[ra+1]);

    regs[ra+1].gt = GT_EMPTY;
}

//================================================================
/*!@brief
  Execute OP_LE

  R(A) := R(A)<=R(A+1)  (Syms[B]=:<=,C=1)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_le(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    ncmp(&regs[ra], <=, &regs[ra+1]);

    regs[ra+1].gt = GT_EMPTY;
}

//================================================================
/*!@brief
  Execute OP_GT

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_gt(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    ncmp(&regs[ra], >, &regs[ra+1]);

    regs[ra+1].gt = GT_EMPTY;
}

//================================================================
/*!@brief
  Execute OP_GE

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_ge(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

    U32 ra = GETARG_A(code);

    ncmp(&regs[ra], >=, &regs[ra+1]);

    regs[ra+1].gt = GT_EMPTY;
}

//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_string(guru_vm *vm)
{
#if GURU_USE_STRING
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

	U32 ra = GETARG_A(code);
    U32 rb = GETARG_Bx(code);

    U8  *str = (U8 *)VM_VAR(vm, rb);			// string pool var
    GV  ret  = guru_str_new(str);				// rc set to 1 already

    _RA(ret);
#else
    QUIT("String class");
#endif
}

//================================================================
/*!@brief
  String Catination

  str_cat(R(A),R(B))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_strcat(guru_vm *vm)
{
#if GURU_USE_STRING
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);

    GV *va  = &regs[ra];
    GV *vb  = &regs[rb];
    GS sid  = name2id((U8P)"to_s");				// from global symbol pool

    guru_proc *pa = proc_by_sid(va, sid);
    guru_proc *pb = proc_by_sid(vb, sid);

    if (pa) pa->func(va, 0);
    if (pb) pb->func(vb, 0);

    GV ret = guru_str_add(va, vb);				// ref counts updated

    ref_dec(va);								// free both strings
    ref_dec(vb);
    vb->gt = GT_EMPTY;

    _RA_X(&ret);

#else
    QUIT("String class");
#endif
}

//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_array(guru_vm *vm)
{
#if GURU_USE_ARRAY
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_B(code);
    U32 n   = GETARG_C(code);

    GV  ret = (GV)guru_array_new(n);	// ref_cnt is 1 already
    guru_array *h  = ret.array;

    GV *s = &regs[rb];					// source elements
	GV *d = h->data;					// target
	for (U32 i=0; i<(n); i++, ref_inc(s), *d++=*s, s->gt=GT_EMPTY, s++);
    h->n = n;

    _RA(ret);							// no need to ref_inc
#else
    QUIT("Array class");
#endif
}

//================================================================
/*!@brief
  Create Hash object

  R(A) := hash_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_hash(guru_vm *vm)
{
#if GURU_USE_ARRAY
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);
    U32 n  = GETARG_C(code);							// entries of hash
    U32 sz = sizeof(GV) * (n<<1);						// size of k,v pairs

    GV *p  = &regs[rb];
    GV ret = guru_hash_new(n);							// ref_cnt is already set to 1
    guru_hash  *h = ret.hash;

    MEMCPY((U8P)h->data, (U8P)p, sz);					// copy k,v pairs

    for (U32 i=0; i<(h->n=(n<<1)); i++, p++) {
    	p->gt = GT_EMPTY;								// clean up call stack
    }
    _RA(ret);						          			// set return value on stack top
#else
    QUIT("Hash class");
#endif
}

//================================================================
/*!@brief
  Execute OP_RANGE

  R(A) := range_new(R(B),R(B+1),C)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_range(guru_vm *vm)
{
#if GURU_USE_ARRAY
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);
    U32 n  = GETARG_C(code);

    GV *pb = &regs[rb];
    GV ret = guru_range_new(pb, pb+1, n);	// pb, pb+1 ref cnt will be increased
    regs[rb+1].gt = GT_EMPTY;

    _RA_X(&ret);							// release and  reassign

#else
    QUIT("Range class");
#endif
}

//================================================================
/*!@brief
  Execute OP_LAMBDA

  R(A) := lambda(SEQ[Bz],Cz)

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_lambda(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    int ra = GETARG_A(code);
    int rb = GETARG_b(code);      		// sequence position in irep list
    int rc = GETARG_C(code);    		// TODO: Add flags support for OP_LAMBDA

    guru_proc *prc = (guru_proc *)guru_alloc(sizeof(guru_proc));

    prc->func = NULL;					// not a c-func (i.e. a Ruby func)
    prc->irep = VM_REPS(vm, rb);		// fetch from children irep list

    _RA_T(GT_PROC, proc=prc);
}

//================================================================
/*!@brief
  Execute OP_CLASS

  R(A) := newclass(R(A),Syms(B),R(A+1))
  Syms(B): class name
  R(A+1): super class

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_class(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);

    guru_class *super = (regs[ra+1].gt==GT_CLASS) ? regs[ra+1].cls : guru_class_object;
    const U8P  name   = VM_SYM(vm, rb);
    guru_class *cls   = guru_define_class(name, super);

    _RA_T(GT_CLASS, cls=cls);
}

//================================================================
/*!@brief
  Execute OP_EXEC

  R(A) := blockexec(R(A),SEQ[Bx])

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_exec(guru_vm *vm)
{
	U32 code  = vm->bytecode;
	GV  *regs = vm->state->regs;

    U32 ra = GETARG_A(code);					// receiver
    U32 rb = GETARG_Bx(code);					// irep pointer

    guru_irep *irep = VM_REPS(vm, rb);

    vm_state_push(vm, irep, regs+ra, 0);		// push call stack
}

//================================================================
/*!@brief
  Execute OP_METHOD

  R(A).newmethod(Syms(B),R(A+1))

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_method(guru_vm *vm)
{
	GV  *regs = vm->state->regs;
	U32 code  = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);

    assert(regs[ra].gt == GT_CLASS);		// enforce class checking

    // check whether the name has been defined in the same class
    guru_class 	*cls = regs[ra].cls;
    GS			sid  = name2id(VM_SYM(vm, rb));
    guru_proc 	*prc = proc_by_sid(&regs[ra], sid);

    assert(prc == NULL);					// reject same name for now

    prc = regs[ra+1].proc;					// setup the new proc

    MUTEX_LOCK(_mutex_op);

    // add proc to class
    prc->sid  = sid;
    prc->next = cls->vtbl;					// add to top of vtable
    cls->vtbl = prc;

    MUTEX_FREE(_mutex_op);

    for (U32 i=1; i<=ra+1; i++) {
    	ref_dec(&regs[i]);
    	regs[ra+1].gt = GT_EMPTY;
    }
}

//================================================================
/*!@brief
  Execute OP_TCLASS

  R(A) := target_class

  @param  vm    A pointer of VM.
  @retval 0  	No error.
*/
__GURU__ void
op_tclass(guru_vm *vm)
{
	U32 code  = vm->bytecode;
	GV  *regs = vm->state->regs;
	U32 ra = GETARG_A(code);

	_RA_T(GT_CLASS, cls=vm->state->klass);
}

//================================================================
/*!@brief
  Execute OP_STOP and OP_ABORT

  stop VM (OP_STOP)
  stop VM without release memory (OP_ABORT)

  @param  vm    A pointer of VM.
  @retval -1  No error and exit from vm.
*/
__GURU__ void
op_stop(guru_vm *vm)
{
	vm->run  = VM_STATUS_HOLD;	// VM suspended
}

__GURU__ void
op_abort(guru_vm *vm)
{
	vm->run = VM_STATUS_FREE;	// exit guru_op loop
}

//===========================================================================================
// GURU engine
//===========================================================================================
__GURU__ void
guru_op(guru_vm *vm)
{
	if (threadIdx.x != 0) return;	// TODO: multi-thread [run|suspend] queues

	guru_state *st = vm->state;		// capture pointer for memory debugging

	//=======================================================================================
	// GURU dispatcher unit
	//=======================================================================================
    switch (vm->op) {
    // LOAD,STORE
    case OP_LOADL:      op_loadl     (vm); break;
    case OP_LOADI:      op_loadi     (vm); break;
    case OP_LOADSYM:    op_loadsym   (vm); break;
    case OP_LOADNIL:    op_loadnil   (vm); break;
    case OP_LOADSELF:   op_loadself  (vm); break;
    case OP_LOADT:      op_loadt     (vm); break;
    case OP_LOADF:      op_loadf     (vm); break;
    // VARIABLES
    case OP_GETGLOBAL:  op_getglobal (vm); break;
    case OP_SETGLOBAL:  op_setglobal (vm); break;
    case OP_GETIV:      op_getiv     (vm); break;
    case OP_SETIV:      op_setiv     (vm); break;
    case OP_GETCONST:   op_getconst  (vm); break;
    case OP_SETCONST:   op_setconst  (vm); break;
    case OP_GETUPVAR:   op_getupvar  (vm); break;
    case OP_SETUPVAR:   op_setupvar  (vm); break;
    // BRANCH
    case OP_JMP:        op_jmp       (vm); break;
    case OP_JMPIF:      op_jmpif     (vm); break;
    case OP_JMPNOT:     op_jmpnot    (vm); break;
    // CALL
    case OP_SEND:       op_send      (vm); break;
    case OP_SENDB:      op_send      (vm); break;  // reuse
    case OP_CALL:       op_call      (vm); break;
    case OP_ENTER:      op_enter     (vm); break;
    case OP_RETURN:     op_return    (vm); break;
    case OP_BLKPUSH:    op_blkpush   (vm); break;
    // ALU
    case OP_MOVE:       op_move      (vm); break;
    case OP_ADD:        op_add       (vm); break;
    case OP_ADDI:       op_addi      (vm); break;
    case OP_SUB:        op_sub       (vm); break;
    case OP_SUBI:       op_subi      (vm); break;
    case OP_MUL:        op_mul       (vm); break;
    case OP_DIV:        op_div       (vm); break;
    case OP_EQ:         op_eq        (vm); break;
    case OP_LT:         op_lt        (vm); break;
    case OP_LE:         op_le        (vm); break;
    case OP_GT:         op_gt        (vm); break;
    case OP_GE:         op_ge        (vm); break;
    // BUILT-IN class (TODO: tensor)
#if GURU_USE_STRING
    case OP_STRING:     op_string    (vm); break;
    case OP_STRCAT:     op_strcat    (vm); break;
#endif
#if GURU_USE_ARRAY
    case OP_ARRAY:      op_array     (vm); break;
    case OP_HASH:       op_hash      (vm); break;
    case OP_RANGE:      op_range     (vm); break;
#endif
    // CLASS, PROC (STACK ops)
    case OP_LAMBDA:     op_lambda    (vm); break;
    case OP_CLASS:      op_class     (vm); break;
    case OP_EXEC:       op_exec      (vm); break;
    case OP_METHOD:     op_method    (vm); break;
    case OP_TCLASS:     op_tclass    (vm); break;
    // CONTROL
    case OP_STOP:       op_stop      (vm); break;
    case OP_ABORT:      op_abort     (vm); break;  	// reuse
    case OP_NOP:        op_nop       (vm); break;
    default:
    	PRINTF("?OP=0x%04x\n", vm->op);
    	vm->err = 1;
    	break;
    }
}

