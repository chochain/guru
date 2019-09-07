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
#define _RA(v)      	(regs[ra]=(v), 0)
#define _RA_T(t, e) 	(regs[ra].gt=(t), regs[ra].e, 0)
#define _RA_X(r)    	(ref_dec(&regs[ra]), regs[ra]=*(r), ref_inc(r), 0)
#define _RA_T2(t,e)     (_R(a).gt=(t), _R(a).e, 0)
//================================================================
/*!@brief
  Execute OP_NOP

  No operation

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_nop(guru_vm *vm)
{
    return 0;
}

//================================================================
/*!@brief
  Execute OP_MOVE

  R(A) := R(B)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_move(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
	U32 ra = GETARG_A(code);
	U32 rb = GETARG_B(code);

    return _RA(regs[rb]);                  	// [ra] <= [rb]
}

//================================================================
/*!@brief
  Execute OP_LOADL

  R(A) := Pool(Bx)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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
    return _RA(obj);
}

//================================================================
/*!@brief
  Execute OP_LOADI

  R(A) := sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_loadi(guru_vm *vm)
{
    GI rb = _ARG(bx) - MAXARG_sBx;		// sBx

    return _RA_T2(GT_INT, i=rb);
}


//================================================================
/*!@brief
  Execute OP_LOADSYM

  R(A) := Syms(Bx)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_loadsym(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS  sid = name2id(VM_SYM(vm, rb));

    return _RA_T(GT_SYM, i=sid);
}

//================================================================
/*!@brief
  Execute OP_LOADNIL

  R(A) := nil

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_loadnil(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    return _RA_T(GT_NIL, i=0);
}

//================================================================
/*!@brief
  Execute OP_LOADSELF

  R(A) := self

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_loadself(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    return _RA(regs[0]);                   	// [ra] <= class
}

//================================================================
/*!@brief
  Execute OP_LOADT

  R(A) := true

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_loadt(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    return _RA_T(GT_TRUE, i=0);
}

//================================================================
/*!@brief
  Execute OP_LOADF

  R(A) := false

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_loadf(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    return _RA_T(GT_FALSE, i=0);
}

//================================================================
/*!@brief
  Execute OP_GETGLOBAL

  R(A) := getglobal(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_getglobal(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS sid  = name2id(VM_SYM(vm, rb));

    guru_obj obj = global_object_get(sid);

    return _RA(obj);
}

//================================================================
/*!@brief
  Execute OP_SETGLOBAL

  setglobal(Syms(Bx), R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_setglobal(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS  sid = name2id(VM_SYM(vm, rb));

    global_object_add(sid, &regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETIV

  R(A) := ivget(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_getiv(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_Bx(code);

    U8P name = VM_SYM(vm, rb);
    GS sid   = name2id(name+1);					// skip '@'
    GV ret   = guru_store_get(&regs[0], sid);

    return _RA(ret);
}

//================================================================
/*!@brief
  Execute OP_SETIV

  ivset(Syms(Bx),R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_setiv(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_Bx(code);

    U8P name = VM_SYM(vm, rb);
    GS  sid  = name2id(name+1);			// skip '@'

    guru_store_set(&regs[0], sid, &regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETCONST

  R(A) := constget(Syms(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_getconst(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS  sid = name2id(VM_SYM(vm, rb));

    guru_obj obj = const_object_get(sid);

    return _RA(obj);
}

//================================================================
/*!@brief
  Execute OP_SETCONST

  constset(Syms(Bx),R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_setconst(guru_vm *vm) {
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    U32 rb  = GETARG_Bx(code);
    GS  sid = name2id(VM_SYM(vm, rb));

    const_object_add(sid, &regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GETUPVAR

  R(A) := uvget(B,C)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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

    return _RA(uregs[rb]);          // ra <= up[rb]
}

//================================================================
/*!@brief
  Execute OP_SETUPVAR

  uvset(B,C,R(A))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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
        st = st->prev;
        n--;
    }
    GV *uregs = st->regs;

    ref_clr(&uregs[rb]);
    uregs[rb] = regs[ra];                   // update outer-scope vars
    ref_inc(&regs[ra]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMP

  pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_jmp(guru_vm *vm)
{
	U32 code = vm->bytecode;
    vm->state->pc += GETARG_sBx(code) - 1;
    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMPIF

  if R(A) pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_jmpif (guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    if (regs[GETARG_A(code)].gt > GT_FALSE) {
        vm->state->pc += GETARG_sBx(code) - 1;
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_JMPNOT

  if not R(A) pc += sBx

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_jmpnot(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    if (regs[GETARG_A(code)].gt <= GT_FALSE) {
        vm->state->pc += GETARG_sBx(code) - 1;
    }
    return 0;
}


//================================================================
/*!@brief
  Execute OP_SEND / OP_SENDB

  OP_SEND   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
  OP_SENDB  R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_send(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);  			// index of method sym
    U32 rc = GETARG_C(code);  			// number of params

    GV  *rcv = &regs[ra];				// receiver object
	GS  sid  = name2id(VM_SYM(vm, rb)); // function sid

	guru_proc *m = (guru_proc *)proc_by_sid(rcv, sid);
    if (m==0) {
    	U8P sym = VM_SYM(vm, rb);
    	guru_na(sym);					// dump error, bail out
    	return 0;
    }

    // Clear block param (needed ?)
    U32 bidx = ra + rc + 1;
    switch(GET_OPCODE(code)) {
    case OP_SEND:
        ref_clr(&regs[bidx]);
        regs[bidx].gt = GT_NIL;
        break;
    case OP_SENDB:						// set Proc object
        if (regs[bidx].gt != GT_NIL && regs[bidx].gt != GT_PROC){
            // TODO: fix the following behavior
            // convert to Proc ?
            // raise exceprion in mruby/c ?
            return 0;
        }
        break;
    default: break;
    }

    if (IS_CFUNC(&regs[rb])) {
    	if (m->func==c_proc_call) {		// because VM is not passed to dispatcher, special handling needed for call() and new()
    		vm_proc_call(vm, regs+ra, rc);
        }
        else if (m->func==c_object_new) {
        	vm_object_new(vm, regs+ra, rc);
        }
        else {
        	if (vm->step) printf("%s#%s\n", m->cname, m->name);
        	m->func(regs+ra, rc);					// call the C-func
            for (U32 i=ra+1; i<=bidx; i++) {		// clean up block parameters
            	ref_clr(&regs[i]);
            }
        }
    }
    else {								// m->func is a Ruby function (aka IREP)
    	vm_state_push(vm, rc);			// append callinfo list

    	vm->state->irep = m->irep;		// call into target context
    	vm->state->pc 	= 0;			// call into target context
    	vm->state->regs += ra;			// add call stack (new register)
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_CALL

  R(A) := self.call(frame.argc, frame.argv)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_call(guru_vm *vm)
{
	GV *regs = vm->state->regs;
    vm_state_push(vm, 0);

    // jump to proc
    vm->state->pc 	= 0;
    vm->state->irep = regs[0].proc->irep;

    return 0;
}



//================================================================
/*!@brief
  Execute OP_ENTER

  arg setup according to flags (23=5:5:1:5:5:1:1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_enter(guru_vm *vm)
{
	U32 code  = vm->bytecode;
    U32 param = GETARG_Ax(code);

    U32 arg0 = (param >> 13) & 0x1f;  // default args
    U32 argc = (param >> 18) & 0x1f;  // given args

    if (arg0 > 0){
        vm->state->pc += vm->state->argc - argc;
    }
    return 0;
}

//================================================================
/*!@brief
  Execute OP_RETURN

  return R(A) (B=normal,in-block return/break)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_return(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    // return value
    U32 ra  = GETARG_A(code);
    GV  ret = regs[ra];

    ref_clr(&regs[0]);
    regs[0]     = ret;
    regs[ra].gt = GT_EMPTY;

    vm_state_pop(vm, regs);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_BLKPUSH

  R(A) := block (16=6:1:5:4)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_blkpush(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    GV *stack = regs + 1;       	// call stack: push 1 GV

    if (stack[0].gt==GT_NIL){		// Check leak?
        return vm->err = 255;  		// EYIELD
    }
    return _RA(*stack);             // ra <= stack[0]
}

//================================================================
/*!@brief
  Execute OP_ADD

  R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_add(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra  = GETARG_A(code);
    GV  *r0 = &regs[ra];
    GV  *r1 = &regs[ra+1];

    if (r0->gt==GT_INT) {
        if      (r1->gt==GT_INT) r0->i += r1->i;
#if GURU_USE_FLOAT
        else if (r1->gt==GT_FLOAT) {	// in case of Fixnum, Float
            r0->gt = GT_FLOAT;
            r0->f = r0->i + r1->f;
        }
        else guru_na("Fixnum + ?");
    }
    else if (r0->gt==GT_FLOAT) {
        if      (r1->gt==GT_INT) r0->f += r1->i;
        else if (r1->gt==GT_FLOAT)	 r0->f += r1->f;
        else guru_na("Float + ?");
#endif
    }
    else {    	// other case
    	op_send(vm);			// should have already released regs[ra + n], ...
    }
    ref_clr(r1);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_ADDI

  R(A) := R(A)+C (Syms[B]=:+)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_addi(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
	U32 ra = GETARG_A(code);
	U32 rc = GETARG_C(code);

    GV *r0 = &regs[ra];

    if (r0->gt==GT_INT)     r0->i += rc;
#if GURU_USE_FLOAT
    else if (r0->gt==GT_FLOAT)	r0->f += rc;
#else
    else guru_na("Float class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_SUB

  R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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
        else guru_na("Fixnum - ?");
    }
    else if (r0->gt==GT_FLOAT) {
        if      (r1->gt==GT_INT)	r0->f -= r1->i;
        else if (r1->gt==GT_FLOAT)		r0->f -= r1->f;
        else guru_na("Float - ?");
#endif
    }
    else {  // other case
    	op_send(vm);
    }
    ref_clr(r1);
	return 0;
}

//================================================================
/*!@brief
  Execute OP_SUBI

  R(A) := R(A)-C (Syms[B]=:-)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_subi(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rc = GETARG_C(code);

    GV *r0 = &regs[ra];

    if (r0->gt==GT_INT) 	r0->i -= rc;
#if GURU_USE_FLOAT
    else if (r0->gt==GT_FLOAT) r0->f -= rc;
#else
    else guru_na("Float class");
#endif
    return 0;
}

//================================================================
/*!@brief
  Execute OP_MUL

  R(A) := R(A)*R(A+1) (Syms[B]=:*)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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
        else guru_na("Fixnum * ?");
    }
    else if (r0->gt==GT_FLOAT) {
        if      (r1->gt==GT_INT) r0->f *= r1->i;
        else if (r1->gt==GT_FLOAT)  r0->f *= r1->f;
        else guru_na("Float * ?");
#endif
    }
    else {   // other case
    	op_send(vm);
    }
    ref_clr(r1);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_DIV

  R(A) := R(A)/R(A+1) (Syms[B]=:/)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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
        else guru_na("Fixnum / ?");
    }
    else if (r0->gt==GT_FLOAT) {
        if      (r1->gt==GT_INT) 	r0->f /= r1->i;
        else if (r1->gt==GT_FLOAT)		r0->f /= r1->f;
        else guru_na("Float / ?");
#endif
    }
    else {   // other case
    	op_send(vm);
    }
    ref_clr(r1);
    return 0;
}

//================================================================
/*!@brief
  Execute OP_EQ

  R(A) := R(A)==R(A+1)  (Syms[B]=:==,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_eq(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    GT  tt = GT_BOOL(guru_cmp(&regs[ra], &regs[ra+1])==0);

    return _RA_T(tt, i=0);
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
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_lt(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
	U32 ra = GETARG_A(code);

	ncmp(&regs[ra], <, &regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_LE

  R(A) := R(A)<=R(A+1)  (Syms[B]=:<=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ U32
op_le(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    ncmp(&regs[ra], <=, &regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GT

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_gt(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);

    ncmp(&regs[ra], >, &regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_GE

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_ge(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

    U32 ra = GETARG_A(code);

    ncmp(&regs[ra], >=, &regs[ra+1]);

    return 0;
}

//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_string(guru_vm *vm)
{
#if GURU_USE_STRING
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

	U32 ra = GETARG_A(code);
    U32 rb = GETARG_Bx(code);

    U32 *v = VM_VAR(vm, rb);
    U8  *str = (U8P)U8PADD(VM_IREP(vm), *v);
    GV  ret  = guru_str_new(str);				// rc set to 1 already

    return _RA(ret);
#else
    guru_na("String class");
#endif
}

//================================================================
/*!@brief
  String Catination

  str_cat(R(A),R(B))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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

    GV ret = guru_str_add(va, vb);

    return _RA(ret);

#else
    guru_na("String class");
#endif
}

//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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

    return _RA(ret);					// no need to ref_inc
#else
    guru_na("Array class");
#endif
}

//================================================================
/*!@brief
  Create Hash object

  R(A) := hash_new(R(B),R(B+1)..R(B+C))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
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
    return _RA(ret);						          	// set return value on stack top
#else
    guru_na("Hash class");
#endif
}

//================================================================
/*!@brief
  Execute OP_RANGE

  R(A) := range_new(R(B),R(B+1),C)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_range(guru_vm *vm)
{
#if GURU_USE_ARRAY
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);
    U32 n  = GETARG_C(code);

    GV *pb = &regs[rb];

    ref_inc(pb);
    ref_inc(pb+1);
    GV ret = guru_range_new(pb, pb+1, n);

    return _RA(ret);						// release and  reassign

#else
    guru_na("Range class");
#endif
}

//================================================================
/*!@brief
  Execute OP_LAMBDA

  R(A) := lambda(SEQ[Bz],Cz)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_lambda(guru_vm *vm)
{
    guru_na("OP_LAMBDA");
	return 0;							// no support for metaprogramming yet
	/*
    int ra = GETARG_A(code);
    int rb = GETARG_b(code);      		// sequence position in irep list
    // int c = GETARG_C(code);    		// TODO: Add flags support for OP_LAMBDA

    guru_proc *prc = (guru_proc *)guru_alloc_proc((U8P)"(lambda)");

    prc->irep = vm_irep_list(vm, rb);
    prc->flag &= ~GURU_CFUNC;           // Ruby IREP

    _RA_T(GT_PROC, proc=prc);

    return 0;
    */
}

//================================================================
/*!@brief
  Execute OP_CLASS

  R(A) := newclass(R(A),Syms(B),R(A+1))
  Syms(B): class name
  R(A+1): super class

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_class(guru_vm *vm)
{
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);

    guru_class *super = (regs[ra+1].gt==GT_CLASS) ? regs[ra+1].cls : guru_class_object;
    const U8P  name   = VM_SYM(vm, rb);
    guru_class *cls   = guru_define_class(name, super);

    return _RA_T(GT_CLASS, cls=cls);
}

//================================================================
/*!@brief
  Execute OP_EXEC

  R(A) := blockexec(R(A),SEQ[Bx])

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_exec(guru_vm *vm)
{
	U32 code  = vm->bytecode;
	GV  *regs = vm->state->regs;

    U32 ra = GETARG_A(code);
    U32 rb = GETARG_Bx(code);

    GV rcv = regs[ra];									// receiver

    vm_state_push(vm, 0);								// push call stack

    vm->state->irep  = VM_REPS(vm, rb);					// fetch designated irep
    vm->state->pc 	 = 0;								// switch context to callee
    vm->state->regs += ra;								// shift regfile (for local stack)
    vm->state->klass = class_by_obj(&rcv);

    return 0;
}

//================================================================
/*!@brief
  Execute OP_METHOD

  R(A).newmethod(Syms(B),R(A+1))

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_method(guru_vm *vm)
{
	GV  *regs = vm->state->regs;
	U32 code  = vm->bytecode;
    U32 ra = GETARG_A(code);
    U32 rb = GETARG_B(code);

    if (regs[ra].gt != GT_CLASS) {
    	PRINTF("?op_method");
    	return 0;
    }

    guru_class 	*cls  = regs[ra].cls;
    GS			sid   = name2id(VM_SYM(vm, rb));
    guru_proc 	*prc0 = proc_by_sid(&regs[ra], sid);

    assert(prc0 == NULL);					// TODO: reject same name for now

    guru_proc 	*prc1 = regs[ra+1].proc;

    MUTEX_LOCK(_mutex_op);

    // add proc to class
    prc1->sid 	= sid;
    prc1->next  = cls->vtbl;				// add to top of vtable
    cls->vtbl   = prc1;

    MUTEX_FREE(_mutex_op);

    regs[ra+1].gt = GT_EMPTY;

    return 0;
}

//================================================================
/*!@brief
  Execute OP_TCLASS

  R(A) := target_class

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval 0  No error.
*/
__GURU__ int
op_tclass(guru_vm *vm)
{
	U32 code  = vm->bytecode;
	GV  *regs = vm->state->regs;
	U32 ra = GETARG_A(code);

	return _RA_T(GT_CLASS, cls=vm->state->klass);
}

//================================================================
/*!@brief
  Execute OP_STOP and OP_ABORT

  stop VM (OP_STOP)
  stop VM without release memory (OP_ABORT)

  @param  vm    A pointer of VM.
  @param  code  bytecode
  @param  regs  pointer to regfile
  @retval -1  No error and exit from vm.
*/
__GURU__ int
op_stop(guru_vm *vm)
{
	vm->run = 0;
    return -1;		// exit guru_op loop
}

__GURU__ int
op_abort(guru_vm *vm)
{
	return -1;		// exit guru_op loop
}

__GURU__ void
_prefetch(guru_vm *vm) {
	vm->bytecode = VM_BYTECODE(vm);

	vm->op  = vm->bytecode & 0x7f;
	vm->opn = vm->bytecode >> 7;
	vm->ar 	= (GAR *)&(vm)->opn;

	vm->state->pc++;
}

//===========================================================================================
// GURU engine
//===========================================================================================
__GURU__ int
guru_op(guru_vm *vm)
{
	if (threadIdx.x != 0) return 0;	// TODO: multi-thread [run|suspend] queues

	//=======================================================================================
	// GURU instruction unit
	//=======================================================================================
	_prefetch(vm);					// fetch & advance program counter, ready for next cycle

    //=======================================================================================
	// GURU dispatcher unit
	//=======================================================================================
	int ret;
    switch (vm->op) {
    // LOAD,STORE
    case OP_LOADL:      ret = op_loadl     (vm); break;
    case OP_LOADI:      ret = op_loadi     (vm); break;
    case OP_LOADSYM:    ret = op_loadsym   (vm); break;
    case OP_LOADNIL:    ret = op_loadnil   (vm); break;
    case OP_LOADSELF:   ret = op_loadself  (vm); break;
    case OP_LOADT:      ret = op_loadt     (vm); break;
    case OP_LOADF:      ret = op_loadf     (vm); break;
    // VARIABLES
    case OP_GETGLOBAL:  ret = op_getglobal (vm); break;
    case OP_SETGLOBAL:  ret = op_setglobal (vm); break;
    case OP_GETIV:      ret = op_getiv     (vm); break;
    case OP_SETIV:      ret = op_setiv     (vm); break;
    case OP_GETCONST:   ret = op_getconst  (vm); break;
    case OP_SETCONST:   ret = op_setconst  (vm); break;
    case OP_GETUPVAR:   ret = op_getupvar  (vm); break;
    case OP_SETUPVAR:   ret = op_setupvar  (vm); break;
    // BRANCH
    case OP_JMP:        ret = op_jmp       (vm); break;
    case OP_JMPIF:      ret = op_jmpif     (vm); break;
    case OP_JMPNOT:     ret = op_jmpnot    (vm); break;
    case OP_SEND:       ret = op_send      (vm); break;
    case OP_SENDB:      ret = op_send      (vm); break;  // reuse
    case OP_CALL:       ret = op_call      (vm); break;
    case OP_ENTER:      ret = op_enter     (vm); break;
    case OP_RETURN:     ret = op_return    (vm); break;
    case OP_BLKPUSH:    ret = op_blkpush   (vm); break;
    // ALU
    case OP_MOVE:       ret = op_move      (vm); break;
    case OP_ADD:        ret = op_add       (vm); break;
    case OP_ADDI:       ret = op_addi      (vm); break;
    case OP_SUB:        ret = op_sub       (vm); break;
    case OP_SUBI:       ret = op_subi      (vm); break;
    case OP_MUL:        ret = op_mul       (vm); break;
    case OP_DIV:        ret = op_div       (vm); break;
    case OP_EQ:         ret = op_eq        (vm); break;
    case OP_LT:         ret = op_lt        (vm); break;
    case OP_LE:         ret = op_le        (vm); break;
    case OP_GT:         ret = op_gt        (vm); break;
    case OP_GE:         ret = op_ge        (vm); break;
    // BUILT-IN class (TODO: tensor)
#if GURU_USE_STRING
    case OP_STRING:     ret = op_string    (vm); break;
    case OP_STRCAT:     ret = op_strcat    (vm); break;
#endif
#if GURU_USE_ARRAY
    case OP_ARRAY:      ret = op_array     (vm); break;
    case OP_HASH:       ret = op_hash      (vm); break;
    case OP_RANGE:      ret = op_range     (vm); break;
#endif
    // CLASS, PROC (STACK ops)
    case OP_LAMBDA:     ret = op_lambda    (vm); break;
    case OP_CLASS:      ret = op_class     (vm); break;
    case OP_EXEC:       ret = op_exec      (vm); break;
    case OP_METHOD:     ret = op_method    (vm); break;
    case OP_TCLASS:     ret = op_tclass    (vm); break;
    // CONTROL
    case OP_STOP:       ret = op_stop      (vm); break;
    case OP_ABORT:      ret = op_abort     (vm); break;  	// reuse
    case OP_NOP:        ret = op_nop       (vm); break;
    default:
    	PRINTF("?OP=0x%04x\n", vm->op);
    	ret = 0;
    	break;
    }
    return ret;
}

