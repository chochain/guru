/*! @file
  @brief
  GURU instruction unit - prefetch & microcode dispatcher

  <pre>
  Copyright (C) 2019 GreenII

  1. a list of opcode (microcode) executor, and
  2. the core opcode dispatcher
  </pre>
*/
#include "guru.h"
#include "base.h"
#include "static.h"
#include "global.h"
#include "symbol.h"
#include "mmu.h"
#include "ostore.h"
#include "iter.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

#include "inspect.h"
#include "state.h"
#include "ucode.h"

#define _LOCK		{ MUTEX_LOCK(_mutex_uc); }
#define _UNLOCK		{ MUTEX_FREE(_mutex_uc); }

__GURU__ U32 _mutex_uc;
//
// becareful with the following macros, because they release regs[ra] first
// so, make sure value is kept before the release
//
#define _R0             (_REGS(VM_STATE(vm)))
#define _R(e)			(_R0 + vm->e)
#define _RA(r)			(ref_dec(_R(a)), *_R(a)=(r))
#define _RA_X(r)    	(ref_inc(r), ref_dec(_R(a)), *_R(a)=*(r))
#define _RA_T(t,e)      (ref_dec(_R(a)), _R(a)->gt=(t), _R(a)->acl=0, _R(a)->e)

#define SKIP(x)			{ NA(x); return; }
#define RAISE(x)	    { _RA(guru_str_new(x)); vm->err = 1; return; }
#define QUIT(x)			{ vm->err=1; vm->run=VM_STATUS_STOP; NA(x); return; }

__GURU__ __INLINE__ GR *nop()	{ return NULL; }

//================================================================
/*!@brief
  OP_NOP
  No operation
*/
__UCODE__
uc_nop(guru_vm *vm)
{
	// do nothing
}

//================================================================
/*!@brief
  OP_MOVE

  R(A) := R(B)
*/
__UCODE__
uc_move(guru_vm *vm)
{
	_RA_X(_R(b)); 	 			// [ra] <= [rb]
}

//================================================================
/*!@brief
  OP_LOADL

  R(A) := Pool(Bx)
*/
__UCODE__
uc_loadl(guru_vm *vm)
{
	GR ret = *VM_VAR(vm, vm->bx);
    _RA(ret);
}

//================================================================
/*!@brief
  OP_LOADI
  Load 16-bit integer

  R(A) := sBx
*/
__UCODE__
uc_loadi(guru_vm *vm)
{
    GI sbx = vm->bx - MAX_sBx;

    _RA_T(GT_INT, i=sbx);
}


//================================================================
/*!@brief
  OP_LOADSYM

  R(A) := Syms(Bx)
*/
__UCODE__
uc_loadsym(guru_vm *vm)
{
	GS sid = VM_SYM(vm, vm->bx);

	_RA_T(GT_SYM, i=(GI)sid);
}

//================================================================
/*!@brief
  OP_LOADNIL

  R(A) := nil
*/
__UCODE__
uc_loadnil(guru_vm *vm)
{
    _RA_T(GT_NIL, i=0);
}

//================================================================
/*!@brief
  OP_LOADSELF

  R(A) := self
*/
__UCODE__
uc_loadself(guru_vm *vm)
{
    _RA_X(_R0);	              		// [ra] <= class
}

//================================================================
/*!@brief
  OP_LOADT

  R(A) := true
*/
__UCODE__
uc_loadt(guru_vm *vm)
{
    _RA_T(GT_TRUE, i=0);
}

//================================================================
/*!@brief
  OP_LOADF

  R(A) := false
*/
__UCODE__
uc_loadf(guru_vm *vm)
{
    _RA_T(GT_FALSE, i=0);
}

//================================================================
/*!@brief
  OP_GETGLOBAL

  R(A) := getglobal(Syms(Bx))
*/
__UCODE__
uc_getglobal(guru_vm *vm)
{
    GS sid = VM_SYM(vm, vm->bx);

    GR *r = global_get(sid);

    _RA(*r);
}

//================================================================
/*!@brief
  OP_SETGLOBAL

  setglobal(Syms(Bx), R(A))
*/
__UCODE__
uc_setglobal(guru_vm *vm)
{
    GS sid = VM_SYM(vm, vm->bx);

    global_set(sid, _R(a));
}

//================================================================
/*!@brief
  OP_GETIV

  R(A) := ivget(Syms(Bx))
*/
__GURU__ __INLINE__ GS
_sid_wo_at_sign(guru_vm *vm)
{
	GS sid     = VM_SYM(vm, vm->bx);
	char *name = (char*)_RAW(sid);

	return guru_rom_add_sym(name+1);	// skip leading '@'
}

__UCODE__
uc_getiv(guru_vm *vm)
{
	GR *r  = _R0;
	ASSERT(r->gt==GT_OBJ || r->gt==GT_CLASS);

    GS sid = _sid_wo_at_sign(vm);
    GR ret = ostore_get(r, sid);

    _RA(ret);
}

//================================================================
/*!@brief
  OP_SETIV	(set instance variable)

  ivset(Syms(Bx),R(A))
*/
__UCODE__
uc_setiv(guru_vm *vm)
{
	GR *r  = _R0;
	ASSERT(r->gt==GT_OBJ || r->gt==GT_CLASS);

	GS sid = _sid_wo_at_sign(vm);
	GR *ra = _R(a);

	guru_pack(ra);							// compact to save space
    ostore_set(r, sid, ra);					// store instance variable
}

//================================================================
/*!@brief
  OP_GETCV

  R(A) := cvget(Syms(Bx))
*/
__UCODE__
uc_getcv(guru_vm *vm)
{
	GR *r  = _R0;
	GS sid = VM_SYM(vm, vm->bx);
	GR ret = ostore_getcv(r, sid);

	_RA(ret);
}

//================================================================
/*!@brief
  OP_SETCV

  cvset(Syms(Bx),R(A))
*/
__UCODE__
uc_setcv(guru_vm *vm)
{
	GR *r = _R0;
	ASSERT(r->gt==GT_CLASS);

    GS sid = VM_SYM(vm, vm->bx);
    ostore_set(r, sid, _R(a));
}

//================================================================
/*!@brief
  OP_GETCONST

  R(A) := constget(Syms(Bx))
*/
__UCODE__
uc_getconst(guru_vm *vm)
{
    GS sid = VM_SYM(vm, vm->bx);				// In Ruby, class is a constant, too

    GP cls = class_by_id(sid);					// search class rom first
    GR ret { GT_CLASS, 0, 0, { cls } };
    if (cls) {
    	_RA(ret);
    	return;									// return ROM class
    }
    // search into constant cache, recursively up class hierarchy if needed
    ret.gt = GT_NIL;
	GR *r0 = _R0;
	cls    = (r0->gt==GT_CLASS) ? r0->off : class_by_obj(r0);
    while (cls) {
    	ret = *const_get(cls, sid);
        if (ret.gt!=GT_NIL) break;
    	cls = _CLS(cls)->super;
   }
    _RA(ret);
}

//================================================================
/*!@brief
  OP_SETCONST

  constset(Syms(Bx),R(A))
*/
__UCODE__
uc_setconst(guru_vm *vm)
{
	GS sid = VM_SYM(vm, vm->bx);
	GR *ra = _R(a);
	GR *r0 = _R0;
	GP cls = (r0->gt==GT_CLASS) ? r0->off : class_by_obj(r0);

	ra->acl &= ~ACL_HAS_REF;				// set it to constant

    const_set(cls, sid, ra);
}


//================================================================
/*!@brief
  get outer scope register file

*/
__GURU__ GR *
_upvar(guru_vm *vm)
{
	guru_state *st = VM_STATE(vm);
	for (int i=0; i<=vm->c; i++) {						// walk up stack frame
		st = IN_LAMBDA(st)
			? _STATE(st->prev)
			: _STATE(_STATE(st->prev)->prev);			// 1 extra for each_loop
	}
	return _REGS(st) + vm->b;
}

//================================================================
/*!@brief
  OP_GETUPVAR

  R(A) := uvget(B,C)
*/
__UCODE__
uc_getupvar(guru_vm *vm)
{
    GR *ur = _upvar(vm);			// outer scope register file
    _RA_X(ur);          			// ra <= up[rb]
}

//================================================================
/*!@brief
  OP_SETUPVAR

  uvset(B,C,R(A))
*/
__UCODE__
uc_setupvar(guru_vm *vm)
{
    GR *ur = _upvar(vm);			// pointer to caller's register file
    GR *ra = _R(a);

	ref_dec(ur);
    ref_inc(ra);
    *ur = *ra;                   	// update outer-scope vars
}

//================================================================
/*!@brief
  OP_JMP

  pc += sBx
*/
__UCODE__
uc_jmp(guru_vm *vm)
{
	GI sbx = vm->bx - MAX_sBx -1;

	VM_STATE(vm)->pc += sbx;
}

//================================================================
/*!@brief
  OP_JMPIF

  if R(A) pc += sBx
*/
__UCODE__
uc_jmpif (guru_vm *vm)
{
	GI sbx = vm->bx - MAX_sBx - 1;
	GR *ra = _R(a);

	if (ra->gt > GT_FALSE) {
		VM_STATE(vm)->pc += sbx;
	}
	*ra = EMPTY;
}

//================================================================
/*!@brief
  OP_JMPNOT

  if not R(A) pc += sBx
*/
__UCODE__
uc_jmpnot(guru_vm *vm)
{
	GI sbx = vm->bx - MAX_sBx -1;
	GR *ra = _R(a);
	if (ra->gt <= GT_FALSE) {
		VM_STATE(vm)->pc += sbx;
	}
	*ra = EMPTY;
}

//================================================================
/*!@brief
  OP_ONERR

  rescue_push(pc+sBx)
*/
__UCODE__
uc_onerr(guru_vm *vm)
{
	ASSERT(vm->xcp < (VM_RESCUE_STACK-1));

	GI sbx = vm->bx - MAX_sBx -1;

	RESCUE_PUSH(vm, VM_STATE(vm)->pc + sbx);
}

//================================================================
/*!@brief
  OP_RESCUE

  if (A)
    if (C) R(A) := R(A+1)		get exception
    else   R(A) := R(A+1)		set exception
*/
__UCODE__
uc_rescue(guru_vm *vm)
{
	U32 c  = vm->c;				// exception 0:set, 1:get
	GR  *r = _R(a);					// object to receive the exception
	GR  *x = r + 1;					// exception on stack

	if (c) {						// 2nd: get cycle
		if (x->gt != GT_NIL) {		// if exception is not given
			_RA_X(x);				// override exception (msg) if not given
		}
		x->gt  = GT_TRUE;			// here: modifying return stack directly is questionable!!
		x->acl = 0;
	}
	else {							// 1st: set cycle
		if (r->gt==GT_CLASS) x++;
		_RA_X(x);					// keep exception in RA
		*(x) = EMPTY;
	}
}

//================================================================
/*!@brief
  OP_POPERR

  A.times{rescue_pop()}
*/
__UCODE__
uc_poperr(guru_vm *vm)
{
	U32 a = vm->a;

	ASSERT(vm->xcp >= a);

	vm->xcp -= a;
}

//================================================================
/*!@brief
  OP_RAISE

  raise(R(A))
*/
__UCODE__
uc_raise(guru_vm *vm)
{
	GR *ra = _R(a);

	_RA(*ra);
}

//================================================================
/*!@brief
  _undef

  create undefined method error message (different between mruby1.4 and ruby2.x
*/
__GURU__ GR *
_undef(GR *buf, GR *r, GS pid)
{
	guru_buf_add_cstr(buf, "undefined method '");
	guru_buf_add_cstr(buf, _RAW(pid));
	guru_buf_add_cstr(buf, "' for class #");
	guru_buf_add_cstr(buf, _RAW(_CLS(class_by_obj(r))->cid));

	return buf;
}

//================================================================
/*!@brief
  OP_SEND / OP_SENDB

  OP_SEND   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
  OP_SENDB  R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))
*/
__UCODE__
uc_send(guru_vm *vm)
{
    GS  xid = VM_SYM(vm, vm->b);					// get given symbol id or object id
    GR  *r  = _R(a);								// call stack, obj is receiver object
    S32 argc= (vm->c & 0x40) ? -(0x80-vm->c) : vm->c;
#if CC_DEBUG
    PRINTF("!!!uc_send(%p) R(%d)=%p, xid=%d\n", vm, vm->a, r, xid);
#endif // CC_DEBUG
    if (vm_method_exec(vm, r, argc, xid)) { 		// in state.cu, call stack will be wiped before return
    	vm->err = 2;								// raise Method Not Found exception
    	GR buf  = guru_str_buf(GURU_STRBUF_SIZE);	// put error message on return stack
    	*(r+1)  = *_undef(&buf, r, xid);			// TODO: exception class
    }
}

//================================================================
/*!@brief
  OP_CALL
  R(A) := self.call(frame.argc, frame.argv)

  TODO: no test case yet
*/
__UCODE__
uc_call(guru_vm *vm)
{
	ASSERT(1==0);				// should not be here, no test case yet!
}

//================================================================
/*!@brief
  OP_RETURN

  return R(A) (B=normal,in-block return/break)
*/
__UCODE__
uc_return(guru_vm *vm)
{
	guru_state *st = VM_STATE(vm);				// get current context

	GR  *ra = _R(a);
	GR  *ma = _R0 - 2;							// mapper array
	U32 map = IS_COLLECT(st);					// is a mapper?
	U32 brk = vm->b;							// break
	GR  ret = *ra;								// return value

	if (IN_LOOP(st)) {
		if (map) {
			guru_array_push(ma, ra);			// collect return value
		}
		if (vm_loop_next(vm) && !brk) return;	// continue

		guru_iter_del(_REGS(st) - 1);			// release iterator

		// pop off iterator state
		vm_state_pop(vm, *ra);					// pop off ITERATOR state and transfer last returned value
		ret = *_R0;								// capture return value from inner loop
	}
	else if (IN_LAMBDA(st)) {
		vm_state_pop(vm, *ra);					// pop off LAMBDA state, transfer current stack top value
	}
	else if (IS_NEW(st)) {
		ret = *_R0;								// return the object itself
	}
	ret.acl &= ~(ACL_SELF|ACL_SCLASS);			// turn off TCLASS and NEW flags if any
	vm_state_pop(vm, map ? EMPTY : ret);		// pop callee's context

	if (map) {									// put return array back to caller stack
		ref_dec(ma-1);							// TODO: this is a hack, needs a better way
		*(ma-1) = *ma;
		*ma     = EMPTY;
	}
}

//================================================================
/*!@brief
  OP_ENTER

  arg setup according to flags (23=5:5:1:5:5:1:1)		// default parameter

*/
#define BAIL(msg, ec)	\
	uc_return(vm);		\
	vm->err = ec;		\
	return

#define SPLAT(r1, r0, n) 	\
	guru_splat(r1, r0, n);	\
	r1 += n;				\
	r0 += n

__GURU__ GR*
_enter(AX *ax, GR *r, guru_state *st, guru_array *h)
{
    S32 n    = st->argc;									// mruby.argc
	U32 m12  = ax->req + ax->pst;							// mruby.m1, m2
	U32 k    = 0;											// TODO: mruby.kargs, for keyword argument
	S32 f1   = n - ax->req;									// filler after REST
	U32 mlen = n < m12										// post argument count
			? (f1 > 0 ? f1 : 0)
			: ax->pst;
	S32 mx   = n - mlen;									// mandatory argument count
	S32 f2   = ax->pst - mlen;								// filter after POST

	if (h) {
		GR  *p = h->data;
    	SPLAT(r, p, mx);									// copy mandatory arguments onto call stack
    	for (int i=0; i<f1; i++, *r++=EMPTY);				// TODO: extra space allocated for REST???
    	SPLAT(r, p, mlen);									// optional arguments
    	for (int i=0; i<f2; i++, *r++=EMPTY);				// TODO: extra space allocated after POST???
	}
	else {
		r += m12 + ax->opt;									// push call stack for block (if any)
	}
    if (ax->opt) {											// see if any optional argument
    	st->pc += n - m12 -k;								// adjust pc for default value jump table
    }
	return r;
}

__GURU__ GR*
_enter_rest(AX *ax, GR *r, guru_state *st, guru_array *h)
{
    S32 n  = st->argc;										// mruby.argc
    S32 mx = ax->req + ax->opt;								// mandatory + optional arguments
    S32 rx = (n > mx) ? n - mx : 0;							// rest of arguments
    GR  *p = h ? h->data : (r += mx);

    if (h) {
    	SPLAT(r, p, mx);									// copy arguments from Array onto call stack
        ref_dec(r);											// stick rest of arguments go into an array if any
    }
    GR a = guru_array_new(rx);								// create the "rest" array
    for (int i=0; i<rx; i++, p++) {
    	guru_array_push(&a, p);
    	if (!h) {
    		ref_dec(p);
        	*p = EMPTY;
    	}
    }
    *r++ = a;												// place "*rest" array on the call stack
    /* TODO: not sure what this does, see mruby::vm.c
    if (ax->rst && (n > m12)) {								// not sure what this does
    	S32 rnum = n - m12 - karg;							// some left over arguments?
    	GR *p = _REGS(st) + 1 + mx;
    	for (int i=0; i<rnum; i++) {
    		ref_dec(r);
    		*r++ = *ref_inc(p++);
    	}
    }
    */
    st->pc += ax->opt + (n > mx ? 0 : n - mx);				// adjust for default value jump table

    return r;
}

__UCODE__
uc_enter(guru_vm *vm)
{
	U32 eax = vm->ax;
	AX  *ax = (AX*)&eax;									// a special decoder case
	U32 m12 = ax->req + ax->pst;							// required + post arguments, mruby.m1, m2
	U32 len = m12 + ax->rst + ax->opt;						// mruby.len
	U32 kd  = ax->key || ax->dic;							// mruby.kd

	guru_state *st = VM_STATE(vm);
    S32 n   = st->argc;										// mruby.argc
    GR  *r  = _REGS(st) + 1;								// mruby.argv
    GR  *b  = r + (n<0 ? 1 : n);							// mruby.blk: callback block (if any)

    guru_array *h = (n<0) ? GR_ARY(r+1) : NULL;				// arguments is passed as Array (by OP_ARYCAT)
    if (h) {
    	st->argc = n = h->n;
    }
    if (n < m12 || (!ax->rst && (n > (len+kd)))) {			// validate parameter count (rst: array_splat)
    	BAIL("parameter count mismatched", 5);
    }
	U32 k = 0;												// TODO: mruby.kargs, for keyword argument
    if (kd) {												// hash (keyword) as the last argument
    	BAIL("SEND with keyword argument", 6);				// not supported yet
    }
    r = ax->rst												// if REST, i.e. func(a,b,*c), is specified
    	? _enter_rest(ax, r, st, h)
    	: _enter(ax, r, st, h);
	if (ax->blk && (r != b)) {								// callback block exists
		ref_dec(r);
		*r = *b;
		*b = EMPTY;
	}
}

//================================================================
/*!@brief
  OP_BLKPUSH (yield implementation)

  R(A) := block (16=6:1:5:4)
*/
__UCODE__
uc_blkpush(guru_vm *vm)
{
	guru_state *st = VM_STATE(vm);
	for (int i=0; i<vm->c; i++) {
		st = _STATE(_STATE(st->prev)->prev);
	}
    GR *prc = _REGS(st)+st->argc+1;       	// get proc, regs[0] is the class

    ASSERT(prc->gt==GT_PROC);				// ensure

    _RA_X(prc);             				// ra <= proc
}

//================================================================
/*!@brief
  OP_ADDI

  R(A) := R(A)+C (Syms[B]=:+)
*/
__UCODE__
uc_addi(guru_vm *vm)
{
	GR *r0 = _R(a);
	U32 n  = vm->c;

    if (r0->gt==GT_INT)     	r0->i += n;
#if GURU_USE_FLOAT
    else if (r0->gt==GT_FLOAT)	r0->f += n;
#else
    else QUIT("Float class");
#endif // GURU_USE_FLOAT
}

//================================================================
/*!@brief
  OP_SUBI

  R(A) := R(A)-C (Syms[B]=:-)
*/
__UCODE__
uc_subi(guru_vm *vm)
{
	GR  *r0 = _R(a);
	U32 n   = vm->c;

    if (r0->gt==GT_INT) 		r0->i -= n;
#if GURU_USE_FLOAT
    else if (r0->gt==GT_FLOAT) 	r0->f -= n;
#else
    else QUIT("Float class");
#endif // GURU_USE_FLOAT
}

//
// arithmetic template (poorman's C++)
//
#define ALU_OP(a, OP) ({				\
	GR *r0 = _R(a);						\
	GR *r1 = r0+1;						\
	if (r0->gt==GT_INT) {				\
		if      (r1->gt==GT_INT)   { 	\
			r0->i = r0->i OP r1->i; 	\
		}								\
		else if (r1->gt==GT_FLOAT) {	\
			r0->gt = GT_FLOAT;			\
			r0->f  = r0->i OP r1->f;	\
		}								\
		else SKIP("Fixnum + ?");		\
    }									\
    else if (r0->gt==GT_FLOAT) {		\
    	if      (r1->gt==GT_INT) 	{	\
    		r0->f = r0->f OP r1->i;		\
    	}								\
    	else if (r1->gt==GT_FLOAT)	{	\
    		r0->f = r0->f OP r1->f;		\
    	}								\
    	else SKIP("Float + ?");			\
	}									\
	else {	/* other cases */			\
		uc_send(vm);					\
	}									\
	*r1 = EMPTY;						\
})

//================================================================
/*!@brief
  OP_ADD

  R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)
*/
__UCODE__
uc_add(guru_vm *vm)
{
	ALU_OP(a, +);
}

//================================================================
/*!@brief
  OP_SUB

  R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)
*/
__UCODE__
uc_sub(guru_vm *vm)
{
	ALU_OP(a, -);
}

//================================================================
/*!@brief
  OP_MUL

  R(A) := R(A)*R(A+1) (Syms[B]=:*)
*/
__UCODE__
uc_mul(guru_vm *vm)
{
	ALU_OP(a, *);
}

//================================================================
/*!@brief
  OP_DIV

  R(A) := R(A)/R(A+1) (Syms[B]=:/)
*/
__UCODE__
uc_div(guru_vm *vm)
{
	GR *r = _R(a);

	if (r->gt==GT_INT && (r+1)->i==0) {
		vm->err = 4;
	}
	else ALU_OP(a, /);
}

//================================================================
/*!@brief
  OP_EQ

  R(A) := R(A)==R(A+1)  (Syms[B]=:==,C=1)
*/
__UCODE__
uc_eq(guru_vm *vm)
{
	GR *r0 = _R(a), *r1 = r0+1;
    GT tt = GT_BOOL(guru_cmp(r0, r1)==0);

    *r1 = EMPTY;
    _RA_T(tt, i=0);
}

// comparator template (poorman's C++)
#define ALU_CMP(a, OP)	({								\
	GR *r0 = _R(a);										\
	GR *r1 = r0+1;										\
	if ((r0)->gt==GT_INT) {								\
		if ((r1)->gt==GT_INT) {							\
			(r0)->gt = GT_BOOL((r0)->i OP (r1)->i);		\
		}												\
		else if ((r1)->gt==GT_FLOAT) {					\
			(r0)->gt = GT_BOOL((r0)->i OP (r1)->f);		\
		}												\
	}													\
	else if ((r0)->gt==GT_FLOAT) {						\
		if ((r1)->gt==GT_INT) {							\
			(r0)->gt = GT_BOOL((r0)->f OP (r1)->i);		\
		}												\
		else if ((r1)->gt==GT_FLOAT) {					\
			(r0)->gt = GT_BOOL((r0)->f OP (r1)->f);		\
		}												\
	}													\
	else {												\
		uc_send(vm);									\
	}													\
    *r1 = EMPTY;	  									\
})

//================================================================
/*!@brief
  OP_LT

  R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)
*/
__UCODE__
uc_lt(guru_vm *vm)
{
	ALU_CMP(a, <);
}

//================================================================
/*!@brief
  OP_LE

  R(A) := R(A)<=R(A+1)  (Syms[B]=:<=,C=1)
*/
__UCODE__
uc_le(guru_vm *vm)
{
	ALU_CMP(a, <=);
}

//================================================================
/*!@brief
  OP_GT

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
*/
__UCODE__
uc_gt(guru_vm *vm)
{
	ALU_CMP(a, >);
}

//================================================================
/*!@brief
  OP_GE

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
*/
__UCODE__
uc_ge(guru_vm *vm)
{
	ALU_CMP(a, >=);
}

//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))
*/
__UCODE__
uc_string(guru_vm *vm)
{
    GR v = *VM_STR(vm, vm->bx);
    _RA(v);
}

//================================================================
/*!@brief
  String Catination

  str_cat(R(A),R(B))
*/
__UCODE__
uc_strcat(guru_vm *vm)
{
	GR *s0 = _R(a), *rb = _R(b), *s1 = rb;

	ASSERT(s0->gt==GT_STR);

	switch (s1->gt) {
	case GT_STR: /* do nothing */	break;
	case GT_SYM: sym_to_s(s1, 0); 	break;
	default:     gr_to_s(s1, 0);
	}
	GR buf = guru_str_add(s0, s1);	// ref_cnt is set

    ref_dec(s1);
    *rb = EMPTY;

    _RA(buf);						// this will clean out sa
}

__GURU__ void
_stack_copy(GR *d, GR *s, U32 n)
{
	for (int i=0; i < n; i++, d++, s++) {
		*d = *ref_inc(s);			// now referenced by array/hash
		*s = EMPTY;					// DEBUG: clean element from call stack
	}
}

#if GURU_USE_ARRAY
//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))
*/
__UCODE__
uc_array(guru_vm *vm)
{
    U32 n   = vm->c;
    GR  ret = (GR)guru_array_new(n);		// ref_cnt is 1 already

    guru_array *h = GR_ARY(&ret);
    if ((h->n=n)>0) _stack_copy(h->data, _R(b), n);

    _RA(ret);								// no need to ref_inc
}

//================================================================
/*!@brief
 * Array tabulate

 R(A) := ary_cat(R(A),R(B))
 */
__UCODE__
uc_arycat(guru_vm *vm)
{
	GR *r1 = _R(a) + 1;
	ref_dec(r1);
	*r1 = *ref_inc(_R(b));
}

//================================================================
/*!@brief
 * Array concat

 R(A) := ary_push(R(A),R(B))
 */
__UCODE__
uc_arypush(guru_vm *vm)
{
	guru_array_push(_R(a), _R(b));
}

//================================================================
/*!@brief
 * get Array element by index

 R(A) := R(B)[C]
 */
__UCODE__
uc_aref(guru_vm *vm)
{
	GR ret = guru_array_get(_R(b), vm->c);

	_RA(ret);
}

//================================================================
/*!@brief
 * set Array element by index

  R(B)[C] := R(A)
  */
__UCODE__
uc_aset(guru_vm *vm)
{
	guru_array_set(_R(b), vm->c, _R(a));
}

//================================================================
/*!@brief
 * update multiple Array elements

  *R(A),R(A+1)..R(A+C) := R(A)[B..]
  */
__UCODE__
uc_apost(guru_vm *vm)
{
    ASSERT(1==0);
}
//================================================================
/*!@brief
  Create Hash object

  R(A) := hash_new(R(B),R(B+1)..R(B+C))
*/
__UCODE__
uc_hash(guru_vm *vm)
{
	U32 n   = vm->c;						// number of kv pairs
    GR  ret = guru_hash_new(n);				// ref_cnt is already set to 1

    guru_hash *h = GR_HSH(&ret);
    if ((h->n=(n<<1))>0) _stack_copy(h->data, _R(b), h->n);

    _RA(ret);							    // new hash on stack top
}

//================================================================
/*!@brief
  OP_RANGE

  R(A) := range_new(R(B),R(B+1),C)
*/
__UCODE__
uc_range(guru_vm *vm)
{
	U32 x   = vm->c;						// exclude_end
	GR  *p0 = _R(b), *p1 = p0+1;
    GR  v   = guru_range_new(p0, p1, !x);	// p0, p1 ref cnt will be increased
    *p1 = EMPTY;

    _RA(v);									// release and  reassign
}
#else
__UCODE__ 	uc_array(guru_vm *vm)	{ QUIT("Array class"); }
__UCODE__	uc_arypush(guru_vm *vm)	{}
__UCODE__	uc_aref(guru_vm *vm)	{}
__UCODE__	uc_aset(guru_vm *vm)	{}
__UCODE__	uc_apost(guru_vm *vm)	{}
__UCODE__	uc_hash(guru_vm *vm)	{ QUIT("Hash class"); }
__UCODE__	uc_range(guru_vm *vm) 	{ QUIT("Range class"); }
#endif // GURU_USE_ARRAY

//================================================================
/*!@brief
  OP_LAMBDA

  R(A) := lambda(SEQ[Bz],Cz)
*/
__UCODE__
uc_lambda(guru_vm *vm)
{
	GR *obj = _R(a) - 1;
	GP cls  = class_by_obj(obj);						// current class
	GP irep = MEMOFF(VM_REPS(vm, vm->bz));				// fetch from children irep list

	GP prc  = guru_define_method(cls, NULL, irep);

	guru_proc *px = _PRC(prc);
    px->kt = PROC_IREP;									// instead of C-function
    px->n  = (obj->gt==GT_HASH) ? vm->cz : vm->cz>>1;	// TODO: not sure how Cz works,  assume this is parameter count

    _RA_T(GT_PROC, off=prc);							// regs[ra].prc = prc
}

//================================================================
/*!@brief
  OP_CLASS, OP_MODULE

  R(A) := newclass(R(A),Syms(B),R(A+1))
  Syms(B): class name
  R(A+1) : super class
*/
__UCODE__
uc_class(guru_vm *vm)
{
	GR *r1 = _R(a)+1;

    GS sid   = VM_SYM(vm, vm->b);
    U8 *name = _RAW(sid);
    GP super = (r1->gt==GT_CLASS) ? r1->off : VM_STATE(vm)->klass;
    GP cls   = guru_define_class(name, super);

	_CLS(cls)->kt |= USER_DEF_CLASS;			// user defined (i.e. non-builtin) class

    _RA_T(GT_CLASS, off=cls);

	*r1 = EMPTY;
}

//================================================================
/*!@brief
  OP_EXEC

  R(A) := blockexec(R(A),SEQ[Bx])
*/
__UCODE__
uc_exec(guru_vm *vm)
{
	ASSERT(_R0->gt == GT_CLASS);				// check
	GP irep = MEMOFF(VM_REPS(vm, vm->bx));		// child IREP[rb]

    vm_state_push(vm, irep, 0, _R(a), 0);		// push call stack
}

//================================================================
/*!@brief
  OP_METHOD

  R(A).newmethod(Syms(B),R(A+1))
*/
__UCODE__
uc_method(guru_vm *vm)
{
	GR *r  = _R(a);
    ASSERT(r->gt==GT_OBJ || r->gt == GT_CLASS);	// enforce class checking

    // check whether the name has been defined in current class (i.e. vm->state->klass)
    GS pid = VM_SYM(vm, vm->b);				// fetch name from IREP symbol table

    guru_proc *px = GR_PRC(r+1);				// override (if exist) with proc by OP_LAMBDA
    _LOCK;
    px->pid = pid;								// assign sid to proc, overload if prc already exists
    _UNLOCK;

#if CC_DEBUG
    PRINTF("!!!uc_method %s:%p->%d\n", _RAW(px->pid), px, px->pid);
#endif // CC_DEBUG
    r->acl &= ~ACL_SELF;						// clear CLASS modification flags if any
    *(r+1) = EMPTY;								// clean up proc
}

//================================================================
/*!@brief
  OP_TCLASS
->self
  R(A) := target_class
*/
__UCODE__
uc_tclass(guru_vm *vm)
{
	GR *ra = _R(a);

	_RA_T(GT_CLASS, off=VM_STATE(vm)->klass);
	ra->acl |= ACL_SELF;
	ra->acl &= ~ACL_SCLASS;
}

//================================================================
/*!@brief
  OP_SCLASS

  R(A) := R(B).singleton_class
*/
__UCODE__
uc_sclass(guru_vm *vm)
{
	GR *r = _R(b);
	if (r->gt==GT_OBJ) {							// singleton class (extending an object)
		U8 *name = (U8*)"_single";
		GP super = class_by_obj(r);
		GP cls   = guru_define_class(name, super);
		GR_OBJ(r)->cls = cls;
	}
	else if (r->gt==GT_CLASS) {						// meta class (for class methods)
		guru_class_add_meta(r);						// lazily add metaclass if needed
	}
	else ASSERT(1==0);

	r->acl |= ACL_SCLASS;
	r->acl &= ~ACL_SELF;
}

//================================================================
/*!@brief
  OP_STOP and OP_ABORT

  stop VM (OP_STOP)
  stop VM without release memory (OP_HOLD)
*/
__UCODE__
uc_stop(guru_vm *vm)
{
	vm->run  = VM_STATUS_STOP;					// VM suspended
}

//================================================================
/*!@brief
  little endian to big endian converter
*/
__GURU__ __INLINE__ U32
_bin_to_u32(const void *s)
{
    U32 x = *((U32*)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
}

#define VM_BYTECODE(vm) (_bin_to_u32(U8PADD(VM_ISEQ(vm), sizeof(U32)*VM_STATE(vm)->pc)))

//===========================================================================================
// GURU engine
//===========================================================================================
/*!@brief
  GURU Instruction Unit - Prefetcher (fetch bytecode and decode)

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GURU__ void
ucode_prefetch(guru_vm *vm)
{
	U32 bcode  = VM_BYTECODE(vm);						// fetch from vm->state->pc
	vm->rbcode = ((bcode & 0x7f)<<25) | (bcode>>7);		// rotate bytecode (for nvcc bit-field limitation)

	guru_state *st = VM_STATE(vm);
	st->pc++;											// advance program counter (ready for next fetch)
}


__GURU__ void
ucode_step(guru_vm *vm)
{
	//=======================================================================================
	// GURU dispatcher unit
	// TODO: dispatch vtable[op](vm) without switch branching
	//=======================================================================================
#if !GURU_DEBUG
    switch (vm->op) {
// LOAD,STORE
    case OP_LOADL:      uc_loadl     (vm); break;
    case OP_LOADI:      uc_loadi     (vm); break;
    case OP_LOADSYM:    uc_loadsym   (vm); break;
    case OP_LOADNIL:    uc_loadnil   (vm); break;
    case OP_LOADSELF:   uc_loadself  (vm); break;
    case OP_LOADT:      uc_loadt     (vm); break;
    case OP_LOADF:      uc_loadf     (vm); break;
// VARIABLES
    case OP_GETGLOBAL:  uc_getglobal (vm); break;
    case OP_SETGLOBAL:  uc_setglobal (vm); break;
    case OP_GETIV:      uc_getiv     (vm); break;
    case OP_SETIV:      uc_setiv     (vm); break;
    case OP_GETCV:		uc_getcv	 (vm); break;
    case OP_SETCV:	    uc_setcv     (vm); break;
    case OP_GETCONST:   uc_getconst  (vm); break;
    case OP_SETCONST:   uc_setconst  (vm); break;
    case OP_GETUPVAR:   uc_getupvar  (vm); break;
    case OP_SETUPVAR:   uc_setupvar  (vm); break;
// BRANCH
    case OP_JMP:        uc_jmp       (vm); break;
    case OP_JMPIF:      uc_jmpif     (vm); break;
    case OP_JMPNOT:     uc_jmpnot    (vm); break;
// EXCEPTION
    case OP_ONERR:		uc_onerr     (vm); break;
    case OP_RESCUE:		uc_rescue    (vm); break;
    case OP_POPERR:		uc_poperr	 (vm); break;
    case OP_RAISE:		uc_raise	 (vm); break;
// CALL
    case OP_SEND:
    case OP_SENDB:      uc_send      (vm); break;
    case OP_CALL:       uc_call      (vm); break;
    case OP_ENTER:      uc_enter     (vm); break;
    case OP_RETURN:     uc_return    (vm); break;
    case OP_BLKPUSH:    uc_blkpush   (vm); break;
// ALU
    case OP_MOVE:       uc_move      (vm); break;
    case OP_ADD:        uc_add       (vm); break;
    case OP_ADDI:       uc_addi      (vm); break;
    case OP_SUB:        uc_sub       (vm); break;
    case OP_SUBI:       uc_subi      (vm); break;
    case OP_MUL:        uc_mul       (vm); break;
    case OP_DIV:        uc_div       (vm); break;
    case OP_EQ:         uc_eq        (vm); break;
    case OP_LT:         uc_lt        (vm); break;
    case OP_LE:         uc_le        (vm); break;
    case OP_GT:         uc_gt        (vm); break;
    case OP_GE:         uc_ge        (vm); break;
// BUILT-IN class (TODO: tensor)
    case OP_ARRAY:      uc_array     (vm); break;
    case OP_ARYCAT:
    case OP_ARYPUSH:	uc_arypush 	 (vm); break;
    case OP_AREF:		uc_aref		 (vm); break;
    case OP_ASET:		uc_aset		 (vm); break;
    case OP_APOST:		uc_apost	 (vm); break;
    case OP_STRING:     uc_string    (vm); break;
    case OP_STRCAT:     uc_strcat    (vm); break;
    case OP_HASH:       uc_hash      (vm); break;
    case OP_RANGE:      uc_range     (vm); break;
// CLASS, PROC (STACK ops)
    case OP_LAMBDA:     uc_lambda    (vm); break;
    case OP_CLASS:
    case OP_MODULE:	    uc_class	 (vm); break;
    case OP_EXEC:       uc_exec      (vm); break;
    case OP_METHOD:     uc_method    (vm); break;
    case OP_SCLASS:	    uc_sclass    (vm); break;
    case OP_TCLASS:     uc_tclass    (vm); break;
// CONTROL
    case OP_STOP:       uc_stop      (vm); break;
    case OP_NOP:        uc_nop       (vm); break;
    default:
    	PRINTF("?OP=0x%04x\n", vm->op);
    	vm->err = 1;
    	break;
    }
#else
	static const UCODE ucode_vtbl[] = {
			NULL, 			// 	  OP_NOP = 0,
	// 0x1 Register File
			uc_move,		//    OP_MOVE       A B     R(A) := R(B)
			uc_loadl,		//    OP_LOADL      A Bx    R(A) := Pool(Bx)
			uc_loadi,		//    OP_LOADI      A sBx   R(A) := sBx
			uc_loadsym,		//    OP_LOADSYM    A Bx    R(A) := Syms(Bx)
			uc_loadnil,		//    OP_LOADNIL    A       R(A) := nil
			uc_loadself,	//    OP_LOADSELF   A       R(A) := self
			uc_loadt,		//    OP_LOADT      A       R(A) := true
			uc_loadf,		//    OP_LOADF      A       R(A) := false
	// 0x9 Load/Store
			uc_getglobal,	//    OP_GETGLOBAL  A Bx    R(A) := getglobal(Syms(Bx))
			uc_setglobal,	//    OP_SETGLOBAL  A Bx    setglobal(Syms(Bx), R(A))
			NULL,			//    OP_GETSPECIAL A Bx    R(A) := Special[Bx]
			NULL,			//    OP_SETSPECIAL	A Bx    Special[Bx] := R(A)
			uc_getiv,		//    OP_GETIV      A Bx    R(A) := ivget(Syms(Bx))
			uc_setiv,		//    OP_SETIV      A Bx    ivset(Syms(Bx),R(A))
			uc_getcv,		//    OP_GETCV      A Bx    R(A) := cvget(Syms(Bx))
			uc_setcv,		//    OP_SETCV      A Bx    cvset(Syms(Bx),R(A))
			uc_getconst,	//    OP_GETCONST   A Bx    R(A) := constget(Syms(Bx))
			uc_setconst,	//    OP_SETCONST   A Bx    constset(Syms(Bx),R(A))
			NULL,			//    OP_GETMCNST   A Bx    R(A) := R(A)::Syms(Bx)
			NULL,			//    OP_SETMCNST   A Bx    R(A+1)::Syms(Bx) := R(A)
			uc_getupvar,	//    OP_GETUPVAR   A B C   R(A) := uvget(B,C)
			uc_setupvar,	//    OP_SETUPVAR   A B C   uvset(B,C,R(A))
	// 0x17 Branch Unit
			uc_jmp,			//    OP_JMP,       sBx     pc+=sBx
			uc_jmpif,		//    OP_JMPIF,     A sBx   if R(A) pc+=sBx
			uc_jmpnot,		//    OP_JMPNOT,    A sBx   if !R(A) pc+=sBx
	// 0x1a Exception Handler
			uc_onerr,		//    OP_ONERR,     sBx     rescue_push(pc+sBx)
			uc_rescue,		//    OP_RESCUE		A B C   if A (if C exc=R(A) else R(A) := exc);
			uc_poperr,		// 	  OP_POPERR,    A       A.times{rescue_pop()}
			uc_raise,		//    OP_RAISE,     A       raise(R(A))
			NULL,			//    OP_EPUSH,     Bx      ensure_push(SEQ[Bx])
			NULL,			//    OP_EPOP,      A       A.times{ensure_pop().call}
	// 0x20 Stack
			uc_send,		//    OP_SEND,      A B C   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
			uc_send,		//    OP_SENDB,     A B C   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))
			NULL,			//    OP_FSEND,     A B C   R(A) := fcall(R(A),Syms(B),R(A+1),...,R(A+C-1))
			uc_call,		//    OP_CALL,      A       R(A) := self.call(frame.argc, frame.argv)
			NULL,			//    OP_SUPER,     A C     R(A) := super(R(A+1),... ,R(A+C+1))
			NULL,			//    OP_ARGARY,    A Bx    R(A) := argument array (16=6:1:5:4)
			uc_enter,		//    OP_ENTER,     Ax      arg setup according to flags (23=5:5:1:5:5:1:1)
			NULL,			//    OP_KARG,      A B C   R(A) := kdict[Syms(B)]; if C kdict.rm(Syms(B))
			NULL,			//    OP_KDICT,     A C     R(A) := kdict
			uc_return,		//    OP_RETURN,    A B     return R(A) (B=normal,in-block return/break)
			NULL,			//    OP_TAILCALL,  A B C   return call(R(A),Syms(B),*R(C))
			uc_blkpush,		//    OP_BLKPUSH,   A Bx    R(A) := block (16=6:1:5:4)
	// 0x2c ALU
			uc_add,			//    OP_ADD,       A B C   R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)
			uc_addi,		//    OP_ADDI,      A B C   R(A) := R(A)+C (Syms[B]=:+)
			uc_sub,			//    OP_SUB,       A B C   R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)
			uc_subi,		//    OP_SUBI,      A B C   R(A) := R(A)-C (Syms[B]=:-)
			uc_mul,			//    OP_MUL,       A B C   R(A) := R(A)*R(A+1) (Syms[B]=:*,C=1)
			uc_div,			//    OP_DIV,       A B C   R(A) := R(A)/R(A+1) (Syms[B]=:/,C=1)
			uc_eq,			//    OP_EQ,        A B C   R(A) := R(A)==R(A+1) (Syms[B]=:==,C=1)
			uc_lt,			//    OP_LT,        A B C   R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)
			uc_le,			//    OP_LE,        A B C   R(A) := R(A)<=R(A+1) (Syms[B]=:<=,C=1)
			uc_gt,			//    OP_GT,        A B C   R(A) := R(A)>R(A+1)  (Syms[B]=:>,C=1)
			uc_ge,			//    OP_GE,        A B C   R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
	// 0x37 Array Object
			uc_array,		//    OP_ARRAY,     A B C   R(A) := ary_new(R(B),R(B+1)..R(B+C))
			uc_arycat,		//    OP_ARYCAT,    A B     ary_cat(R(A),R(B))
			uc_arypush,		//    OP_ARYPUSH,   A B     ary_push(R(A),R(B))
			uc_aref,		//    OP_AREF,      A B C   R(A) := R(B)[C]
			uc_aset,		//    OP_ASET,      A B C   R(B)[C] := R(A)
			uc_apost,		//    OP_APOST,     A B C   *R(A),R(A+1)..R(A+C) := R(A)[B..]
	// 0x3d String Object
			uc_string,		//    OP_STRING,    A Bx    R(A) := str_dup(Lit(Bx))
			uc_strcat,		//    OP_STRCAT,    A B     str_cat(R(A),R(B))
			uc_hash,		//    OP_HASH,      A B C   R(A) := hash_new(R(B),R(B+1)..R(B+C))
			uc_lambda,		//    OP_LAMBDA,    A Bz Cz R(A) := lambda(SEQ[Bz],Cz)
			uc_range,		//    OP_RANGE,     A B C   R(A) := range_new(R(B),R(B+1),C)
	// 0x42 Class
			NULL,			//    OP_OCLASS,    A       R(A) := ::Object
			uc_class,		//    OP_CLASS,     A B     R(A) := newclass(R(A),Syms(B),R(A+1))
			uc_class,		//    OP_MODULE,    A B     R(A) := newmoducule(R(A),Syms(B))
			uc_exec,		//    OP_EXEC,      A Bx    R(A) := blockexec(R(A),SEQ[Bx])
			uc_method,		//    OP_METHOD,    A B     R(A).newmethod(Syms(B),R(A+1))
			uc_sclass,		//    OP_SCLASS,    A B     R(A) := R(B).singleton_class
			uc_tclass,		//    OP_TCLASS,    A       R(A) := target_class
			NULL,			//    OP_DEBUG,     A B C   print R(A),R(B),R(C)
	// 0x4a Exit
			uc_stop,		//    OP_STOP,      stop VM
			NULL			//    OP_ERR,       Bx      raise RuntimeError with message Lit(Bx)
	};
	GR *r = _R0;										// for debugging

	ucode_vtbl[vm->op](vm);
#endif // GURU_DEBUG

    if (vm->err && vm->xcp>0) {							// simple exception handler
    	VM_STATE(vm)->pc = RESCUE_POP(vm);				// bubbling up
    	vm->err = 0;									// TODO: add exception type or code on stack
    }
}

