/*! @file
  @brief
  GURU instruction unit - prefetch & microcode dispatcher

  <pre>
  Copyright (C) 2019 GreenII

  1. a list of opcode (microcode) executor, and
  2. the core opcode dispatcher
  </pre>
*/
#include <assert.h>

#include "mmu.h"
#include "static.h"
#include "symbol.h"
#include "global.h"
#include "value.h"
#include "inspect.h"

#include "ostore.h"
#include "class.h"
#include "state.h"
#include "ucode.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"
#include "iter.h"


#define _LOCK		{ MUTEX_LOCK(_mutex_uc); }
#define _UNLOCK		{ MUTEX_FREE(_mutex_uc); }

__GURU__ U32 _mutex_uc;
//
// becareful with the following macros, because they release regs[ra] first
// so, make sure value is kept before the release
//
#define _AR(r)          ((vm->ar.r))
#define _R0             (&(vm->state->regs[0]))
#define _R(r)			(&(vm->state->regs[_AR(r)]))
#define _RA(v)			(ref_dec(_R(a)), *_R(a)=(v))
#define _RA_X(r)    	(ref_inc(r), ref_dec(_R(a)), *_R(a)=*(r))
#define _RA_T(t,e)      (_R(a)->gt=(t), _R(a)->acl=0, _R(a)->e)

#define SKIP(x)			{ guru_na(x); return; }
#define RAISE(x)	    { _RA(guru_str_new(x)); vm->err = 1; return; }
#define QUIT(x)			{ vm->quit=1; guru_na(x); return; }

//================================================================
/*!@brief
  little endian to big endian converter
*/
__GURU__ __INLINE__ U32
_bin_to_u32(const void *s)
{
#if GURU_32BIT_ALIGN_REQUIRED
    U8 *p = (U8*)s;
    return (U32)(p[0]<<24) | (p[1]<<16) |  (p[2]<<8) | p[3];
#else
    U32 x = *((U32P)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
#endif
}

__GURU__ __INLINE__ GV *nop()	{ return NULL; }

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
	GV ret = VM_VAR(vm, _AR(bx));
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
    GI sbx = _AR(bx) - MAX_sBx;

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
	GS sid = VM_SYM(vm, _AR(bx));

	_RA_T(GT_SYM, i=sid);
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
    GS sid = VM_SYM(vm, _AR(bx));

    GV *v = global_get(sid);

    _RA(*v);
}

//================================================================
/*!@brief
  OP_SETGLOBAL

  setglobal(Syms(Bx), R(A))
*/
__UCODE__
uc_setglobal(guru_vm *vm)
{
    GS sid = VM_SYM(vm, _AR(bx));

    global_set(sid, _R(a));
}

//================================================================
/*!@brief
  OP_GETIV

  R(A) := ivget(Syms(Bx))
*/
__UCODE__
uc_getiv(guru_vm *vm)
{
	GV *v = _R0;
	assert(v->gt==GT_OBJ);

    GS sid0  = VM_SYM(vm, _AR(bx));
    U8 *name = id2name(sid0);			// attribute name with leading '@'
    GS sid1  = name2id(name+1);			// skip the '@'
    GV ret = ostore_get(v, sid1);

    _RA(ret);
}

//================================================================
/*!@brief
  OP_SETIV

  ivset(Syms(Bx),R(A))
*/
__UCODE__
uc_setiv(guru_vm *vm)
{
	GV *v = _R0;
	assert(v->gt==GT_OBJ);

    GS sid0  = VM_SYM(vm, _AR(bx));
    U8 *name = id2name(sid0);			// attribute name with leading '@'
    GS sid1  = name2id(name+1);			// skip the '@'
    ostore_set(v, sid1, _R(a));

    _RA(*v);
}

//================================================================
/*!@brief
  OP_GETCV

  R(A) := cvget(Syms(Bx))
*/
__UCODE__
uc_getcv(guru_vm *vm)
{
	GV *v  = _R0;
	GS sid = VM_SYM(vm, _AR(bx));

	assert(v->gt==GT_OBJ);

	GV cv; { cv.gt=GT_CLASS; cv.cls=v->self->cls; }
	GV ret;
	for (guru_class *cls=v->self->cls;
			cls && (ret=ostore_get(&cv, sid)).gt!=GT_NIL; cls=cls->super);

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
	GV *v = _R0;
	assert(v->gt==GT_CLASS);

    GS sid = VM_SYM(vm, _AR(bx));
    ostore_set(v, sid, _R(a));

    _RA(*v);
}

//================================================================
/*!@brief
  OP_GETCONST

  R(A) := constget(Syms(Bx))
*/
__UCODE__
uc_getconst(guru_vm *vm)
{
    GS sid = VM_SYM(vm, _AR(bx));

    GV *v = const_get(sid);

    _RA(*v);
}

//================================================================
/*!@brief
  OP_SETCONST

  constset(Syms(Bx),R(A))
*/
__UCODE__
uc_setconst(guru_vm *vm)
{
	GV *v  = _R0;
	GS sid = VM_SYM(vm, _AR(bx));

    const_set(sid, _R(a));

    _RA(*v);
}

//================================================================
/*!@brief
  OP_GETUPVAR

  R(A) := uvget(B,C)
*/
__UCODE__
uc_getupvar(guru_vm *vm)
{
    U32 n = (_AR(c)+1) << 1;		// depth of call stack (2 for each level)

    guru_state *st;
    for (st=vm->state; st && n>0; st=st->prev, n--);	// walk up call stack

    GV *ur = st->regs + _AR(b);		// outer scope register file

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
    U32 n = (_AR(c) + 1) << 1;				// 2 per outer scope level

    guru_state *st;
    for (st=vm->state; st && n>0; st=st->prev, n--);
    GV *ur = st->regs + _AR(b);				// pointer to caller's register file

    ref_dec(ur);
    ref_inc(_R(a));
    *ur = *_R(a);                   		// update outer-scope vars
}

//================================================================
/*!@brief
  OP_JMP

  pc += sBx
*/
__UCODE__
uc_jmp(guru_vm *vm)
{
	GI sbx = _AR(bx) - MAX_sBx -1;

	vm->state->pc += sbx;
}

//================================================================
/*!@brief
  OP_JMPIF

  if R(A) pc += sBx
*/
__UCODE__
uc_jmpif (guru_vm *vm)
{
	GI sbx = _AR(bx) - MAX_sBx - 1;
	GV *ra = _R(a);

	if (ra->gt > GT_FALSE) {
		vm->state->pc += sbx;
	}
	*ra = EMPTY();
}

//================================================================
/*!@brief
  OP_JMPNOT

  if not R(A) pc += sBx
*/
__UCODE__
uc_jmpnot(guru_vm *vm)
{
	GI sbx = _AR(bx) - MAX_sBx -1;
	GV *ra = _R(a);
	if (ra->gt <= GT_FALSE) {
		vm->state->pc += sbx;
	}
	*ra = EMPTY();
}

//================================================================
/*!@brief
  OP_ONERR

  rescue_push(pc+sBx)
*/
__UCODE__
uc_onerr(guru_vm *vm)
{
	GI sbx = _AR(bx) - MAX_sBx -1;
	vm->rescue[vm->depth++] = vm->state->pc + sbx;
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
	U32 c   = _AR(c);				// exception 0:set, 1:get
	GV  *v  = _R(a);				// object to receive the exception
	GV  *v1 = v+1;					// exception message (if any)

	if (c) {						// 2nd: get cycle
		if (v->gt==GT_EMPTY) {		// if exception is not given
			_RA_X(v1);				// override exception (msg) if not given
		}
		v1->gt  = GT_TRUE;			// here: modifying return stack directly is questionable!!
		v1->acl = 0;
	}
	else {							// 1st: set cycle
		_RA_X(v1);					// keep exception in RA
		*v1 = EMPTY();
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
	U32 a = _AR(a);

	assert(vm->depth >= a);

	vm->depth -= a;
}

//================================================================
/*!@brief
  OP_RAISE

  raise(R(A))
*/
__UCODE__
uc_raise(guru_vm *vm)
{
	GV *ra = _R(a);

	_RA(*ra);
}

//================================================================
/*!@brief
  _undef

  create undefined method error message (different between mruby1.4 and ruby2.x
*/
__GURU__ GV *
_undef(GV *buf, GV *v, GS sid)
{
	U8 *name = id2name(class_by_obj(v)->sid);

	guru_str_add_cstr(buf, "undefined method '");
	guru_str_add_cstr(buf, id2name(sid));
	guru_str_add_cstr(buf, "' for class #");
	guru_str_add_cstr(buf, name);

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
    GS  sid = VM_SYM(vm, _AR(b));					// get given symbol
    GV  *v  = _R(a);								// call stack, obj is receiver object

    if (vm_method_exec(vm, v, _AR(c), sid)) {		// in state.cu, call stack will be wiped before return
    	// put error message on return stack
    	GV buf = guru_str_buf(80);
    	*(v+1) = *_undef(&buf, v, sid);				// TODO: exception class
    	vm->err = 1;								// raise exception
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
	assert(1==0);				// should not be here, no test case yet!

	guru_irep *irep = _R0->proc->irep;

	vm_state_push(vm, irep, _R0, 0);
}

//================================================================
/*!@brief
  OP_ENTER

  arg setup according to flags (23=5:5:1:5:5:1:1)
*/
__UCODE__
uc_enter(guru_vm *vm)
{
	U32 ax   = (vm->bytecode>>7) & 0x1ffffff;			// a special decoder case

	U32 arg0 = (ax >> 13) & 0x1f;  // default args
    U32 argc = (ax >> 18) & 0x1f;  // given args

    if (arg0 > 0){
        vm->state->pc += vm->state->argc - argc;
    }
}

//================================================================
/*!@brief
  OP_RETURN

  return R(A) (B=normal,in-block return/break)
*/
__UCODE__
uc_return(guru_vm *vm)
{
	GV  ret = *_R(a);							// return value
	U32 n   = _AR(a);							// pc adjustment
	guru_state *st = vm->state;

	if (IS_ITERATOR(st)) {
		GV        *r0 = _R0;					// top of stack
		guru_iter *it = (r0-1)->iter;
		U32 nvar = guru_iter_next(r0-1);		// get next iterator element
		if (nvar) {
			*(r0+1) = *it->ivar;
			if (nvar>1) *(r0+2) = *(it->ivar+1);
			vm->state->pc = 0;
			return;
		}
		// pop off iterator state
		guru_iter_del(r0-1);					// free the iterator object
		vm_state_pop(vm, ret, n);
		vm->state->flag &= ~STATE_LOOP;
	}
	vm_state_pop(vm, ret, n);					// pop callee's context
}

//================================================================
/*!@brief
  OP_BLKPUSH (yield implementation)

  R(A) := block (16=6:1:5:4)
*/
__UCODE__
uc_blkpush(guru_vm *vm)
{
    GV *prc = _R0+1;       				// get proc, regs[0] is the class

    assert(prc->gt==GT_PROC);			// ensure

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
	GV *r0 = _R(a);
	U32 n  = _AR(c);

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
	GV  *r0 = _R(a);
	U32 n   = _AR(c);

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
#define AOP(a, OP)						\
do {									\
	GV *r0 = _R(a);						\
	GV *r1 = r0+1;						\
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
	*r1 = EMPTY();						\
} while(0)

//================================================================
/*!@brief
  OP_ADD

  R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)
*/
__UCODE__
uc_add(guru_vm *vm)
{
	AOP(a, +);
}

//================================================================
/*!@brief
  OP_SUB

  R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)
*/
__UCODE__
uc_sub(guru_vm *vm)
{
	AOP(a, -);
}

//================================================================
/*!@brief
  OP_MUL

  R(A) := R(A)*R(A+1) (Syms[B]=:*)
*/
__UCODE__
uc_mul(guru_vm *vm)
{
	AOP(a, *);
}

//================================================================
/*!@brief
  OP_DIV

  R(A) := R(A)/R(A+1) (Syms[B]=:/)
*/
__UCODE__
uc_div(guru_vm *vm)
{
	GV *r1 = _R(a)+1;

	if (r1->i==0) {
		vm->err = 1;
	}
	else AOP(a, /);
}

//================================================================
/*!@brief
  OP_EQ

  R(A) := R(A)==R(A+1)  (Syms[B]=:==,C=1)
*/
__UCODE__
uc_eq(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;
    GT tt = GT_BOOL(guru_cmp(r0, r1)==0);

    *r1 = EMPTY();
    _RA_T(tt, i=0);
}

// comparator template (poorman's C++)
#define NCMP(a, OP)										\
do {													\
	GV *r0 = _R(a);										\
	GV *r1 = r0+1;										\
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
    *r1 = EMPTY();  									\
} while (0)

//================================================================
/*!@brief
  OP_LT

  R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)
*/
__UCODE__
uc_lt(guru_vm *vm)
{
	NCMP(a, <);
}

//================================================================
/*!@brief
  OP_LE

  R(A) := R(A)<=R(A+1)  (Syms[B]=:<=,C=1)
*/
__UCODE__
uc_le(guru_vm *vm)
{
	NCMP(a, <=);
}

//================================================================
/*!@brief
  OP_GT

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
*/
__UCODE__
uc_gt(guru_vm *vm)
{
	NCMP(a, >);
}

//================================================================
/*!@brief
  OP_GE

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
*/
__UCODE__
uc_ge(guru_vm *vm)
{
	NCMP(a, >=);
}

//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))
*/
__UCODE__
uc_string(guru_vm *vm)
{
#if GURU_USE_STRING
    GV v = VM_STR(vm, _AR(bx));
    _RA(v);
#else
    QUIT("String class");
#endif // GURU_USE_STRING
}

//================================================================
/*!@brief
  String Catination

  str_cat(R(A),R(B))
*/
__UCODE__
uc_strcat(guru_vm *vm)
{
#if GURU_USE_STRING
    GS sid = name2id((U8*)"to_s");				// from global symbol pool
	GV *sa = _R(a), *sb = _R(b);

    guru_proc *pa = proc_by_sid(sa, sid);
    guru_proc *pb = proc_by_sid(sb, sid);

    if (pa) pa->func(sa, 0);					// can it be an IREP?
    if (pb) pb->func(sb, 0);

    guru_str_add_cstr(ref_inc(sa), (U8*)sb->str->raw);	// ref counts increased as _dup updated

    ref_dec(sb);
    *sb = EMPTY();

    _RA(*sa);									// this will clean out sa

#else
    QUIT("String class");
#endif // GURU_USE_STRING
}

__UCODE__
_stack_copy(GV *d, GV *s, U32 n)
{
	for (U32 i=0; i < n; i++, d++, s++) {
		*d = *ref_inc(s);			// now referenced by array/hash
		*s = EMPTY();				// DEBUG: clean element from call stack
	}
}
//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))
*/
__UCODE__
uc_array(guru_vm *vm)
{
#if GURU_USE_ARRAY
    U32 n = _AR(c);
    GV  v = (GV)guru_array_new(n);		// ref_cnt is 1 already

    guru_array *h = v.array;
    _stack_copy(h->data, _R(b), h->n=n);

    _RA(v);									// no need to ref_inc
#else
    QUIT("Array class");
#endif // GURU_USE_ARRAY
}

//================================================================
/*!@brief
  Create Hash object

  R(A) := hash_new(R(B),R(B+1)..R(B+C))
*/
__UCODE__
uc_hash(guru_vm *vm)
{
#if GURU_USE_ARRAY
	U32 n   = _AR(c);						// number of kv pairs
    GV  ret = guru_hash_new(n);				// ref_cnt is already set to 1

    guru_hash *h = ret.hash;
    _stack_copy(h->data, _R(b), h->n=(n<<1));

    _RA(ret);							    // new hash on stack top
#else
    QUIT("Hash class");
#endif // GURU_USE_ARRAY
}

//================================================================
/*!@brief
  OP_RANGE

  R(A) := range_new(R(B),R(B+1),C)
*/
__UCODE__
uc_range(guru_vm *vm)
{
#if GURU_USE_ARRAY
	U32 x   = _AR(c);						// exclude_end
	GV  *p0 = _R(b), *p1 = p0+1;
    GV  v   = guru_range_new(p0, p1, !x);	// p0, p1 ref cnt will be increased
    *p1 = EMPTY();

    _RA(v);									// release and  reassign
#else
    QUIT("Range class");
#endif // GURU_USE_ARRAY
}

//================================================================
/*!@brief
  OP_LAMBDA

  R(A) := lambda(SEQ[Bz],Cz)
*/
__UCODE__
uc_lambda(guru_vm *vm)
{
	U32 bz = _AR(bx) >> 2;					// Bz, Cz a special decoder case

    guru_proc *prc = (guru_proc *)guru_alloc(sizeof(guru_proc));

    prc->sid  = 0xffff;						// anonymous function
    prc->func = NULL;
    prc->irep = VM_REPS(vm, bz);			// fetch from children irep list

    _RA_T(GT_PROC, proc=prc);				// regs[ra].proc = prc
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
	GV *r1 = _R(a)+1;

    guru_class *super = (r1->gt==GT_CLASS) ? r1->cls : vm->state->klass;
    GS         sid    = VM_SYM(vm, _AR(b));
    const U8   *name  = id2name(sid);
    guru_class *cls   = guru_define_class(name, super);

    _RA_T(GT_CLASS, cls=cls);
}

//================================================================
/*!@brief
  OP_EXEC

  R(A) := blockexec(R(A),SEQ[Bx])
*/
__UCODE__
uc_exec(guru_vm *vm)
{
	assert(_R0->gt == GT_CLASS);				// check
	guru_irep *irep = VM_REPS(vm, _AR(bx));		// child IREP[rb]

    vm_state_push(vm, irep, _R(a), 0);			// push call stack
}

//================================================================
/*!@brief
  OP_METHOD

  R(A).newmethod(Syms(B),R(A+1))
*/
__UCODE__
uc_method(guru_vm *vm)
{
	GV  *v = _R(a);

    assert(v->gt==GT_OBJ || v->gt == GT_CLASS);	// enforce class checking

    // check whether the name has been defined in current class (i.e. vm->state->klass)
    GS sid = VM_SYM(vm, _AR(b));				// fetch name from IREP symbol table
    guru_proc *prc = proc_by_sid(v, sid);		// fetch proc from obj->klass->vtbl

#if GURU_DEBUG
    if (prc != NULL) {
    	// same proc name exists (in either current or parent class)
		// printf("WARN: %s#%s override base\n", id2name(class_by_obj(v)->sid), id2name(sid));
    }
#endif
    prc = (v+1)->proc;							// override (if exist) with proc by OP_LAMBDA

    _LOCK;

    // add proc to class
    guru_class 	*cls = class_by_obj(v);
    prc->sid  = sid;							// assign sid to proc, overload if prc already exists
    prc->next = cls->vtbl;						// add to top of vtable, so it will be found first
    cls->vtbl = prc;							// if there is a sub-class override

    _UNLOCK;

    CLR_SCLASS(v);								// clear SCLASS (meta) flag
    *(v+1) = EMPTY();							// clean up proc
}

//================================================================
/*!@brief
  OP_TCLASS

  R(A) := target_class
*/
__UCODE__
uc_tclass(guru_vm *vm)
{
	_RA_T(GT_CLASS, cls=vm->state->klass);
}

//================================================================
/*!@brief
  OP_SCLASS

  R(A) := R(B).singleton_class
*/
__UCODE__
uc_sclass(guru_vm *vm)
{
	GV *o = _R(b);
	if (o->gt==GT_OBJ) {							// singleton class (extending an object)
		const U8   *name  = (U8*)"_single";
		guru_class *super = class_by_obj(o);
		guru_class *cls   = guru_define_class(name, super);
		o->self->cls 	  = cls;
	}
	else if (o->gt==GT_CLASS) {						// meta class (for class methods)
		if (o->cls->cls==NULL) {					// lazy allocation
			const U8	*name = (U8*)"_meta";
			guru_class 	*cls  = guru_define_class(name, guru_class_object);
			o->cls->cls   = cls;					// self pointing =~ meta class
			SET_META(o);
		}
	}
	else assert(1==0);
	SET_SCLASS(o);
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
	U32 b  = vm->bytecode = VM_BYTECODE(vm);	// fetch from vm->state->pc
	U32 n  = b >> 7;	      					// operands
	vm->ar = *((GAR *)&n);        				// operands struct/union

	vm->state->pc++;				// advance program counter (ready for next fetch)
}

__GURU__ void
ucode_exec(guru_vm *vm)
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
    case OP_GETCONST:   uc_getconst  (vm); break;
    case OP_SETCONST:   uc_setconst  (vm); break;
    case OP_GETUPVAR:   uc_getupvar  (vm); break;
    case OP_SETUPVAR:   uc_setupvar  (vm); break;
// BRANCH
    case OP_JMP:        uc_jmp       (vm); break;
    case OP_JMPIF:      uc_jmpif     (vm); break;
    case OP_JMPNOT:     uc_jmpnot    (vm); break;
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
    case OP_STRING:     uc_string    (vm); break;
    case OP_STRCAT:     uc_strcat    (vm); break;
    case OP_ARRAY:      uc_array     (vm); break;
    case OP_HASH:       uc_hash      (vm); break;
    case OP_RANGE:      uc_range     (vm); break;
// CLASS, PROC (STACK ops)
    case OP_LAMBDA:     uc_lambda    (vm); break;
    case OP_CLASS:      uc_class     (vm); break;
    case OP_EXEC:       uc_exec      (vm); break;
    case OP_METHOD:     uc_method    (vm); break;
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
	//=======================================================================================
	// GURU dispatcher unit
	// using vtable (i.e. without switch branching)
	//=======================================================================================
    static UCODE vtbl[] = {
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
		uc_send,		//    OP_SENDB,     A B C   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))*/
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
	// 0x37 Complex Object
		uc_array,		//    OP_ARRAY,     A B C   R(A) := ary_new(R(B),R(B+1)..R(B+C))
		NULL,			//    OP_ARYCAT,    A B     ary_cat(R(A),R(B))
		NULL,			//    OP_ARYPUSH,   A B     ary_push(R(A),R(B))
		NULL,			//    OP_AREF,      A B C   R(A) := R(B)[C]
		NULL,			//    OP_ASET,      A B C   R(B)[C] := R(A)
		NULL,			//    OP_APOST,     A B C   *R(A),R(A+1)..R(A+C) := R(A)[B..]
		uc_string,		//    OP_STRING,    A Bx    R(A) := str_dup(Lit(Bx))
		uc_strcat,		//    OP_STRCAT,    A B     str_cat(R(A),R(B))
		uc_hash,		//    OP_HASH,      A B C   R(A) := hash_new(R(B),R(B+1)..R(B+C))
		uc_lambda,		//    OP_LAMBDA,    A Bz Cz R(A) := lambda(SEQ[Bz],Cz)
		uc_range,		//    OP_RANGE,     A B C   R(A) := range_new(R(B),R(B+1),C)
	// 0x42 Class
		NULL,			//    OP_OCLASS,    A       R(A) := ::Object
		uc_class,		//    OP_CLASS,     A B     R(A) := newclass(R(A),Syms(B),R(A+1))
		uc_class,		//    OP_MODULE,    A B     R(A) := newmodule(R(A),Syms(B))
		uc_exec,		//    OP_EXEC,      A Bx    R(A) := blockexec(R(A),SEQ[Bx])
		uc_method,		//    OP_METHOD,    A B     R(A).newmethod(Syms(B),R(A+1))
		uc_sclass,		//    OP_SCLASS,    A B     R(A) := R(B).singleton_class
		uc_tclass,		//    OP_TCLASS,    A       R(A) := target_class
		NULL,			//    OP_DEBUG,     A B C   print R(A),R(B),R(C)
	// 0x4a Exit
		uc_stop,		//    OP_STOP,      stop VM
		NULL			//    OP_ERR,       Bx      raise RuntimeError with message Lit(Bx)
	};

    guru_state 	*st = vm->state;
    vtbl[vm->op](vm);

    if (vm->err && vm->depth>0) {						// simple exception handler
    	vm->state->pc = vm->rescue[--vm->depth];		// bubbling up
    	vm->err = 0;									// TODO: add exception type or code on stack
    }
#endif // GURU_DEBUG
}

