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

#include "alloc.h"
#include "static.h"
#include "symbol.h"
#include "global.h"
#include "ucode.h"
#include "object.h"
#include "ostore.h"
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
#define _RA_X(r)    	(ref_dec(_R(a)), *_R(a)=*(r), ref_inc(r))
#define _RA_T(t,e)      (_R(a)->gt=(t), _R(a)->e)

#define SKIP(x)	        { guru_na(x); return; }
#define QUIT(x)			{ vm->quit=1; guru_na(x); return; }
//================================================================
/*!@brief
  OP_NOP

  No operation
*/
__GURU__ void
uc_nop(guru_vm *vm)
{
}

//================================================================
/*!@brief
  OP_MOVE

  R(A) := R(B)
*/
__GURU__ void
uc_move(guru_vm *vm)
{
    _RA(*_R(b)); 	 	// [ra] <= [rb]
}

//================================================================
/*!@brief
  OP_LOADL

  R(A) := Pool(Bx)
*/
__GURU__ void
uc_loadl(guru_vm *vm)
{
    U32 *p = VM_VAR(vm, _AR(bx));
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
  OP_LOADI

  R(A) := sBx
*/
__GURU__ void
uc_loadi(guru_vm *vm)
{
    GI sbx = _AR(bx) - MAX_sBx;		// sBx
    _RA_T(GT_INT, i=sbx);
}


//================================================================
/*!@brief
  OP_LOADSYM

  R(A) := Syms(Bx)
*/
__GURU__ void
uc_loadsym(guru_vm *vm)
{
    GS  sid = name2id(VM_SYM(vm, _AR(bx)));

	_RA_T(GT_SYM, i=sid);
}

//================================================================
/*!@brief
  OP_LOADNIL

  R(A) := nil
*/
__GURU__ void
uc_loadnil(guru_vm *vm)
{
    _RA_T(GT_NIL, i=0);
}

//================================================================
/*!@brief
  OP_LOADSELF

  R(A) := self
*/
__GURU__ void
uc_loadself(guru_vm *vm)
{
    _RA(*_R0);	              		// [ra] <= class
}

//================================================================
/*!@brief
  OP_LOADT

  R(A) := true
*/
__GURU__ void
uc_loadt(guru_vm *vm)
{
    _RA_T(GT_TRUE, i=0);
}

//================================================================
/*!@brief
  OP_LOADF

  R(A) := false
*/
__GURU__ void
uc_loadf(guru_vm *vm)
{
    _RA_T(GT_FALSE, i=0);
}

//================================================================
/*!@brief
  OP_GETGLOBAL

  R(A) := getglobal(Syms(Bx))
*/
__GURU__ void
uc_getglobal(guru_vm *vm)
{
    GS sid  = name2id(VM_SYM(vm, _AR(bx)));

    guru_obj *obj = global_object_get(sid);

    _RA(*obj);
}

//================================================================
/*!@brief
  OP_SETGLOBAL

  setglobal(Syms(Bx), R(A))
*/
__GURU__ void
uc_setglobal(guru_vm *vm)
{
    GS sid = name2id(VM_SYM(vm, _AR(bx)));

    global_object_add(sid, _R(a));
}

//================================================================
/*!@brief
  OP_GETIV

  R(A) := ivget(Syms(Bx))
*/
__GURU__ void
uc_getiv(guru_vm *vm)
{
    U8P name = VM_SYM(vm, _AR(bx));
    GS  sid  = name2id(name+1);					// skip '@'
    GV  ret  = ostore_get(_R0, sid);

    _RA(ret);
}

//================================================================
/*!@brief
  OP_SETIV

  ivset(Syms(Bx),R(A))
*/
__GURU__ void
uc_setiv(guru_vm *vm)
{
    U8P name = VM_SYM(vm, _AR(bx));
    GS  sid  = name2id(name+1);			// skip '@'

    ostore_set(_R0, sid, _R(a));
}

//================================================================
/*!@brief
  OP_GETCONST

  R(A) := constget(Syms(Bx))
*/
__GURU__ void
uc_getconst(guru_vm *vm)
{
    GS  sid = name2id(VM_SYM(vm, _AR(bx)));

    guru_obj *obj = const_object_get(sid);

    _RA(*obj);
}

//================================================================
/*!@brief
  OP_SETCONST

  constset(Syms(Bx),R(A))
*/
__GURU__ void
uc_setconst(guru_vm *vm)
{
	GS sid = name2id(VM_SYM(vm, _AR(bx)));

    const_object_add(sid, _R(a));
}

//================================================================
/*!@brief
  OP_GETUPVAR

  R(A) := uvget(B,C)
*/
__GURU__ void
uc_getupvar(guru_vm *vm)
{
    U32 n = (_AR(c)+1) << 1;		// depth of call stack (2 for each level)

    guru_state *st;
    for (st=vm->state; st && n>0; st=st->prev, n--);	// walk up call stack

    GV *ur = st->regs + _AR(b);		// outer scope register file

    _RA(*ur);          				// ra <= up[rb]
}

//================================================================
/*!@brief
  OP_SETUPVAR

  uvset(B,C,R(A))
*/
__GURU__ void
uc_setupvar(guru_vm *vm)
{
    U32 n = (_AR(c) + 1) << 1;				// 2 per outer scope level

    guru_state *st;
    for (st=vm->state; st && n>0; st=st->prev, n--);
    GV *ur = st->regs + _AR(b);				// pointer to caller's register file

    ref_dec(ur);
    *ur = *_R(a);                   		// update outer-scope vars
}

//================================================================
/*!@brief
  OP_JMP

  pc += sBx
*/
__GURU__ void
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
__GURU__ void
uc_jmpif (guru_vm *vm)
{
	GI sbx = _AR(bx) - MAX_sBx - 1;
	if (_R(a)->gt > GT_FALSE) {
		vm->state->pc += sbx;
	}
}

//================================================================
/*!@brief
  OP_JMPNOT

  if not R(A) pc += sBx
*/
__GURU__ void
uc_jmpnot(guru_vm *vm)
{
	GI sbx = _AR(bx) - MAX_sBx -1;
	if (_R(a)->gt <= GT_FALSE) {
		vm->state->pc += sbx;
	}
}


//================================================================
/*!@brief
  OP_SEND / OP_SENDB

  OP_SEND   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
  OP_SENDB  R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))
*/
__GURU__ void
uc_send(guru_vm *vm)
{
	U32 _b   = _AR(b);
	U32 _c   = _AR(c);
    U8  *sym = VM_SYM(vm, _b);						// get given symbol
	GS  sid  = name2id(sym); 						// get method sid of the current class
    GV  *v   = _R(a);								// call stack, obj is receiver object

	guru_proc *m = (guru_proc *)proc_by_sid(v, sid);

	if (vm_method_exec(vm, v, _c, m)) {
		SKIP(sym);
	}
}

//================================================================
/*!@brief
  OP_CALL

  R(A) := self.call(frame.argc, frame.argv)
*/
__GURU__ void
uc_call(guru_vm *vm)
{
	guru_irep *irep = _R0->proc->irep;

	vm_state_push(vm, irep, _R0, 0);
}

//================================================================
/*!@brief
  OP_ENTER

  arg setup according to flags (23=5:5:1:5:5:1:1)
*/
__GURU__ void
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
__GURU__ void
uc_return(guru_vm *vm)
{
	U32 n   = _AR(a);					// pc adjustment
	GV  ret = *_R(a);

    vm_state_pop(vm, ret, n);			// pass return value
}

//================================================================
/*!@brief
  OP_BLKPUSH (yield implementation)

  R(A) := block (16=6:1:5:4)
*/
__GURU__ void
uc_blkpush(guru_vm *vm)
{
    GV *prc = _R0+1;       				// get proc, regs[0] is the class

    assert(prc->gt==GT_PROC);			// ensure

    _RA(*prc);             				// ra <= proc
}

//================================================================
/*!@brief
  OP_ADD

  R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)
*/
__GURU__ void
uc_add(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;

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
    	uc_send(vm);			// should have already released regs[ra + n], ...
    }
    r1->gt = GT_EMPTY;
}

//================================================================
/*!@brief
  OP_ADDI

  R(A) := R(A)+C (Syms[B]=:+)
*/
__GURU__ void
uc_addi(guru_vm *vm)
{
	GV *r0 = _R(a);
	U32 n  = _AR(c);

    if (r0->gt==GT_INT)     	r0->i += n;
#if GURU_USE_FLOAT
    else if (r0->gt==GT_FLOAT)	r0->f += n;
#else
    else QUIT("Float class");
#endif
}

//================================================================
/*!@brief
  OP_SUB

  R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)
*/
__GURU__ void
uc_sub(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;

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
    	uc_send(vm);
    }
    ref_clr(r1);
}

//================================================================
/*!@brief
  OP_SUBI

  R(A) := R(A)-C (Syms[B]=:-)
*/
__GURU__ void
uc_subi(guru_vm *vm)
{
	GV  *r0 = _R(a);
	U32 n   = _AR(c);

    if (r0->gt==GT_INT) 		r0->i -= n;
#if GURU_USE_FLOAT
    else if (r0->gt==GT_FLOAT) 	r0->f -= n;
#else
    else QUIT("Float class");
#endif
}

//================================================================
/*!@brief
  OP_MUL

  R(A) := R(A)*R(A+1) (Syms[B]=:*)
*/
__GURU__ void
uc_mul(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;

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
    	uc_send(vm);
    }
    ref_clr(r1);
}

//================================================================
/*!@brief
  OP_DIV

  R(A) := R(A)/R(A+1) (Syms[B]=:/)
*/
__GURU__ void
uc_div(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;

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
    	uc_send(vm);
    }
    ref_clr(r1);
}

//================================================================
/*!@brief
  OP_EQ

  R(A) := R(A)==R(A+1)  (Syms[B]=:==,C=1)
*/
__GURU__ void
uc_eq(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;
    GT tt = GT_BOOL(guru_cmp(r0, r1)==0);
    r1->gt = GT_EMPTY;

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
		uc_send(vm);									\
	}													\
    ref_clr(r1);										\
} while (0)

//================================================================
/*!@brief
  OP_LT

  R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)
*/
__GURU__ void
uc_lt(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;
	ncmp(r0, <, r1);
}

//================================================================
/*!@brief
  OP_LE

  R(A) := R(A)<=R(A+1)  (Syms[B]=:<=,C=1)
*/
__GURU__ void
uc_le(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;
    ncmp(r0, <=, r1);
}

//================================================================
/*!@brief
  OP_GT

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
*/
__GURU__ void
uc_gt(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;
    ncmp(r0, >, r1);
}

//================================================================
/*!@brief
  OP_GE

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
*/
__GURU__ void
uc_ge(guru_vm *vm)
{
	GV *r0 = _R(a), *r1 = r0+1;
    ncmp(r0, >=, r1);
}

//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))
*/
__GURU__ void
uc_string(guru_vm *vm)
{
#if GURU_USE_STRING
	U32 bx   = _AR(bx);
    U8  *str = (U8 *)VM_VAR(vm, bx);			// string pool var
    GV  v    = guru_str_new(str);				// rc set to 1 already

    _RA(v);
#else
    QUIT("String class");
#endif
}

//================================================================
/*!@brief
  String Catination

  str_cat(R(A),R(B))
*/
__GURU__ void
uc_strcat(guru_vm *vm)
{
#if GURU_USE_STRING
    GS sid = name2id((U8P)"to_s");				// from global symbol pool
	GV *sa = _R(a), *sb = _R(b);

    guru_proc *pa = proc_by_sid(sa, sid);
    guru_proc *pb = proc_by_sid(sb, sid);

    if (pa) pa->func(sa, 0);					// can it be an IREP?
    if (pb) pb->func(sb, 0);

    GV v = guru_str_add(sa, sb);				// ref counts updated

    ref_dec(sa);								// free both strings
    ref_dec(sb);
    sb->gt = GT_EMPTY;

    _RA_X(&v);

#else
    QUIT("String class");
#endif
}

//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))
*/
__GURU__ void
uc_array(guru_vm *vm)
{
#if GURU_USE_ARRAY
    U32 n = _AR(c);
    GV  v = (GV)guru_array_new(n);		// ref_cnt is 1 already
    guru_array *h = v.array;

    GV *s = _R(b);						// source elements
	GV *d = h->data;					// target
	for (U32 i=0; i<(n); i++, ref_inc(s), *d++=*s, s->gt=GT_EMPTY, s++);
    h->n = n;

    _RA(v);								// no need to ref_inc
#else
    QUIT("Array class");
#endif
}

//================================================================
/*!@brief
  Create Hash object

  R(A) := hash_new(R(B),R(B+1)..R(B+C))
*/
__GURU__ void
uc_hash(guru_vm *vm)
{
#if GURU_USE_ARRAY
/*
	GV *regs = vm->state->regs;
	U32 code = vm->bytecode;

    U32 ra = GET_RA(code);
    U32 rb = GET_RB(code);
    U32 n  = GET_RC(code);							// entries of hash
	U32 sz = sizeof(GV) * (n<<1);						// size of k,v pairs

    GV *p  = &regs[rb];
    GV ret = guru_hash_new(n);							// ref_cnt is already set to 1
    guru_hash  *h = ret.hash;

    MEMCPY((U8P)h->data, (U8P)p, sz);					// copy k,v pairs
*/
	U32 n  = _AR(c);
    GV  *p = _R(b);
    GV  v  = guru_hash_new(n);							// ref_cnt is already set to 1
    guru_hash  *h = v.hash;

    for (U32 i=0; i<(h->n=(n<<1)); i++, p++) {
    	p->gt = GT_EMPTY;								// clean up call stack
    }
    _RA(v);							          			// set return value on stack top
#else
    QUIT("Hash class");
#endif
}

//================================================================
/*!@brief
  OP_RANGE

  R(A) := range_new(R(B),R(B+1),C)
*/
__GURU__ void
uc_range(guru_vm *vm)
{
#if GURU_USE_ARRAY
	U32 n   = _AR(c);
	GV  *p0 = _R(b), *p1 = p0+1;
    GV  v   = guru_range_new(p0, p1, n);		// p0, p1 ref cnt will be increased
    p1->gt = GT_EMPTY;

    _RA_X(&v);									// release and  reassign
#else
    QUIT("Range class");
#endif
}

//================================================================
/*!@brief
  OP_LAMBDA

  R(A) := lambda(SEQ[Bz],Cz)
*/
__GURU__ void
uc_lambda(guru_vm *vm)
{
	U32 bz = _AR(bx) >> 2;				// Bz, Cz a special decoder case

    guru_proc *prc = (guru_proc *)guru_alloc(sizeof(guru_proc));

    prc->sid  = 0xffffffff;				// anonymous function
    prc->func = NULL;
    prc->irep = VM_REPS(vm, bz);		// fetch from children irep list

    _RA_T(GT_PROC, proc=prc);			// regs[ra].proc = prc
}

//================================================================
/*!@brief
  OP_CLASS

  R(A) := newclass(R(A),Syms(B),R(A+1))
  Syms(B): class name
  R(A+1): super class
*/
__GURU__ void
uc_class(guru_vm *vm)
{
	GV *ra1 = _R(a)+1;

    guru_class *super = (ra1->gt==GT_CLASS) ? ra1->cls : vm->state->klass;
    const U8P  name   = VM_SYM(vm, _AR(b));
    guru_class *cls   = guru_define_class(name, super);

    _RA_T(GT_CLASS, cls=cls);
}

//================================================================
/*!@brief
  OP_EXEC

  R(A) := blockexec(R(A),SEQ[Bx])
*/
__GURU__ void
uc_exec(guru_vm *vm)
{
    guru_irep *irep = VM_REPS(vm, _AR(bx));		// child IREP[rb]

    vm_state_push(vm, irep, _R(a), 0);			// push call stack
}

//================================================================
/*!@brief
  OP_METHOD

  R(A).newmethod(Syms(B),R(A+1))
*/
__GURU__ void
uc_method(guru_vm *vm)
{
	GV  *obj = _R(a);

    assert(obj->gt == GT_CLASS);			// enforce class checking

    // check whether the name has been defined in current class (i.e. vm->state->klass)
    U8		  *sym = VM_SYM(vm, _AR(b));	// fetch name from IREP symbol table
    GS	      sid  = name2id(sym);			// lookup symbol id (which should be registered already)
    guru_proc *prc = proc_by_sid(obj, sid);	// fetch proc from obj->klass->vtbl

#if GURU_DEBUG
    if (prc != NULL) {
    	printf("WARN: %s#%s already defined\n", obj->cls->name, sym);
    	// same proc name exists (in either current or parent class)
    	// do nothing for now
    }
#endif
    prc = (obj+1)->proc;					// override (if exist) with proc by OP_LAMBDA

    _LOCK;

    // add proc to class
    guru_class 	*cls = obj->cls;
    prc->sid  = sid;						// assign sid to proc, overload if prc already exists
    prc->next = cls->vtbl;					// add to top of vtable, so it will be found first
    cls->vtbl = prc;						// if there is a sub-class override

    _UNLOCK;

	U32 _a = _AR(a);
    GV  *r = _R0+1;
    for (U32 i=0; i<_a+1; i++, r++) {		// sweep block parameters
    	ref_dec(r);
    	r->gt  = GT_EMPTY;					// clean up for stat dumper
    	r->cls = NULL;
    }
}

//================================================================
/*!@brief
  OP_TCLASS

  R(A) := target_class
*/
__GURU__ void
uc_tclass(guru_vm *vm)
{
	_RA_T(GT_CLASS, cls=vm->state->klass);
}

//================================================================
/*!@brief
  OP_STOP and OP_ABORT

  stop VM (OP_STOP)
  stop VM without release memory (OP_HOLD)
*/
__GURU__ void
uc_stop(guru_vm *vm)
{
	vm->run  = VM_STATUS_STOP;	// VM suspended
}

//===========================================================================================
// GURU engine
//===========================================================================================
//================================================================
/*!@brief
  GURU Instruction Unit - Prefetcher (fetch bytecode and decode)

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
__GURU__ void
ucode_prefetch(guru_vm *vm)
{
	U32 b = vm->bytecode = VM_BYTECODE(vm);	// fetch from vm->state->pc
	U32 n = b >> 7;	      					// operands
//	vm->opn = n;							// keep operands
//	vm->op  = b & 0x7f;      				// opcode (cannot take address from bitfield yet)
	vm->ar  = *((GAR *)&n);        			// operands struct/union

	vm->state->pc++;				// advance program counter (ready for next fetch)
}

__GURU__ void
ucode_exec(guru_vm *vm)
{
	//=======================================================================================
	// GURU dispatcher unit
	//=======================================================================================
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
#if GURU_USE_STRING
    case OP_STRING:     uc_string    (vm); break;
    case OP_STRCAT:     uc_strcat    (vm); break;
#endif
#if GURU_USE_ARRAY
    case OP_ARRAY:      uc_array     (vm); break;
    case OP_HASH:       uc_hash      (vm); break;
    case OP_RANGE:      uc_range     (vm); break;
#endif
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
}

