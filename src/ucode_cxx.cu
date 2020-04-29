/*! @file
  @brief
  GURU instruction unit implementation - prefetch &Impl:: microcode dispatcher

  <pre>
  Copyright (C) 2019 GreenII

  1. a list of opcode (microcode) executor, and
  2. the core opcode dispatcher
  </pre>
*/
#include "guru.h"
#include "symbol.h"
#include "global.h"
#include "mmu.h"
#include "inspect.h"
#include "ostore.h"
#include "iter.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

#include "base.h"
#include "state.h"
#include "ucode.h"

//
// becareful with the following macros, because they release regs[ra] first
// so, make sure value is kept before the release
//
#define _AR(r)          ((_vm->ar.r))
#define _R0             (&(_vm->state->regs[0]))
#define _R(r)			(&(_vm->state->regs[_AR(r)]))
#define _RA(v)			(ref_dec(_R(a)), *_R(a)=(v))
#define _RA_X(r)    	(ref_inc(r), ref_dec(_R(a)), *_R(a)=*(r))
#define _RA_T(t,e)      (_R(a)->gt=(t), _R(a)->acl=0, _R(a)->e)

#define SKIP(x)			{ NA(x); return; }

class Ucode::Impl
{
    typedef void (Impl::*UCODEX)();	// microcode function prototype

	guru_vm *_vm;
	UCODEX  *_vt;
	U32     _mutex;

//================================================================
/*!@brief
  sid of attr name with '@' sign removed
*/
    __GURU__ __INLINE__ GS
    _name2id_wo_at_sign()
    {
        GS sid   = VM_SYM(_vm, _AR(bx));
        U8 *name = id2name(sid);			// attribute name with leading '@'

        return name2id(name+1);				// skip the '@'
    }

//================================================================
/*!@brief
  get outer scope register file

*/
    __GURU__ GV *
    _upvar()
    {
        guru_state *st = _vm->state;
        for (U32 i=0; i<=_AR(c); i++) {		// walk up stack frame
            st = IN_LAMBDA(st)
                ? st->prev
                : st->prev->prev;			// 1 extra for each_loop
        }
        return st->regs + _AR(b);
    }

//================================================================
/*!@brief
  _undef

  create undefined method error message (different between mruby1.4 and ruby2.x
*/
    __GURU__ GV *
    _undef(GV *buf, GV *v, GS sid)
    {
        U8 *fname = id2name(sid);
        U8 *cname = id2name(class_by_obj(v)->sid);

        guru_buf_add_cstr(buf, "undefined method '");
        guru_buf_add_cstr(buf, fname);
        guru_buf_add_cstr(buf, "' for class #");
        guru_buf_add_cstr(buf, cname);

        return buf;
    }

    __GURU__ void
    _stack_copy(GV *d, GV *s, U32 n)
    {
        for (U32 i=0; i < n; i++, d++, s++) {
            *d = *ref_inc(s);			// now referenced by array/hash
            *s = EMPTY;					// DEBUG: clean element from call stack
        }
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

//================================================================
/*!@brief
  OP_NOP
  No operation
*/
    __GURU__ void
    nop()
    {
        // do nothing
    }

//================================================================
/*!@brief
  OP_MOVE

  R(A) := R(B)
*/
    __UCODE__
    move()
    {
        _RA_X(_R(b)); 	 			// [ra] <= [rb]
    }

//================================================================
/*!@brief
  OP_LOADL

  R(A) := Pool(Bx)
*/
    __UCODE__
    loadl()
    {
        GV ret = VM_VAR(_vm, _AR(bx));
        _RA(ret);
    }

//================================================================
/*!@brief
  OP_LOADI
  Load 16-bit integer

  R(A) := sBx
*/
    __UCODE__
    loadi()
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
    loadsym()
    {
        GS sid = VM_SYM(_vm, _AR(bx));

        _RA_T(GT_SYM, i=sid);
    }

//================================================================
/*!@brief
  OP_LOADNIL

  R(A) := nil
*/
    __UCODE__
    loadnil()
    {
        _RA_T(GT_NIL, i=0);
    }

//================================================================
/*!@brief
  OP_LOADSELF

  R(A) := self
*/
    __UCODE__
    loadself()
    {
        _RA_X(_R0);	              		// [ra] <= class
    }

//================================================================
/*!@brief
  OP_LOADT

  R(A) := true
*/
    __UCODE__
    loadt()
    {
        _RA_T(GT_TRUE, i=0);
    }

//================================================================
/*!@brief
  OP_LOADF

  R(A) := false
*/
    __UCODE__
    loadf()
    {
        _RA_T(GT_FALSE, i=0);
    }

//================================================================
/*!@brief
  OP_GETGLOBAL

  R(A) := getglobal(Syms(Bx))
*/
    __UCODE__
    getglobal()
    {
        GS sid = VM_SYM(_vm, _AR(bx));

        GV *v = global_get(sid);

        _RA(*v);
    }

//================================================================
/*!@brief
  OP_SETGLOBAL

  setglobal(Syms(Bx), R(A))
*/
    __UCODE__
    setglobal()
    {
        GS sid = VM_SYM(_vm, _AR(bx));

        global_set(sid, _R(a));
    }

//================================================================
/*!@brief
  OP_GETIV

  R(A) := ivget(Syms(Bx))
*/
    __UCODE__
    getiv()
    {
        GV *v  = _R0;
        ASSERT(v->gt==GT_OBJ || v->gt==GT_CLASS);

        GS sid = _name2id_wo_at_sign();
        GV ret = ostore_get(v, sid);

        _RA(ret);
    }

//================================================================
/*!@brief
  OP_SETIV

  ivset(Syms(Bx),R(A))
*/
    __UCODE__
    setiv()
    {
        GV *v  = _R0;
        ASSERT(v->gt==GT_OBJ || v->gt==GT_CLASS);

        GS sid = _name2id_wo_at_sign();
        ostore_set(v, sid, _R(a));
    }

//================================================================
/*!@brief
  OP_GETCV

  R(A) := cvget(Syms(Bx))
*/
    __UCODE__
    getcv()
    {
        GV *v  = _R0;
        GS sid = VM_SYM(_vm, _AR(bx));

        ASSERT(v->gt==GT_OBJ);

        GV cv; { cv.gt=GT_CLASS; cv.acl=0; cv.cls=v->self->cls; }
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
    setcv()
    {
        GV *v = _R0;
        ASSERT(v->gt==GT_CLASS);

        GS sid = VM_SYM(_vm, _AR(bx));
        ostore_set(v, sid, _R(a));
    }

//================================================================
/*!@brief
  OP_GETCONST

  R(A) := constget(Syms(Bx))
*/
    __UCODE__
    getconst()
    {
        GS sid = VM_SYM(_vm, _AR(bx));
        GV *v  = const_get(sid);

        _RA(*v);
    }

//================================================================
/*!@brief
  OP_SETCONST

  constset(Syms(Bx),R(A))
*/
    __UCODE__
    setconst()
    {
        GS sid = VM_SYM(_vm, _AR(bx));
        GV *ra = _R(a);

        ra->acl &= ~ACL_HAS_REF;		// set it to constant

        const_set(sid, ra);
    }


//================================================================
/*!@brief
  OP_GETUPVAR

  R(A) := uvget(B,C)
*/
    __UCODE__
    getupvar()
    {
        GV *ur = _upvar();				// outer scope register file
        _RA_X(ur);          			// ra <= up[rb]
    }

//================================================================
/*!@brief
  OP_SETUPVAR

  uvset(B,C,R(A))
*/
    __UCODE__
    setupvar()
    {
        GV *ur = _upvar();				// pointer to caller's register file
        GV *va = _R(a);

        ref_dec(ur);
        ref_inc(va);
        *ur = *va;                   	// update outer-scope vars
    }

//================================================================
/*!@brief
  OP_JMP

  pc += sBx
*/
    __UCODE__
    jmp()
    {
        GI sbx = _AR(bx) - MAX_sBx -1;

        _vm->state->pc += sbx;
    }

//================================================================
/*!@brief
  OP_JMPIF

  if R(A) pc += sBx
*/
    __UCODE__
    jmpif()
    {
        GI sbx = _AR(bx) - MAX_sBx - 1;
        GV *ra = _R(a);

        if (ra->gt > GT_FALSE) {
            _vm->state->pc += sbx;
        }
        *ra = EMPTY;
    }

//================================================================
/*!@brief
  OP_JMPNOT

  if not R(A) pc += sBx
*/
    __UCODE__
    jmpnot()
    {
        GI sbx = _AR(bx) - MAX_sBx -1;
        GV *ra = _R(a);
        if (ra->gt <= GT_FALSE) {
            _vm->state->pc += sbx;
        }
        *ra = EMPTY;
    }

//================================================================
/*!@brief
  OP_ONERR

  rescue_push(pc+sBx)
*/
    __UCODE__
    onerr()
    {
        ASSERT(_vm->depth < (MAX_RESCUE_STACK-1));

        GI sbx = _AR(bx) - MAX_sBx -1;

        _vm->rescue[_vm->depth++] = _vm->state->pc + sbx;
    }

//================================================================
/*!@brief
  OP_RESCUE

  if (A)
  if (C) R(A) := R(A+1)		get exception
  else   R(A) := R(A+1)		set exception
*/
    __UCODE__
    rescue()
    {
        U32 c  = _AR(c);				// exception 0:set, 1:get
        GV  *v = _R(a);					// object to receive the exception
        GV  *x = v + 1;					// exception on stack

        if (c) {						// 2nd: get cycle
            if (x->gt != GT_NIL) {		// if exception is not given
                _RA_X(x);				// override exception (msg) if not given
            }
            x->gt  = GT_TRUE;			// here: modifying return stack directly is questionable!!
            x->acl = 0;
        }
        else {							// 1st: set cycle
            if (v->gt==GT_CLASS) x++;
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
    poperr()
    {
        U32 a = _AR(a);

        ASSERT(_vm->depth >= a);

        _vm->depth -= a;
    }

//================================================================
/*!@brief
  OP_RAISE

  raise(R(A))
*/
    __UCODE__
    raise()
    {
        GV *ra = _R(a);

        _RA(*ra);
    }

//================================================================
/*!@brief
  OP_SEND / OP_SENDB

  OP_SEND   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
  OP_SENDB  R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&R(A+C+1))
*/
    __UCODE__
    send()
    {
        GS  sid = VM_SYM(_vm, _AR(b));				// get given symbol or object id
        GV  *r  = _R(a);							// call stack, obj is receiver object

        if (vm_method_exec(_vm, r, _AR(c), sid)) { 	// in state.cu, call stack will be wiped before return
            // put error message on return stack
            GV buf = guru_str_buf(80);
            *(r+1) = *_undef(&buf, r, sid);			// TODO: exception class
            _vm->err = 1;							// raise exception
        }
    }

//================================================================
/*!@brief
  OP_CALL
  R(A) := self.call(frame.argc, frame.argv)

  TODO: no test case yet
*/
    __UCODE__
    call()
    {
        ASSERT(1==0);				// should not be here, no test case yet!
    }

//================================================================
/*!@brief
  OP_ENTER

  arg setup according to flags (23=5:5:1:5:5:1:1)		// default parameter
*/
    __UCODE__
    enter()
    {
        U32 ax  = (_vm->bytecode>>7) & 0x1ffffff;		// a special decoder case

        U32 adj = (ax >> 13) & 0x1f;  					// has default args
        U32 off = (ax >> 18) & 0x1f;  					// number of args given

        if (adj){
            _vm->state->pc += _vm->state->argc - off;	// jmp table lookup
        }
    }

//================================================================
/*!@brief
  OP_RETURN

  return R(A) (B=normal,in-block return/break)
*/
    __UCODE__
    uc_return()
    {
        GV  ret = *_R(a);							// return value
        U32 brk = _AR(b);							// break

        guru_state *st = _vm->state;
        if (IN_LOOP(st)) {
            if (vm_loop_next(_vm) && !brk) return;	// continue

            ret = *_R(a);							// fetch last returned value
            guru_iter_del(st->regs - 1);			// release iterator

            // pop off iterator state
            vm_state_pop(_vm, ret);					// pop off ITERATOR state
            ret = *_R0;								// return the object itself
        }
        else if (IN_LAMBDA(st)) {
            vm_state_pop(_vm, ret);					// pop off LAMBDA state
        }
        else if (IS_NEW(st)) {
            ret = *_R0;								// return the object itself
        }
        ret.acl &= ~(ACL_SELF|ACL_SCLASS);			// turn off TCLASS and NEW flags if any
        vm_state_pop(_vm, ret);						// pop callee's context
    }

//================================================================
/*!@brief
  OP_BLKPUSH (yield implementation)

  R(A) := block (16=6:1:5:4)
*/
    __UCODE__
    blkpush()
    {
        guru_state *st = _vm->state;
        for (U32 i=0; i<_AR(c); i++) {
            st = st->prev->prev;
        }
        GV *prc = st->regs+st->argc+1;       	// get proc, regs[0] is the class

        ASSERT(prc->gt==GT_PROC);				// ensure

        _RA_X(prc);             				// ra <= proc
    }

//================================================================
/*!@brief
  OP_ADDI

  R(A) := R(A)+C (Syms[B]=:+)
*/
    __UCODE__
    addi()
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
    subi()
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
#define AOP(a, OP)                              \
    do {                                        \
        GV *r0 = _R(a);                         \
        GV *r1 = r0+1;                          \
        if (r0->gt==GT_INT) {                   \
            if      (r1->gt==GT_INT)   {        \
                r0->i = r0->i OP r1->i;         \
            }                                   \
            else if (r1->gt==GT_FLOAT) {        \
                r0->gt = GT_FLOAT;              \
                r0->f  = r0->i OP r1->f;        \
            }                                   \
            else SKIP("Fixnum + ?");            \
        }                                       \
        else if (r0->gt==GT_FLOAT) {            \
            if      (r1->gt==GT_INT) 	{       \
                r0->f = r0->f OP r1->i;         \
            }                                   \
            else if (r1->gt==GT_FLOAT)	{       \
                r0->f = r0->f OP r1->f;         \
            }                                   \
            else SKIP("Float + ?");             \
        }                                       \
        else {	/* other cases */               \
            send();                             \
        }                                       \
        *r1 = EMPTY;                            \
    } while(0)

//================================================================
/*!@brief
  OP_ADD

  R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)
*/
    __UCODE__
    add() {	AOP(a, +); }

//================================================================
/*!@brief
  OP_SUB

  R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)
*/
    __UCODE__
    sub() { AOP(a, -); }

//================================================================
/*!@brief
  OP_MUL

  R(A) := R(A)*R(A+1) (Syms[B]=:*)
*/
    __UCODE__
    mul() { AOP(a, *); }

//================================================================
/*!@brief
  OP_DIV

  R(A) := R(A)/R(A+1) (Syms[B]=:/)
*/
    __UCODE__
    div()
    {
    	GV *r1 = _R(a)+1;

    	if (r1->i==0) {
    		_vm->err = 1;
    	}
    	else AOP(a, /);
    }

//================================================================
/*!@brief
  OP_EQ

  R(A) := R(A)==R(A+1)  (Syms[B]=:==,C=1)
*/
    __UCODE__
    eq()
    {
    	GV *r0 = _R(a), *r1 = r0+1;
    	GT tt = GT_BOOL(guru_cmp(r0, r1)==0);

    	*r1 = EMPTY;
    	_RA_T(tt, i=0);
    }

// comparator template (poorman's C++)
#define NCMP(a, OP)                             \
    do {                                        \
	GV *r0 = _R(a);                             \
	GV *r1 = r0+1;                              \
	if ((r0)->gt==GT_INT) {                     \
    if ((r1)->gt==GT_INT) {                     \
    (r0)->gt = GT_BOOL((r0)->i OP (r1)->i);		\
}												\
    else if ((r1)->gt==GT_FLOAT) {              \
    (r0)->gt = GT_BOOL((r0)->i OP (r1)->f);		\
}												\
}                                               \
	else if ((r0)->gt==GT_FLOAT) {              \
    if ((r1)->gt==GT_INT) {                     \
    (r0)->gt = GT_BOOL((r0)->f OP (r1)->i);		\
}												\
    else if ((r1)->gt==GT_FLOAT) {              \
    (r0)->gt = GT_BOOL((r0)->f OP (r1)->f);		\
}												\
}                                               \
	else {                                      \
    send();                                     \
}                                               \
    *r1 = EMPTY;                                \
} while (0)

//================================================================
/*!@brief
  OP_LT

  R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)
*/
    __UCODE__
    lt() { NCMP(a, <); }

//================================================================
/*!@brief
  OP_LE

  R(A) := R(A)<=R(A+1)  (Syms[B]=:<=,C=1)
*/
    __UCODE__
    le() { NCMP(a, <=); }

//================================================================
/*!@brief
  OP_GT

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
*/
    __UCODE__
    gt() { NCMP(a, >); }

//================================================================
/*!@brief
  OP_GE

  R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
*/
    __UCODE__
    ge() { NCMP(a, >=); }

//================================================================
/*!@brief
  Create string object

  R(A) := str_dup(Lit(Bx))
*/
    __UCODE__
    string()
    {
        GV v = VM_STR(_vm, _AR(bx));
        _RA(v);
    }

//================================================================
/*!@brief
  String Catination

  str_cat(R(A),R(B))
*/
    __UCODE__
    strcat()
    {
        GS sid = name2id((U8*)"to_s");				// from global symbol pool
        GV *sa = _R(a), *sb = _R(b);

        guru_proc *pa = proc_by_sid(sa, sid);
        guru_proc *pb = proc_by_sid(sb, sid);

        if (pa) pa->func(sa, 0);					// can it be an IREP?
        if (pb) pb->func(sb, 0);

        guru_buf_add_cstr(ref_inc(sa), (U8*)sb->str->raw);	// ref counts increased as _dup updated

        ref_dec(sb);
        *sb = EMPTY;

        _RA(*sa);									// this will clean out sa
    }

//================================================================
/*!@brief
  Create Array object

  R(A) := ary_new(R(B),R(B+1)..R(B+C))
*/
    __UCODE__
    array()
    {
#if GURU_USE_ARRAY
        U32 n = _AR(c);
        GV  v = (GV)guru_array_new(n);			// ref_cnt is 1 already

        guru_array *h = v.array;
        if ((h->n=n)>0) _stack_copy(h->data, _R(b), n);

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
    hash()
    {
#if GURU_USE_ARRAY
        U32 n   = _AR(c);						// number of kv pairs
        GV  ret = guru_hash_new(n);				// ref_cnt is already set to 1

        guru_hash *h = ret.hash;
        if ((h->n=(n<<1))>0) _stack_copy(h->data, _R(b), h->n);

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
    range()
    {
#if GURU_USE_ARRAY
        U32 x   = _AR(c);						// exclude_end
        GV  *p0 = _R(b), *p1 = p0+1;
        GV  v   = guru_range_new(p0, p1, !x);	// p0, p1 ref cnt will be increased
        *p1 = EMPTY;

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
    lambda()
    {
        U32 bz = _AR(bx) >> 2;					// Bz, Cz a special decoder case

        guru_proc *prc = (guru_proc *)guru_alloc(sizeof(guru_proc));

        prc->rc   = 0;
        prc->n    = 0;							// no param
        prc->sid  = 0xffff;						// anonymous function
        prc->kt   = PROC_IREP;
        prc->irep = VM_REPS(_vm, bz);			// fetch from children irep list

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
    klass()
    {
        GV *r1 = _R(a)+1;

        guru_class *super = (r1->gt==GT_CLASS) ? r1->cls : _vm->state->klass;
        GS         sid    = VM_SYM(_vm, _AR(b));
        const U8   *name  = id2name(sid);
        guru_class *cls   = guru_define_class(name, super);

        cls->kt |= USER_DEF_CLASS;					// user defined (i.e. non-builtin) class

        _RA_T(GT_CLASS, cls=cls);
        *r1 = EMPTY;
    }

//================================================================
/*!@brief
  OP_EXEC

  R(A) := blockexec(R(A),SEQ[Bx])
*/
    __UCODE__
    exec()
    {
        ASSERT(_R0->gt == GT_CLASS);				// check
        guru_irep *irep = VM_REPS(_vm, _AR(bx));		// child IREP[rb]

        vm_state_push(_vm, irep, 0, _R(a), 0);		// push call stack
    }

//================================================================
/*!@brief
  OP_METHOD

  R(A).newmethod(Syms(B),R(A+1))
*/
    __UCODE__
    method()
    {
        GV *v  = _R(a);
        ASSERT(v->gt==GT_OBJ || v->gt == GT_CLASS);	// enforce class checking

        // check whether the name has been defined in current class (i.e. _vm->state->klass)
        GS sid = VM_SYM(_vm, _AR(b));				// fetch name from IREP symbol table
        guru_class *cls = class_by_obj(v);			// fetch active class
        guru_proc  *prc = proc_by_sid(v, sid);		// fetch proc from class or obj's vtbl

#if GURU_DEBUG
        if (prc != NULL) {
            // same proc name exists (in either current or parent class)
#if CC_DEBUG
            printf("WARN: %s#%s override base\n", id2name(cls->sid), id2name(sid));
#endif // CC_DEBUG
        }
#endif
        prc = (v+1)->proc;							// override (if exist) with proc by OP_LAMBDA

        MUTEX_LOCK(_mutex);

        // add proc to class
        prc->sid   = sid;							// assign sid to proc, overload if prc already exists
        prc->next  = cls->flist;					// add to top of vtable, so it will be found first
        cls->flist = prc;							// if there is a sub-class override

        MUTEX_FREE(_mutex);

        v->acl &= ~ACL_SELF;						// clear CLASS modification flags if any
        *(v+1) = EMPTY;								// clean up proc
    }

//================================================================
/*!@brief
  OP_TCLASS

  R(A) := target_class
*/
    __UCODE__
    tclass()
    {
        GV *ra = _R(a);

        _RA_T(GT_CLASS, cls=_vm->state->klass);
        ra->acl |= ACL_SELF;
        ra->acl &= ~ACL_SCLASS;
    }

//================================================================
/*!@brief
  OP_SCLASS

  R(A) := R(B).singleton_class
*/
    __UCODE__
    sclass()
    {
        GV *o = _R(b);
        if (o->gt==GT_OBJ) {							// singleton class (extending an object)
            const U8   *name  = (U8*)"_single";
            guru_class *super = class_by_obj(o);
            guru_class *cls   = guru_define_class(name, super);
            o->self->cls = cls;
        }
        else if (o->gt==GT_CLASS) {						// meta class (for class methods)
            guru_class_add_meta(o);						// lazily add metaclass if needed
        }
        else ASSERT(1==0);

        o->acl |= ACL_SCLASS;
        o->acl &= ~ACL_SELF;
    }

//================================================================
/*!@brief
  OP_STOP and OP_ABORT

  stop VM (OP_STOP)
  stop VM without release memory (OP_HOLD)
*/
    __UCODE__
    stop()
    {
        _vm->run  = VM_STATUS_STOP;					// VM suspended
    }

//===========================================================================================
// GURU engine
//===========================================================================================
/*!@brief
  GURU Instruction Unit - Prefetcher (fetch bytecode and decode)

  @param  vm    A pointer of VM.
  @retval 0  No error.
*/
    __GURU__ void prefetch()
    {
        U32 b  = _vm->bytecode = 							// fetch from _vm->state->pc
			_bin_to_u32(U8PADD(VM_ISEQ(_vm), sizeof(U32)*_vm->state->pc));
        U32 n  = b >> 7;	      							// operands
        _vm->ar = *((GAR *)&n);        						// operands struct/union

        _vm->state->pc++;				// advance program counter (ready for next fetch)
    }

    __GURU__ void dispatch()
    {
        guru_state *st = _vm->state;						// for debugging
        (*this.*_vt[_vm->op])();							// C++ calling a pointer to a member function

        if (_vm->err && _vm->depth>0) {						// simple exception handler
            _vm->state->pc = _vm->rescue[--_vm->depth];		// bubbling up
            _vm->err = 0;									// TODO: add exception type or code on stack
        }
    }

public:
    __GURU__ Impl(guru_vm *vm)
    {
        static UCODEX vtbl[] = {
			&Impl::nop, 			// 	  OP_NOP = 0,
			// 0x1 Register File
            &Impl::move,			//    OP_MOVE       A B     R(A) := R(B)
            &Impl::loadl,			//    OP_LOADL      A Bx    R(A) := Pool(Bx)
            &Impl::loadi,			//    OP_LOADI      A sBx   R(A) := sBx
            &Impl::loadsym,			//    OP_LOADSYM    A Bx    R(A) := Syms(Bx)
            &Impl::loadnil,			//    OP_LOADNIL    A       R(A) := nil
            &Impl::loadself,		//    OP_LOADSELF   A       R(A) := self
            &Impl::loadt,			//    OP_LOADT      A       R(A) := true
            &Impl::loadf,			//    OP_LOADF      A       R(A) := false
            // 0x9 Load/Store
            &Impl::getglobal,		//    OP_GETGLOBAL  A Bx    R(A) := getglobal(Syms(Bx))
            &Impl::setglobal,		//    OP_SETGLOBAL  A Bx    setglobal(Syms(Bx), R(A))
            &Impl::nop,				//    OP_GETSPECIAL A Bx    R(A) := Special[Bx]
            &Impl::nop,				//    OP_SETSPECIAL	A Bx    Special[Bx] := R(A)
            &Impl::getiv,			//    OP_GETIV      A Bx    R(A) := ivget(Syms(Bx))
            &Impl::setiv,			//    OP_SETIV      A Bx    ivset(Syms(Bx),R(A))
            &Impl::getcv,			//    OP_GETCV      A Bx    R(A) := cvget(Syms(Bx))
            &Impl::setcv,			//    OP_SETCV      A Bx    cvset(Syms(Bx),R(A))
            &Impl::getconst,		//    OP_GETCONST   A Bx    R(A) := constget(Syms(Bx))
            &Impl::setconst,		//    OP_SETCONST   A Bx    constset(Syms(Bx),R(A))
            &Impl::nop,				//    OP_GETMCNST   A Bx    R(A) := R(A)::Syms(Bx)
            &Impl::nop,				//    OP_SETMCNST   A Bx    R(A+1)::Syms(Bx) := R(A)
            &Impl::getupvar,		//    OP_GETUPVAR   A B C   R(A) := uvget(B,C)
            &Impl::setupvar,		//    OP_SETUPVAR   A B C   uvset(B,C,R(A))
            // 0x17 Branch Unit
            &Impl::jmp,				//    OP_JMP,       sBx     pc+=sBx
            &Impl::jmpif,			//    OP_JMPIF,     A sBx   if R(A) pc+=sBx
            &Impl::jmpnot,			//    OP_JMPNOT,    A sBx   if !R(A) pc+=sBx
            // 0x1a Exception Handler
            &Impl::onerr,			//    OP_ONERR,     sBx     rescue_push(pc+sBx)
            &Impl::rescue,			//    OP_RESCUE		A B C   if A (if C exc=R(A) else R(A) := exc);
            &Impl::poperr,			// 	  OP_POPERR,    A       A.times{rescue_pop()}
            &Impl::raise,			//    OP_RAISE,     A       raise(R(A))
            &Impl::nop,				//    OP_EPUSH,     Bx      ensure_push(SEQ[Bx])
            &Impl::nop,				//    OP_EPOP,      A       A.times{ensure_pop().call}
            // 0x20 Stack
            &Impl::send,			//    OP_SEND,      A B C   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C))
            &Impl::send,			//    OP_SENDB,     A B C   R(A) := call(R(A),Syms(B),R(A+1),...,R(A+C),&Impl::R(A+C+1))
            &Impl::nop,				//    OP_FSEND,     A B C   R(A) := fcall(R(A),Syms(B),R(A+1),...,R(A+C-1))
            &Impl::call,			//    OP_CALL,      A       R(A) := self.call(frame.argc, frame.argv)
            &Impl::nop,				//    OP_SUPER,     A C     R(A) := super(R(A+1),... ,R(A+C+1))
            &Impl::nop,				//    OP_ARGARY,    A Bx    R(A) := argument array (16=6:1:5:4)
            &Impl::enter,			//    OP_ENTER,     Ax      arg setup according to flags (23=5:5:1:5:5:1:1)
            &Impl::nop,				//    OP_KARG,      A B C   R(A) := kdict[Syms(B)]; if C kdict.rm(Syms(B))
            &Impl::nop,				//    OP_KDICT,     A C     R(A) := kdict
            &Impl::uc_return,		//    OP_RETURN,    A B     return R(A) (B=normal,in-block return/break)
            &Impl::nop,				//    OP_TAILCALL,  A B C   return call(R(A),Syms(B),*R(C))
            &Impl::blkpush,			//    OP_BLKPUSH,   A Bx    R(A) := block (16=6:1:5:4)
            // 0x2c ALU
            &Impl::add,				//    OP_ADD,       A B C   R(A) := R(A)+R(A+1) (Syms[B]=:+,C=1)
            &Impl::addi,			//    OP_ADDI,      A B C   R(A) := R(A)+C (Syms[B]=:+)
            &Impl::sub,				//    OP_SUB,       A B C   R(A) := R(A)-R(A+1) (Syms[B]=:-,C=1)
            &Impl::subi,			//    OP_SUBI,      A B C   R(A) := R(A)-C (Syms[B]=:-)
            &Impl::mul,				//    OP_MUL,       A B C   R(A) := R(A)*R(A+1) (Syms[B]=:*,C=1)
            &Impl::div,				//    OP_DIV,       A B C   R(A) := R(A)/R(A+1) (Syms[B]=:/,C=1)
            &Impl::eq,				//    OP_EQ,        A B C   R(A) := R(A)==R(A+1) (Syms[B]=:==,C=1)
            &Impl::lt,				//    OP_LT,        A B C   R(A) := R(A)<R(A+1)  (Syms[B]=:<,C=1)
            &Impl::le,				//    OP_LE,        A B C   R(A) := R(A)<=R(A+1) (Syms[B]=:<=,C=1)
            &Impl::gt,				//    OP_GT,        A B C   R(A) := R(A)>R(A+1)  (Syms[B]=:>,C=1)
            &Impl::ge,				//    OP_GE,        A B C   R(A) := R(A)>=R(A+1) (Syms[B]=:>=,C=1)
            // 0x37 Complex Object
            &Impl::array,			//    OP_ARRAY,     A B C   R(A) := ary_new(R(B),R(B+1)..R(B+C))
            &Impl::nop,				//    OP_ARYCAT,    A B     ary_cat(R(A),R(B))
            &Impl::nop,				//    OP_ARYPUSH,   A B     ary_push(R(A),R(B))
            &Impl::nop,				//    OP_AREF,      A B C   R(A) := R(B)[C]
            &Impl::nop,				//    OP_ASET,      A B C   R(B)[C] := R(A)
            &Impl::nop,				//    OP_APOST,     A B C   *R(A),R(A+1)..R(A+C) := R(A)[B..]
            &Impl::string,			//    OP_STRING,    A Bx    R(A) := str_dup(Lit(Bx))
            &Impl::strcat,			//    OP_STRCAT,    A B     str_cat(R(A),R(B))
            &Impl::hash,			//    OP_HASH,      A B C   R(A) := hash_new(R(B),R(B+1)..R(B+C))
            &Impl::lambda,			//    OP_LAMBDA,    A Bz Cz R(A) := lambda(SEQ[Bz],Cz)
            &Impl::range,			//    OP_RANGE,     A B C   R(A) := range_new(R(B),R(B+1),C)
            // 0x42 Class
            &Impl::nop,				//    OP_OCLASS,    A       R(A) := ::Object
            &Impl::klass,			//    OP_CLASS,     A B     R(A) := newclass(R(A),Syms(B),R(A+1))
            &Impl::klass,			//    OP_MODULE,    A B     R(A) := newmodule(R(A),Syms(B))
            &Impl::exec,			//    OP_EXEC,      A Bx    R(A) := blockexec(R(A),SEQ[Bx])
            &Impl::method,			//    OP_METHOD,    A B     R(A).newmethod(Syms(B),R(A+1))
            &Impl::sclass,			//    OP_SCLASS,    A B     R(A) := R(B).singleton_class
            &Impl::tclass,			//    OP_TCLASS,    A       R(A) := target_class
            &Impl::nop,				//    OP_DEBUG,     A B C   print R(A),R(B),R(C)
            // 0x4a Exit
            &Impl::stop,			//    OP_STOP,      stop VM
            &Impl::nop				//    OP_ERR,       Bx      raise RuntimeError with message Lit(Bx)
        };
        _vm = vm;
        _vt = vtbl;
    }

    __GURU__ int run()
    {
    	while (_vm->run==VM_STATUS_RUN) {					// run my (i.e. blockIdx.x) VM
    		// add before_fetch hooks here
    		prefetch();
    		// add before_exec hooks here
    		dispatch();
    		// add after_exec hooks here
    		if (_vm->step) break;
    	}
    	return _vm->run==VM_STATUS_STOP;
    }
};	// end of class Ucode::Impl

__GURU__ Ucode::Ucode(guru_vm *vm) : _impl(new Impl(vm)) {}
__GURU__ Ucode::~Ucode() = default;

__GURU__ int Ucode::run()
{
	return _impl->run();
}
