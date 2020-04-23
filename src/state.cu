/*! @file
  @brief
  GURU VM state transition management functions and missing functions (which needs VM)

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  1. VM attribute accessor macros
  2. internal state management functions
  </pre>
*/
#include "guru.h"
#include "util.h"
#include "symbol.h"		// id2name
#include "mmu.h"
#include "ostore.h"		// ostore_new
#include "iter.h"

#include "base.h"
#include "class.h"		// proc_by_id
#include "state.h"

//================================================================
/*!@brief
  Clean up call stack
*/
__GURU__ void
_wipe_stack(GV v[], U32 vi)
{
    GV *r = v;
    for (U32 i=0; i<vi; i++, r++) {
    	ref_dec(r);
    	r->gt   = GT_EMPTY;
    	r->acl  = 0;
    	r->self = NULL;
    }
}

__GURU__ void
_call(guru_vm *vm, GV v[], U32 vi)
{
	guru_proc 	*prc  = v->proc;
	guru_irep	*irep = prc->irep;

	if (AS_LAMBDA(prc)) {
		vm_state_push(vm, vm->state->irep, vm->state->pc, prc->regs, vi);	// switch into callee's context
		vm->state->flag |= STATE_LAMBDA;
		vm_state_push(vm, irep, 0, v, vi);			// switch into lambda using closure stack frame
	}
	else if (AS_IREP(prc)){
		vm_state_push(vm, prc->irep, 0, v, vi);		// switch into callee's context
	}
	else ASSERT(1==0);
}

__GURU__ void
_each(guru_vm *vm, GV v[], U32 vi)
{
	GV *v1 = v+1;
	ASSERT(v1->gt==GT_PROC);						// ensure it is a code block

	U32			pc0    = vm->state->pc;
	guru_irep  	*irep0 = vm->state->irep;
	guru_irep 	*irep1 = v1->proc->irep;
	GV 			git    = guru_iter_new(v, NULL);	// create iterator

	// push stack out (1 space for iterator)
//	GV  *p = v1;
//	for (U32 i=0; i<=vi; i++, *(p+1)=*p, p--);
	*(v+1) = git;
	*(v+2) = *vm->state->regs;

	// allocate iterator state (using same stack frame)
	vm_state_push(vm, irep0, pc0, v+2, vi);
	vm->state->flag |= STATE_LOOP;

	// switch into callee's context with v[1]=1st element
	vm_state_push(vm, irep1, 0, v+2, vi);
	guru_iter *it = git.iter;
	*(v+3) = *(it->inc);
	if (it->n==GT_HASH) {
		*(v+4) = *(it->inc+1);
	}
}

__GURU__ void
_new(guru_vm *vm, GV v[], U32 vi)
{
	ASSERT(v->gt==GT_CLASS);					// ensure it is a class object
	GV obj = v[0] = ostore_new(v->cls);			// instantiate object itself (with 0 var);
	GS sid = name2id((U8*)"initialize"); 		// search for initializer

	if (vm_method_exec(vm, v, vi, sid)) {		// run custom initializer if any
		vm->err = 1;
	}
	vm->state->flag |= STATE_NEW;
}

__GURU__ void
_lambda(guru_vm *vm, GV v[], U32 vi)
{
	ASSERT(v->gt==GT_CLASS && (v+1)->gt==GT_PROC);		// ensure it is a proc

	guru_proc *prc = (v+1)->proc;						// mark it as a lambda
	prc->kt |= PROC_LAMBDA;

	U32	n   = prc->n 	= vm->ar.a;
	GV  *r  = prc->regs = guru_gv_alloc(n);
	GV  *r0 = vm->state->regs;							// deep copy register file
	for (U32 i=0; i<n; *r++=*r0++, i++);

    *v = *(v+1);
	(v+1)->gt = GT_EMPTY;
}

__GURU__ void
_raise(guru_vm *vm, GV v[], U32 vi)
{
	ASSERT(vm->depth > 0);

	vm->state->pc = vm->rescue[--vm->depth];	// pop from exception return stack
}

typedef void (*Xfunc)(guru_vm *vm, GV v[], U32 vi);
struct Xf {
	const char  *name;				// raw string usually
	Xfunc 		func;				// C-function pointer
};
__GURU__ __const__ Xf miss_vtbl[] = {
	{ "call", 	_call	},			// C-based prc_call (hacked handler, it needs vm->state)
	{ "each",   _each   },			// push into call stack, obj at stack[0]
	{ "times",  _each   },			// looper
	{ "new",    _new    },
	{ "lambda", _lambda },			// create object
	{ "raise",  _raise  }			// exception handler
};
#define XFSZ	(sizeof(miss_vtbl)/sizeof(Xf))

__GURU__ U32
_method_missing(guru_vm *vm, GV v[], U32 vi, GS sid)
{
	U8 *f = id2name(sid);

#if CC_DEBUG
	printf("0x%02x:%s not found -------\n", sid, f);
#endif // CC_DEBUG

	struct Xf *p = (Xf*)miss_vtbl;	// dispatcher
	for (int i=0; i<XFSZ; i++, p++) {
		if (STRCMP(p->name, f)==0) {
			p->func(vm, v, vi);
			return 0;
		}
	}
	_wipe_stack(v+1, vi+1);			// wipe call stack and return
	return 1;
}
//================================================================
/*!@brief
  Push current status to callinfo stack
*/
__GURU__ void
vm_state_push(guru_vm *vm, guru_irep *irep, U32 pc, GV v[], U32 vi)
{
	guru_state 	*top = vm->state;
    guru_state 	*st  = (guru_state *)guru_alloc(sizeof(guru_state));

    switch(v->gt) {
    case GT_OBJ:
    case GT_CLASS: 	st->klass = v->cls;				break;
    case GT_PROC: 	st->klass = top->regs[0].cls; 	break;
    default: ASSERT(1==0);
    }
    st->irep  = irep;
    st->pc    = pc;
    st->regs  = v;					// TODO: should allocate another regfile
    st->argc  = vi;					// argument count
    st->flag  = 0;					// non-iterator
    st->prev  = top;				// push into context stack

    if (top) {						// keep stack frame depth
    	top->nv = IN_LAMBDA(st) ? v->proc->n : vm->ar.a;
    }
    else st->nv = irep->nr;			// top most stack frame depth

    vm->state = st;					// TODO: use array-based stack
}

//================================================================
/*!@brief
  Push current status to callinfo stack

	@param 	vm
	@param	ret_val - return_value
	@param	rsz		- stack depth used
*/
__GURU__ void
vm_state_pop(guru_vm *vm, GV ret_val)
{
    guru_state 	*st = vm->state;

    if (!(st->flag & STATE_LAMBDA)) {
    	ref_inc(&ret_val);								// to be referenced by the caller
    	_wipe_stack(st->regs, st->irep->nr);
    	st->regs[0] = ret_val;							// put return value on top of current stack
    }
    vm->state = st->prev;								// restore previous state
    guru_free(st);										// release memory block
}

__GURU__ U32
vm_loop_next(guru_vm *vm)
{
	guru_state *st = vm->state;
	GV *r0 = st->regs;
	GV *ri = r0 - 1;									// iterator pointer

	U32 nvar = guru_iter_next(ri);						// get next iterator element
	if (nvar==0) return 0;								// end of loop, bail

	GV  *r = r0 + (nvar+1);								// wipe stack for next loop
	U32 n  = st->irep->nr - (nvar+1);
	_wipe_stack(r, n);

	guru_iter *it = ri->iter;							// get iterator itself
	*(r0+1) = *it->inc;									// fetch next loop index
	if (nvar>1) *(r0+2) = *(it->inc+1);					// range
	st->pc = 0;

	return 1;
}

__GURU__ U32
vm_method_exec(guru_vm *vm, GV v[], U32 vi, GS sid)
{
    guru_proc  *prc = proc_by_sid(v, sid);				// v->gt in [GT_OBJ, GT_CLASS]

    if (prc==0) {										// VM functions
    	return _method_missing(vm, v, vi, sid);
    }
    if (AS_IREP(prc)) {									// a Ruby-based IREP
    	vm_state_push(vm, prc->irep, 0, v, vi);			// switch to callee's context
    }
    else {
    	v->oid = sid;									// pass as parameter borrowing object id field
    	prc->func(v, vi);								// call C-based function
    	_wipe_stack(v+1, vi+1);
    	v->acl &= ~(ACL_SCLASS|ACL_SELF);
    }
    return 0;
}

