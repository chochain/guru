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
_wipe_stack(GR r[], U32 ri)
{
    GR *x = r;
    for (U32 i=0; i<ri; i++, x++) {
    	ref_dec(x);
    	x->gt  = GT_EMPTY;
    	x->acl = 0;
    	x->off = 0;
    }
}

__GURU__ void
_call(guru_vm *vm, GR r[], U32 ri)
{
	guru_proc 	*prc  = GR_PRC(r);
	guru_irep	*irep = prc->irep;

	if (AS_LAMBDA(prc)) {
		vm_state_push(vm, vm->state->irep, vm->state->pc, prc->regs, ri);	// switch into callee's context
		vm->state->flag |= STATE_LAMBDA;
		vm_state_push(vm, irep, 0, r, ri);			// switch into lambda using closure stack frame
	}
	else if (AS_IREP(prc)){
		vm_state_push(vm, prc->irep, 0, r, ri);		// switch into callee's context
	}
	else ASSERT(1==0);
}

__GURU__ void
_each(guru_vm *vm, GR r[], U32 ri)
{
	GR *r1 = r+1;
	ASSERT(r1->gt==GT_PROC);						// ensure it is a code block

	U32			pc0    = vm->state->pc;
	guru_irep  	*irep0 = vm->state->irep;
	guru_irep 	*irep1 = GR_PRC(r1)->irep;
	GR 			git    = guru_iter_new(r, NULL);	// create iterator

	// push stack out (1 space for iterator)
//	GR  *p = r1;
//	for (U32 i=0; i<=ri; i++, *(p+1)=*p, p--);
	*(r+1) = git;
	*(r+2) = *vm->state->regs;

	// allocate iterator state (using same stack frame)
	vm_state_push(vm, irep0, pc0, r+2, ri);
	vm->state->flag |= STATE_LOOP;

	// switch into callee's context with v[1]=1st element
	vm_state_push(vm, irep1, 0, r+2, ri);
	guru_iter *it = GR_ITR(&git);
	*(r+3) = *(it->inc);
	if (it->n==GT_HASH) {
		*(r+4) = *(it->inc+1);
	}
}

__GURU__ void
_new(guru_vm *vm, GR r[], U32 ri)
{
	ASSERT(r->gt==GT_CLASS);					// ensure it is a class object
	GR obj = r[0] = ostore_new(r->off);			// instantiate object itself (with 0 var);
	GS sid = name2id((U8*)"initialize"); 		// search for initializer

	if (vm_method_exec(vm, r, ri, sid)) {		// run custom initializer if any
		vm->err = 1;
	}
	vm->state->flag |= STATE_NEW;
}

__GURU__ void
_lambda(guru_vm *vm, GR r[], U32 ri)
{
	ASSERT(r->gt==GT_CLASS && (r+1)->gt==GT_PROC);		// ensure it is a proc

	guru_proc *prc = GR_PRC(r+1);						// mark it as a lambda
	prc->kt |= PROC_LAMBDA;

	U32	n   = prc->n 	= vm->ar.a;
	GR  *rf = prc->regs = guru_gr_alloc(n);
	GR  *r0 = vm->state->regs;							// deep copy register file
	for (U32 i=0; i<n; *rf++=*r0++, i++);

    *r = *(r+1);
	(r+1)->gt = GT_EMPTY;
}

__GURU__ void
_raise(guru_vm *vm, GR r[], U32 ri)
{
	ASSERT(vm->depth > 0);

	vm->state->pc = vm->rescue[--vm->depth];	// pop from exception return stack
}

typedef void (*Xfunc)(guru_vm *vm, GR r[], U32 ri);
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
_method_missing(guru_vm *vm, GR r[], U32 ri, GS sid)
{
	U8 *f = id2name(sid);

#if CC_DEBUG
	printf("0x%02x:%s not found -------\n", sid, f);
#endif // CC_DEBUG

	struct Xf *p = (Xf*)miss_vtbl;	// dispatcher
	for (int i=0; i<XFSZ; i++, p++) {
		if (STRCMP(p->name, f)==0) {
			p->func(vm, r, ri);
			return 0;
		}
	}
	_wipe_stack(r+1, ri+1);			// wipe call stack and return
	return 1;
}
//================================================================
/*!@brief
  Push current status to callinfo stack
*/
__GURU__ void
vm_state_push(guru_vm *vm, guru_irep *irep, U32 pc, GR r[], U32 ri)
{
	guru_state 	*top = vm->state;
    guru_state 	*st  = (guru_state *)guru_alloc(sizeof(guru_state));

    switch(r->gt) {
    case GT_OBJ:
    case GT_CLASS: 	st->klass = r->off;				break;
    case GT_PROC: 	st->klass = top->regs[0].off; 	break;
    default: ASSERT(1==0);
    }
    st->irep  = irep;
    st->pc    = pc;
    st->regs  = r;					// TODO: should allocate another regfile
    st->argc  = ri;					// argument count
    st->flag  = 0;					// non-iterator
    st->prev  = top;				// push into context stack

    if (top) {						// keep stack frame depth
    	top->nv = IN_LAMBDA(st) ? GR_PRC(r)->n : vm->ar.a;
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
vm_state_pop(guru_vm *vm, GR ret_val)
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
	GR *r0 = st->regs;
	GR *rr = r0 - 1;									// iterator pointer

	U32 nvar = guru_iter_next(rr);						// get next iterator element
	if (nvar==0) return 0;								// end of loop, bail

	GR  *x = r0 + (nvar+1);								// wipe stack for next loop
	U32 n  = st->irep->nr - (nvar+1);
	_wipe_stack(x, n);

	guru_iter *it = GR_ITR(rr);							// get iterator itself
	*(r0+1) = *it->inc;									// fetch next loop index
	if (nvar>1) *(r0+2) = *(it->inc+1);					// range
	st->pc = 0;

	return 1;
}

__GURU__ U32
vm_method_exec(guru_vm *vm, GR r[], U32 ri, GS sid)
{
    guru_proc  *prc = proc_by_sid(r, sid);				// v->gt in [GT_OBJ, GT_CLASS]

    if (prc==0) {										// VM functions
    	return _method_missing(vm, r, ri, sid);
    }
    if (AS_IREP(prc)) {									// a Ruby-based IREP
    	vm_state_push(vm, prc->irep, 0, r, ri);			// switch to callee's context
    }
    else {
    	r->oid = sid;									// parameter sid is passed as object id
    	prc->func(r, ri);								// call C-based function
    	_wipe_stack(r+1, ri+1);
    	r->acl &= ~(ACL_SCLASS|ACL_SELF);
    }
    return 0;
}

