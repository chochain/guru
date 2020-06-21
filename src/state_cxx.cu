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
#include "static.h"
#include "symbol.h"		// id2name
#include "mmu.h"
#include "ostore.h"		// ostore_new
#include "c_array.h"	// guru_array_new
#include "iter.h"

#include "base.h"
#include "class.h"		// proc_by_id
#include "state.h"

class StateMgr::Impl
{
	guru_vm *_vm;

	//================================================================
	__GURU__ void
    __loop(GR r[], S32 ri, U32 collect)
	{
		GR *r1 = r+1;
		ASSERT(r1->gt==GT_PROC);						// ensure it is a code block

		guru_state *st = VM_STATE(_vm);
		guru_proc  *px = GR_PRC(r1);
		U32	pc0   = st->pc;
		GP 	irep0 = st->irep;
		GP 	irep1 = px->irep;
		GR 	git   = guru_iter_new(r, NULL);				// create iterator

		// push stack out (1 space for iterator)
		GR *p = r;
		//	for (int i=0; i<=ri; i++, *(p+1)=*p, p--);
        *(++p) = collect ? guru_array_new(4) : EMPTY;   // replace prc with collection array
		*(++p) = git;
		*(++p) = *_REGS(st);

		// allocate iterator state (using same stack frame)
		push_state(irep0, pc0, p, px->n);
		VM_STATE(_vm)->flag |= STATE_LOOP;

		// switch into callee's context with v[1]=1st element
		push_state(irep1, 0, p, px->n);
		guru_iter *it = GR_ITR(&git);
		*(++p) = *(it->inc);
		if (it->n==GT_HASH) {
			*(++p) = *(it->inc+1);
		}
        VM_STATE(_vm)->flag |= (collect ? STATE_COLLECT : 0);
	}

	/*!@brief
  	  Clean up call stack
	 */
	__GURU__ void
	_wipe_stack(GR r[], S32 ri)
	{
		GR *x = r;
		for (int i=0; i<ri; i++, x++) {
			ref_dec(x);
			*x = EMPTY;
		}
	}

	__GURU__ void
	_call(GR r[], S32 ri)
	{
		ASSERT(r->gt==GT_PROC);

		guru_proc *px = GR_PRC(r);
		GR  *regs = _REGS(px);
		GP 	irep  = px->irep;

		if (AS_LAMBDA(px)) {
			guru_state *st = VM_STATE(_vm);
			push_state(st->irep, st->pc, regs, ri);		// switch into callee's context
			VM_STATE(_vm)->flag |= STATE_LAMBDA;		// vm->state changed
			push_state(irep, 0, r, ri);					// switch into lambda using closure stack frame
		}
		else if (AS_IREP(px)){
			push_state(irep, 0, r, ri);					// switch into callee's context
		}
		else ASSERT(1==0);
	}

    __GURU__ void
    _each(GR r[], S32 ri)
    {
        __loop(r, ri, 0);
    }

    __GURU__ void
    _map(GR r[], S32 ri)
    {
        __loop(r, ri, 1);
    }

	__GURU__ void
	_new(GR r[], S32 ri)
	{
		ASSERT(r->gt==GT_CLASS);					// ensure it is a class object
		GR obj = r[0] = ostore_new(r->off);			// instantiate object itself (with 0 var);
		GS sid = name2id((U8*)"initialize"); 		// search for initializer

		if (exec_method(r, ri, sid)) {				// run custom initializer if any
			_vm->err = 1;							// initializer not found
		}
		VM_STATE(_vm)->flag |= STATE_NEW;
	}

	__GURU__ void
	_lambda(GR r[], S32 ri)
	{
		ASSERT(r->gt==GT_CLASS && (r+1)->gt==GT_PROC);		// ensure it is a proc

		guru_proc *px = GR_PRC(r+1);						// mark it as a lambda
		px->kt |= PROC_LAMBDA;

		U32	n   = px->n = _vm->a;
		GR  *rf = guru_gr_alloc(n);
		px->regs = MEMOFF(rf);

		GR  *r0 = _REGS(VM_STATE(_vm));						// deep copy register file
		for (int i=0; i<n; *rf++=*r0++, i++);

		*r = *(r+1);
		(r+1)->gt = GT_EMPTY;
	}

	__GURU__ void
	_raise(GR r[], S32 ri)
	{
		ASSERT(_vm->xcp > 0);

		VM_STATE(_vm)->pc = RESCUE_POP(_vm);		// pop from exception return stack
	}

	__GURU__ U32
	_exec_missing(GR r[], S32 ri, GS pid)
	{
        typedef void (Impl::*MISSX)(GR r[], S32);	// internal handler of missing function
		typedef struct {
			const char  *name;						// raw string usually
			MISSX	    func;						// C-function pointer
			GS			pid;
		} Xf;
		static Xf miss_mtbl[] = {
			{ "call", 	&Impl::_call,   0 },		// C-based prc_call (hacked handler, it needs vm->state)
			{ "each",   &Impl::_each,   0 },		// push into call stack, obj at stack[0]
			{ "times",  &Impl::_each,   0 },		// looper
			{ "map",    &Impl::_map,    0 },		// mapper
			{ "collect",&Impl::_map,    0 },
			{ "new",    &Impl::_new,    0 },
			{ "lambda", &Impl::_lambda, 0 },		// create object
			{ "raise",  &Impl::_raise,  0 }			// exception handler
		};
		static int xfcnt = sizeof(miss_mtbl)/sizeof(Xf);

		Xf *xp = miss_mtbl;
		if (miss_mtbl[0].pid==0) {					// lazy init
			for (int i=0; i<xfcnt; i++, xp++) {
				xp->pid = guru_rom_add_sym(xp->name);
			}
			xp = miss_mtbl;							// rewind
		}
		for (int i=0; i<xfcnt; i++, xp++) {
			if (xp->pid==pid) {
#if CC_DEBUG
                PRINTF("!!!missing_func %p:%s -> %d\n", xp, xp->name, xp->pid);
#endif // CC_DEBUG
				(*this.*xp->func)(r, ri);
				return 0;
			}
		}
#if CC_DEBUG
		printf("0x%02x not found -------\n", pid);
#endif // CC_DEBUG
		_wipe_stack(r+1, ri+1);						// wipe call stack and return
		return 1;
	}

public:
	__GURU__ Impl(guru_vm *vm) : _vm(vm) {}

	//================================================================
	/*!@brief
	  Push current context to callinfo stack

		@param 	vm
		@param	ret_val - return_value
		@param	rsz		- stack depth used
	*/
	__GURU__ void
	push_state(GP irep, U32 pc, GR r[], S32 ri)
	{
	#if CC_DEBUG
		PRINTF("!!!vm_state_push(%p, x%x, %d, %p, %d)\n", _vm, irep, pc, r, ri);
	#endif // CC_DEBUG
		guru_state 	*top = VM_STATE(_vm);
	    guru_state 	*st  = (guru_state *)guru_alloc(sizeof(guru_state));

	    ASSERT(st);

	    switch(r->gt) {
	    case GT_OBJ:
	    case GT_CLASS: 	st->klass = r->off;				break;
	    case GT_PROC: 	st->klass = _REGS(top)->off; 	break;	// top->regs[0].off
	    default: ASSERT(1==0);
	    }
	    st->irep  = irep;
	    st->pc    = pc;
	    st->regs  = MEMOFF(r);			// TODO: should allocate another regfile
	    st->argc  = ri;					// argument count
	    st->flag  = 0;					// non-iterator
	    st->prev  = _vm->state;			// push current state into context stack

	    if (top) {						// keep stack frame depth
	    	top->nv = IN_LAMBDA(st) ? GR_PRC(r)->n : _vm->a;
	    }
	    else {
	    	st->nv  = ((guru_irep*)MEMPTR(irep))->nr;			// top most stack frame depth
	    }
	    _vm->state = MEMOFF(st);		// TODO: use array-based stack
	}

	//================================================================
	/*!@brief
	 Pop current context from callinfo stack
	 */
	__GURU__ void
	pop_state(GR ret_val)
	{
	    guru_state 	*st = VM_STATE(_vm);

	    if (!(st->flag & STATE_LAMBDA)) {
	        guru_irep  *irep = (guru_irep*)MEMPTR(st->irep);
	        GR         *regs = _REGS(st);
	    	ref_inc(&ret_val);								// to be referenced by the caller
	    	_wipe_stack(regs, irep->nr);
	    	regs[0] = ret_val;								// put return value on top of current stack
	    }
	    _vm->state = st->prev;								// restore previous state

	    guru_free(st);										// release memory block
	}

	__GURU__ U32
	loop_next()
	{
		guru_state *st   = VM_STATE(_vm);
		guru_irep  *irep = (guru_irep*)MEMPTR(st->irep);
		GR *r0 = _REGS(st);
		GR *rr = r0 - 1;									// iterator pointer

		U32 nvar = guru_iter_next(rr);						// get next iterator element
		if (nvar==0) return 0;								// end of loop, bail

		GR  *x = r0 + (nvar+1);								// wipe stack for next loop
		U32 n  = irep->nr - (nvar+1);
		_wipe_stack(x, n);

		guru_iter *it = GR_ITR(rr);							// get iterator itself
		*(r0+1) = *it->inc;									// fetch next loop index
		if (nvar>1) *(r0+2) = *(it->inc+1);					// range
		st->pc = 0;

		return 1;
	}

	__GURU__ U32
	exec_method(GR r[], S32 ri, GS pid)
	{

	#if CC_DEBUG
	    PRINTF("!!!vm_method_exec(%p, %p, %d, %d)\n", _vm, r, ri, pid);
	#endif // CC_DEBUG
	    GP prc = ClassMgr::getInstance()->proc_by_id(r, pid);	// v->gt in [GT_OBJ, GT_CLASS]
	    if (prc==0) {											// not found, try VM functions
	    	return _exec_missing(r, ri, pid);
	    }
	    guru_proc *px = _PRC(prc);
	    if (AS_IREP(px)) {										// a Ruby-based IREP
	    	push_state(px->irep, 0, r, ri);						// switch to callee's context
	    }
	    else {													// must be a C-function
	#if CC_DEBUG
	    	PRINTF("!!!_CALL(x%x, %p, %d)\n", prc, r, ri);
	#endif // CC_DEBUG
	    	r->oid = pid;										// parameter sid is passed as object id
	    	_CALL(prc, r, ri);									// call C-based function
	    	_wipe_stack(r+1, ri+1);
	    	r->acl &= ~(ACL_SCLASS|ACL_TCLASS);
	    }
	    return 0;
	}

	__GURU__ void
	free_states()
	{
		guru_state *st = _STATE(_vm->state);
		while (st) {										// pop off call stack
			pop_state(_REGS(st)[1]);						// passing value of regs[1]
			st = _STATE(_vm->state);
		}
		_vm->run   = VM_STATUS_FREE;						// release the vm
		_vm->state = NULL;									// redundant?

	}
};
//
// interface class
//
__GURU__ StateMgr::StateMgr(VM *vm) : _impl(new Impl((guru_vm*)vm)) {}
__GURU__ StateMgr::~StateMgr() = default;
__GURU__ void
StateMgr::push_state(GP irep, U32 pc, GR r[], S32 ri)
{
	_impl->push_state(irep, pc, r, ri);
}
__GURU__ void
StateMgr::pop_state(GR ret_val)
{
	_impl->pop_state(ret_val);
}
__GURU__ U32
StateMgr::loop_next()
{
	return _impl->loop_next();
}
__GURU__ U32
StateMgr::exec_method(GR r[], S32 ri, GS sid)
{
	return _impl->exec_method(r, ri, sid);
}
__GURU__ void
StateMgr::free_states()
{
	_impl->free_states();
}

