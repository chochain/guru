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
#include "static.h"		// guru_rom_get_sym
#include "symbol.h"
#include "mmu.h"
#include "ostore.h"		// ostore_new
#include "c_array.h"
#include "iter.h"

#include "base.h"
#include "class.h"		// find_class_by_obj, find_proc
#include "state.h"
#include "puts.h"		// guru_puts

//================================================================
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

__GURU__ U32
_exec(guru_vm *vm, GR r[], S32 ri, GP prc)
{
    guru_proc *px = _PRC(prc);
    if (AS_IREP(px)) {									// a Ruby-based IREP
    	vm_state_push(vm, px->irep, 0, r, ri);			// switch to callee's context
    }
    else {												// must be a C-function
#if CC_DEBUG
    	PRINTF("!!!_CALL(x%x, %p, %d)\n", prc, r, ri);
#endif // CC_DEBUG
    	r->oid = px->pid;								// parameter pid is passed as object id
    	_CALL(prc, r, ri);								// call C-based function
    	_wipe_stack(r+1, ri+1);
    }
    return 0;
}

__GURU__ void
_call(guru_vm *vm, GR r[], S32 ri)
{
	ASSERT(r->gt==GT_PROC);

	guru_proc *px = GR_PRC(r);
	GR  *regs = _REGS(px);
	GP 	irep  = px->irep;

	if (AS_LAMBDA(px)) {
		guru_state *st = VM_STATE(vm);
		vm_state_push(vm, st->irep, st->pc, regs, ri);	// switch into callee's context
		VM_STATE(vm)->flag |= STATE_LAMBDA;				// vm->state changed
		vm_state_push(vm, irep, 0, r, ri);				// switch into lambda using closure stack frame
	}
	else if (AS_IREP(px)){
		vm_state_push(vm, irep, 0, r, ri);				// switch into callee's context
	}
	else ASSERT(1==0);
}

__GURU__ void
__loop(guru_vm *vm, GR r[], S32 ri, U32 collect)
{
	GR *r1 = r + 1;

	ASSERT(r1->gt==GT_PROC);						// ensure it is a code block

	guru_state *st = VM_STATE(vm);
	guru_proc  *px = GR_PRC(r1);
	U32	pc0   = st->pc;
	GP 	irep0 = st->irep;							// current context
	GP 	irep1 = px->irep;							// callee IREP
	GR 	git   = guru_iter_new(r, NULL);				// create iterator

	// push stack out (1 space for iterator)
	GR *p = r;
	*(++p) = collect ? guru_array_new(4) : EMPTY;	// replace prc with map array
	*(++p) = git;
	*(++p) = *_REGS(st);

	// allocate iterator state (using same stack frame)
	vm_state_push(vm, irep0, pc0, p, px->n);
	VM_STATE(vm)->flag |= STATE_LOOP;

	// switch into callee's context with v[1]=1st element
	vm_state_push(vm, irep1, 0, p, px->n);
	guru_iter *it = GR_ITR(&git);
	*(++p) = *(it->inc);
	if (it->n==GT_HASH) {
		*(++p) = *(it->inc+1);
	}
	VM_STATE(vm)->flag |= (collect ? STATE_COLLECT : 0);
}

__GURU__ void
_each(guru_vm *vm, GR r[], S32 ri)
{
	__loop(vm, r, ri, 0);
}

__GURU__ void
_map(guru_vm *vm, GR r[], S32 ri)
{
	__loop(vm, r, ri, 1);
}

__GURU__ void
_new(guru_vm *vm, GR r[], S32 ri)
{
	ASSERT(r->gt==GT_CLASS);							// ensure it is a class object
	GR obj = r[0] = ostore_new(r->off);					// instantiate object itself (with 0 var);
	GS sid = name2id((U8*)"initialize"); 				// search for initializer

	vm_method_exec(vm, r, ri, sid);						// run custom initializer if any

	VM_STATE(vm)->flag |= STATE_NEW;
}

__GURU__ void
_lambda(guru_vm *vm, GR r[], S32 ri)
{
	ASSERT(r->gt==GT_CLASS && (r+1)->gt==GT_PROC);		// ensure it is a proc

	guru_proc *px = GR_PRC(r+1);						// mark it as a lambda
	px->kt |= PROC_LAMBDA;

	U32	n   = px->n = vm->a;
	GR  *rf = guru_gr_alloc(n);
	px->regs = MEMOFF(rf);

	GR  *r0 = _REGS(VM_STATE(vm));						// deep copy register file
	for (int i=0; i<n; *rf++=*r0++, i++);

    *r = *(r+1);
	(r+1)->gt = GT_EMPTY;
}

__GURU__ void
_raise(guru_vm *vm, GR r[], S32 ri)
{
	ASSERT(vm->xcp > 0);

	VM_STATE(vm)->pc = RESCUE_POP(vm);		// pop from exception return stack
}

typedef struct {
	const char  *name;								// raw string usually
	void (*func)(guru_vm *vm, GR r[], S32 ri);		// C-function pointer
	GS			pid;
} Xf;

__GURU__ U32
_method_missing(guru_vm *vm, GR r[], S32 ri, GS pid)
{
	static Xf miss_mtbl[] = {
		{ "call", 		_call,   0 },			// C-based prc_call (hacked handler, it needs vm->state)
		{ "each",   	_each,   0 },			// push into call stack, obj at stack[0]
		{ "times",  	_each,   0 },			// looper
		{ "map",    	_map,    0 },			// mapper
		{ "collect",	_map,    0 },
		{ "new",    	_new,    0 },
		{ "lambda", 	_lambda, 0 },			// create object
		{ "raise",  	_raise,  0 }			// exception handler
	};
	static int xfcnt = sizeof(miss_mtbl)/sizeof(Xf);

	Xf *xp = miss_mtbl;
	if (miss_mtbl[0].pid==0) {				// lazy init
		for (int i=0; i<xfcnt; i++, xp++) {
			xp->pid = guru_rom_add_sym(xp->name);
		}
		xp = miss_mtbl;						// rewind
	}
	for (int i=0; i<xfcnt; i++, xp++) {
		if (xp->pid==pid) {
#if CC_DEBUG
			PRINTF("!!!missing_func %p:%s -> %d\n", xp, xp->name, pid);
#endif // CC_DEBUG
			xp->func(vm, r, ri);
			return 0;
		}
	}
#if CC_DEBUG
	PRINTF("ERROR: method not found (pid=x%04x)-------\n", pid);
#endif // CC_DEBUG
	_wipe_stack(r+1, ri+1);					// wipe call stack and return
	return (vm->err = 1);
}
//================================================================
/*!@brief
  Push current status to callinfo stack
*/
__GURU__ void
vm_state_push(guru_vm *vm, GP irep, U32 pc, GR r[], S32 ri)
{
#if CC_DEBUG
	PRINTF("!!!vm_state_push(%p, x%x, %d, %p, %d)\n", vm, irep, pc, r, ri);
#endif // CC_DEBUG
	guru_state  *top = vm->state ? VM_STATE(vm) : NULL;
    guru_state 	*st  = (guru_state *)guru_alloc(sizeof(guru_state));

    ASSERT(st);

    switch(r->gt) {
    case GT_OBJ:	st->klass = GR_OBJ(r)->cls;		break;
    case GT_CLASS: 	st->klass = r->off;			 	break;
    case GT_PROC: 	st->klass = _REGS(top)->off; 	break; 	// top->regs[0].off (top != NULL)
    default: ASSERT(1==0);
    }
    st->irep  = irep;
    st->pc    = pc;
    st->regs  = MEMOFF(r);			// TODO: should allocate another regfile
    st->argc  = ri;					// argument count
    st->flag  = 0;					// non-iterator
    st->prev  = vm->state;			// push current state into context stack

    if (top) {						// keep stack frame depth
    	top->nv = IN_LAMBDA(st) ? GR_PRC(r)->n : vm->a;
    }
    else {
    	st->nv = ((guru_irep*)MEMPTR(irep))->nr;			// top most stack frame depth
    }
    vm->state = MEMOFF(st);			// TODO: use array-based stack
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
    guru_state 	*st = VM_STATE(vm);

    if (!IS_LAMBDA(st)) {
        guru_irep  *irep = (guru_irep*)MEMPTR(st->irep);
        GR         *regs = _REGS(st);
    	if (ret_val.off!=regs->off) {					// keep ref cnt when object returns itself, i.g. new()
    		ref_inc(&ret_val);							// to be referenced by the caller
    		ref_dec(&regs[0]);
    	}
    	_wipe_stack(regs+1, irep->nr);
    	regs[0] = ret_val;								// put return value on top of current stack
    }
    vm->state = st->prev;								// restore previous state
    guru_free(st);										// release memory block
}

__GURU__ U32
vm_loop_next(guru_vm *vm)
{
	guru_state *st   = VM_STATE(vm);
	guru_irep  *irep = (guru_irep*)MEMPTR(st->irep);
	GR *r0 = _REGS(st);
	GR *it = r0 - 1;									// iterator pointer

	U32 nvar = guru_iter_next(it);						// get next iterator element
	if (nvar==0) return 0;								// end of loop, bail

	GR  *x = r0 + (nvar+1);								// wipe stack for next loop
	U32 n  = irep->nr - (nvar+1);
	_wipe_stack(x, n);

	guru_iter *ix = GR_ITR(it);							// get iterator object itself
	*(r0+1) = *ix->inc;									// fetch next loop index
	if (nvar>1) *(r0+2) = *(ix->inc+1);					// range
	st->pc = 0;

	return 1;
}

__GURU__ U32
vm_method_exec(guru_vm *vm, GR r[], S32 ri, GS pid)
{
#if CC_DEBUG
    PRINTF("!!!vm_method_exec(%p, %p, %d, %d)\n", vm, r, ri, pid);
#endif // CC_DEBUG
	GP cls = find_class_by_obj(r);						// determine active class
    GP prc = find_proc(cls, pid);						// r->gt in [GT_OBJ, GT_CLASS]

    if (prc==0) {										// not found, try VM functions
    	return _method_missing(vm, r, ri, pid);
    }
    return _exec(vm, r, ri, prc);
}

