/*! @file
  @brief
  GURU VM state management

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  1. VM attribute accessor macros
  2. internal state management functions
  </pre>
*/
#include <assert.h>

#include "mmu.h"
#include "refcnt.h"
#include "symbol.h"		// id2name
#include "ostore.h"
#include "class.h"		// proc_by_sid
#include "state.h"
#include "c_range.h"
#include "iter.h"

//================================================================
/*!@brief
  Clean up call stack
*/
__GURU__ void
_wipe_stack(GV v[], U32 vi, GV *rv)
{
    GV *r = v;
    for (U32 i=0; i<vi; i++, r++) {
    	ref_dec(r);
    	r->gt   = GT_EMPTY;
    	r->acl  = 0;
    	r->self = NULL;
    }
}

__GURU__ S32
__match(const U8* s0, U8* s1)
{
	while (*s0 != '\0') {
		if (*s0++ != *s1++) return 0;
	}
	return 1;
}

__GURU__ void
_object_new(guru_vm *vm, GV v[], U32 vi)
{
	assert(v[0].gt==GT_CLASS);						// ensure it is a class object

    GV  obj = ostore_new(v[0].cls, 0);				// instenciate object (with zero ivar)
	GS  sid  = name2id((U8P)"initialize"); 			// search for custom initializer (or Object#c_nop)

	if (vm_method_exec(vm, v, vi, sid)) {			// run custom initializer if any
		assert(1==0);
	}
	v[0] = obj;
}

__GURU__ void
_raise(guru_vm *vm, GV v[], U32 vi)
{
	assert(vm->depth > 0);

	vm->state->pc = vm->rescue[--vm->depth];		// pop from exception return stack
}

__GURU__ void
_proc_call(guru_vm *vm, GV v[], U32 vi)
{
	assert(v[0].gt==GT_PROC);						// ensure it is a proc

	guru_irep *irep = v[0].proc->irep;				// callee's IREP pointer

	vm_state_push(vm, irep, v, vi);					// switch into callee's context
}

__GURU__ void
_each(guru_vm *vm, GV v[], U32 vi)
{
	assert(v[1].gt==GT_PROC);						// ensure it is a code block

	GV 			itr    = guru_iter_new(v, NULL);	// create iterator
	U32        	pc0    = vm->state->pc;
	guru_irep  	*irep0 = vm->state->irep;
	guru_irep  	*irep1 = v[1].proc->irep;

	// push stack out (space for iterator)
	GV  *p0 = (v+1);
	GV  *p1 = (p0 + 2 + vi);
	for (U32 i=0; i<vi+1; i++, *p1--=*p0--);

	// allocate iterator state
	vm_state_push(vm, irep0, v+1, vi);
	vm->state->pc = pc0;
	*(v+2) = itr;

	// switch into callee's context with v[1]=1st element
	vm_state_push(vm, irep1, v+3, vi);
	*(v+4) = itr.iter->ivar;
	vm->state->flag |= STATE_LOOP;
}

__GURU__ U32
_method_missing(guru_vm *vm, GV v[], U32 vi, GS sid)
{
	U8 *f = id2name(sid);

	// function dispatcher
	if      (__match("call", f)) { 					// C-based prc_call (hacked handler, it needs vm->state)
		_proc_call(vm, v, vi);						// push into call stack, obj at stack[0]
	}
	else if (__match("each", f)) {
		_each(vm, v, vi);
	}
	else if (__match("new", f)) {					// other default C-based methods
		_object_new(vm, v, vi);
	}
	else if (__match("raise", f)) {
		_raise(vm, v, vi);
	}
	else {
		_wipe_stack(v+1, vi+1, NULL);				// wipe call stack and return
		return 1;
	}
	return 0;
}
//================================================================
/*!@brief
  Push current status to callinfo stack
*/
__GURU__ void
vm_state_push(guru_vm *vm, guru_irep *irep, GV v[], U32 vi)
{
	guru_state *top = vm->state;
    guru_state *st  = (guru_state *)guru_alloc(sizeof(guru_state));

    switch(v[0].gt) {
    case GT_OBJ:
    case GT_CLASS: st->klass = v[0].cls;			break;
    case GT_PROC:  st->klass = top->regs[0].cls; 	break;
    default: assert(1==0);
    }
    st->irep  = irep;
    st->pc    = 0;
    st->regs  = v;				// TODO: should allocate another regfile
    st->argc  = vi;				// allocate local stack
    st->flag  = 0;				// non-iterator
    st->prev  = top;			// push into context stack

    vm->state = st;				// TODO: use array-based stack
}

//================================================================
/*!@brief
  Push current status to callinfo stack

	@param 	vm
	@param	ret_val - return_value
	@param	rsz		- stack depth used
*/
__GURU__ void
vm_state_pop(guru_vm *vm, GV ret_val, U32 rsz)
{
    guru_state 	*st = vm->state;

    ref_inc(&ret_val);			// to be referenced by the caller
    _wipe_stack(st->regs, rsz + st->argc + 1, &ret_val);		// pop off all elements from stack

    st->regs[0] = ret_val;		// put return value on top of current stack
    vm->state = st->prev;		// restore previous state
    guru_free(st);				// release memory block
}

__GURU__ U32
vm_method_exec(guru_vm *vm, GV v[], U32 vi, GS sid)
{
    guru_proc *prc = (guru_proc *)proc_by_sid(&v[0], sid);

    if (prc==0) {
    	return _method_missing(vm, v, vi, sid);
    }

    if (HAS_IREP(prc)) {						// a Ruby-based IREP
    	vm_state_push(vm, prc->irep, v, vi);	// switch to callee's context
    }
    else {
    	prc->func(v, vi);						// call C-based function
    	_wipe_stack(v+1, vi+1, NULL);
    }
    return 0;
}


