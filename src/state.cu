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

#include "alloc.h"
#include "store.h"
#include "symbol.h"
#include "ucode.h"
#include "state.h"
#include "vm.h"

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__ void
vm_state_push(guru_vm *vm, guru_irep *irep, GV *regs, U32 argc)
{
	guru_state *top = vm->state;
    guru_state *st  = (guru_state *)guru_alloc(sizeof(guru_state));

    switch(regs[0].gt) {
    case GT_CLASS:
    case GT_OBJ:   st->klass = regs[0].cls;			break;
    case GT_PROC:  st->klass = top->regs[0].cls; 	break;
    default: CHECK_NULL(NULL);
    }
    st->irep  = irep;
    st->pc    = 0;
    st->regs  = regs;
    st->argc  = argc;			// allocate local stack

    st->prev  = top;			// push into context stack

    vm->state = st;				// TODO: use array-based stack
    vm->depth++;
}

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__ void
vm_state_pop(guru_vm *vm, GV ret_val, U32 ra)
{
    guru_state 	*st = vm->state;

    GV *r = st->regs + ra;		// TODO: check whether 2 is correct
    for (U32 i=0; i<=st->argc; i++) {
    	ref_dec(&r[i]);
        r[i].gt = GT_EMPTY;
    }
    st->regs[0] = ret_val;

    vm->state = st->prev;		// restore previous state
    vm->depth--;
    guru_free(st);				// release memory block
}

__GURU__ void
vm_proc_call(guru_vm *vm, GV v[], U32 argc)
{
	assert(v[0].gt==GT_PROC);				// ensure it is a proc

	guru_irep *irep = v[0].proc->irep;

	vm_state_push(vm, irep, v, argc);		// switch into callee's context
}

// Object.new
__GURU__ void
vm_object_new(guru_vm *vm, GV v[], U32 argc)
{
	assert(v[0].gt==GT_CLASS);	// ensure it is a class object

    GV  obj = guru_store_new(v[0].cls, 0);			// allot memory for the new object instance
    //
    // build a temp IREP for calling "initialize"
    // TODO: make the image static
    //
    guru_irep irep[2] = {							// IREP to call super#initialize
        {
            sizeof(irep)*2, 0, 						// size, reps	u32, u32
            0, 										// pool			u32
            sizeof(guru_irep),						// sym			u32		(symbol offset)
            sizeof(guru_irep)+2*sizeof(U32),		// iseq			u32		(ISEQ offset)
            2,										// i  			u32
            0, 0,									// c, p			u16, u16
            2, 0, 0 								// s, nv, nr	u16, u8, u8 (2 symbols)
        },
        {
        	sizeof(guru_irep)+4*sizeof(U32), 0,		// symbol table (for 2 symbols to align ISEQ block)
        	OP_ABC(OP_SEND,0,0,argc),				// ISEQ block
        	MK_OP(OP_STOP),							// keep the memory
        	0x74696e69,								// "initialize"
        	0x696c6169,
        	0x657a,	0x0000
        }
    };

    v[0] = obj;

    // context switch, which is not multi-thread ready
    // TODO: create a vm context object with separate regfile, i.e. explore _push_state/_pop_state
    vm_state_push(vm, irep, v, 0);
    do {						// execute the mini IREP
    	ucode_prefetch(vm);
    	ucode_exec(vm);
    } while (vm->run==VM_STATUS_RUN);

    vm_state_pop(vm, obj, 1);	// pop stack but needs to keep the newly created object
    vm->run = VM_STATUS_RUN;
//
//    RETURN_VAL(obj);
}

