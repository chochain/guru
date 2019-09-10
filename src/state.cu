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
#include "opcode.h"
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

    assert(regs[0].gt==GT_CLASS);	// just to make sure

    st->klass = regs[0].cls;	// receiver class
    st->irep  = irep;
    st->pc    = 0;
    st->regs  = regs;
    st->argc  = argc;			// allocate local stack

    st->prev  = top;			// push into context stack
    vm->state = st;				// TODO: use array-based stack
}

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__ void
vm_state_pop(guru_vm *vm, GV ret_val)
{
    guru_state 	*st = vm->state;

    st->regs[0] = ret_val;
    vm->state   = st->prev;
    
    guru_free(st);
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

    GV obj = guru_store_new(v[0].cls, 0);
    //
    // build a temp IREP for calling "initialize"
    // TODO: make the image static
    //
    const U32 sym  = sizeof(guru_irep);			// symbol offset
    const U32 iseq = sym  + 2 * sizeof(U32);	// iseq   offset
    const U32 stbl = iseq + 2 * sizeof(U32);	// symbol string table
    guru_irep irep[2] = {
        {
            0, 					// size						u32
            0, 0, sym, iseq,	// reps, pool, sym, iseq	u32, u32, u32, u32
            0,					// i  						u32
            0, 0,				// c, p						u16, u16
            2, 0, 0 			// s, nv, nr				u16, u8, u8
        },
        {
        	stbl,														// symbol table
        	(MKOPCODE(OP_SEND)|MKARG_A(0)|MKARG_B(0)|MKARG_C(argc)),	// ISEQ block
        	(MKOPCODE(OP_ABORT)),
        	0x74696e69, 0x696c6169, 0x0a00657a,	0						// "initialize"
        }
    };

    ref_clr(&v[0]);
    v[0] = obj;
    ref_inc(&obj);

    // context switch, which is not multi-thread ready
    // TODO: create a vm context object with separate regfile, i.e. explore _push_state/_pop_state
    vm_state_push(vm, irep, v, 0);
    do {					// execute the mini IREP
    	guru_op(vm);
    } while (!vm->done);
    vm->done = 0;
    vm_state_pop(vm, vm->state->regs[0]);

    RETURN_VAL(obj);
}

