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
vm_state_push(guru_vm *vm, U32 argc)
{
	guru_state *top = vm->state;
    guru_state *st  = (guru_state *)guru_alloc(sizeof(guru_state));

    st->reg   = top->reg;			// pass register file
    st->irep  = top->irep;
    st->pc 	  = top->pc;
    st->argc  = argc;				// allocate local stack
    st->klass = top->klass;
    st->prev  = top;

    vm->state = st;
}

//================================================================
/*!@brief
  Push current status to callinfo stack

*/
__GURU__ void
vm_state_pop(guru_vm *vm, GV *regs)
{
    guru_state *st = vm->state;
    
    vm->state = st->prev;
    
    GV *p = regs+1;						// clear stacked arguments
    for (U32 i=0; i < st->argc; i++) {
        ref_clr(p++);
    }
    guru_free(st);
}

__GURU__ void
vm_proc_call(guru_vm *vm, GV v[], U32 argc)
{
	vm_state_push(vm, argc);			// check _funcall which is not used

	vm->state->pc   = 0;
	vm->state->irep = v[0].proc->irep;	// switch into callee context
	vm->state->reg  = v;				// shift register file pointer (for local stack)

	v[0].proc->rc++;					// CC: 20181027 added to track proc usage
}

// Object.new
__GURU__ void
vm_object_new(guru_vm *vm, GV v[], U32 argc)
{
    GV obj = guru_store_new(v[0].cls, 0);
    //
    // build a temp IREP for calling "initialize"
    // TODO: make the image static
    //
    const U32 iseq = sizeof(guru_irep);				// iseq   offset
    const U32 sym  = iseq + 2 * sizeof(U32);		// symbol offset
    const U32 stbl = iseq + 4 * sizeof(U32);		// symbol string table
    guru_irep irep[2] = {
        {
            0,                  		// size (u32)
            0, 0, 0, 2, 0, 0,   		// nlv, rreg, rlen, ilen, plen, slen (u16)
            iseq, sym, 0, 0   			// iseq (u32), sym (u32), pool, list
        },
        {
        	(MKOPCODE(OP_SEND)|MKARG_A(0)|MKARG_B(0)|MKARG_C(argc)),	// ISEQ block
        	(MKOPCODE(OP_ABORT)) & 0xffff, (MKOPCODE(OP_ABORT)) >> 16,
        	stbl & 0xffff, stbl >> 16, 	0xaaaa, 0xaaaa,					// symbol table
        	0x74696e69, 0x696c6169, 0x0a00657a, 0xaaaaaaaa				// "initialize"
        }
    };

    ref_clr(&v[0]);
    v[0] = obj;
    ref_inc(&obj);

    // context switch, which is not multi-thread ready
    // TODO: create a vm context object with separate regfile, i.e. explore _push_state/_pop_state
    guru_state  *st = vm->state;

    U16 pc0    = st->pc;
    GV  *reg0  = st->reg;
    guru_irep *irep0 = st->irep;

    st->pc 	 = 0;
    st->irep = irep;
    st->reg  = v;		   // new register file (shift for call stack)

    // start a VM
    // TODO: enter into a VM run queue (also a suspended queue)
    while (guru_op(vm)==0); // run til ABORT, or exception

    st->pc 	 = pc0;
    st->reg  = reg0;
    st->irep = irep0;

    RETURN_VAL(obj);
}

