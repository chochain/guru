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
#ifndef GURU_SRC_STATE_H_
#define GURU_SRC_STATE_H_
#include "guru.h"
#include "vm.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  IREP Internal REPresentation
*/
typedef struct RIrep {			// 32-byte
	U16 nv;						// # of local variables
    U16 nr;						// # of register used
    U16 r;						// # of child IREP blocks (into list below)
    U16	s;						// # of symbols (into sym below)
    U16 p;						// # of objects in pool (into pool below)
    S16	reps;					// offset to REPS block
    S16	pool;					// offset to POOL block
    S16	iseq;					// offset to ISEQ block
} guru_irep;

#define IREP_ISEQ(i)	((U32*)U8PADD(i, (i)->iseq))
#define IREP_REPS(i)    ((guru_irep*)U8PADD(i, (i)->reps))
#define IREP_POOL(i)    ((GR*)U8PADD(i, (i)->pool))

//================================================================
/*!@brief
  Call information
*/
typedef struct RState {			// 20-byte
    U32 pc;						// program counter
    U8  argc;  					// number of arguments
    U8  flag;					// iterator flag
    U8  nv;						// number of local vars (for screen dump)
    U8  temp;					// reserved

    GP	klass;					// (RClass*) current class
    GP	irep;					// (guru_irep*) pointer to current irep block

    GP  regs;					// (GR*) pointer to current register (in VM register file)
    GP  prev;					// previous state (call stack)
} guru_state;					// VM context

#define STATE_LOOP				0x1
#define STATE_LAMBDA			0x2
#define STATE_NEW				0x4

#define IN_LOOP(st)				((st)->prev && (_STATE((st)->prev)->flag & STATE_LOOP))
#define IN_LAMBDA(st)			((st)->prev && (_STATE((st)->prev)->flag & STATE_LAMBDA))
#define IS_LAMBDA(st)			((st)->flag & STATE_LAMBDA)
#define IS_NEW(st)				((st)->flag & STATE_NEW)

__GURU__ void 	vm_state_push(guru_vm *vm, GP irep, U32 pc, GR r[], U32 ri);
__GURU__ void	vm_state_pop(guru_vm *vm, GR ret_val);

// TODO: temp functions for call and new (due to VM passing required)
__GURU__ U32	vm_loop_next(guru_vm *vm);
__GURU__ U32	vm_method_exec(guru_vm *vm, GR r[], U32 ri, GS sid);

#ifdef __cplusplus
}
#endif

class StateMgr
{
public:
	__GURU__ StateMgr(VM *vm);
	__GURU__ ~StateMgr();

	__GURU__ void 	push_state(GP irep, U32 pc, GR r[], U32 ri);
	__GURU__ void	pop_state(GR ret_val);

	// TODO: temp functions for call and new (due to VM passing required)
	__GURU__ U32	loop_next();
	__GURU__ U32	exec_method(GR r[], U32 ri, GS sid);
	__GURU__ void	free_states();

private:
	class Impl;
	Impl  *_impl;
};

#endif	// _GURU_SRC_STATE_H_
