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
#include "guru.h"
#include "vm.h"
#ifndef GURU_SRC_STATE_H_
#define GURU_SRC_STATE_H_

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void 	vm_state_push(guru_vm *vm, guru_irep *irep, U32 pc, GV *regs, U32 argc);
__GURU__ void	vm_state_pop(guru_vm *vm, GV *ret_val);

// TODO: temp functions for call and new (due to VM passing required)
__GURU__ void	vm_proc_call(guru_vm *vm, GV v[], U32 argc);
__GURU__ void	vm_object_new(guru_vm *vm, GV v[], U32 argc);

#ifdef __cplusplus
}
#endif
#endif	// _GURU_SRC_STATE_H_
