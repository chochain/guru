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

__GURU__ void 	vm_state_push(guru_vm *vm, guru_irep *irep, GV *regs, U32 vi);
__GURU__ void	vm_state_pop(guru_vm *vm, GV ret_val, U32 rsz);

// TODO: temp functions for call and new (due to VM passing required)
__GURU__ U32	vm_method_exec(guru_vm *vm, GV v[], U32 vi, guru_proc *prc);

#ifdef __cplusplus
}
#endif
#endif	// _GURU_SRC_STATE_H_
