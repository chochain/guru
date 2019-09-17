/*! @file
  @brief
  GURU - VM public interfaces

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  Fetch mruby VM bytecodes, decode and execute.

  </pre>
*/

#ifndef GURU_SRC_VMX_H_
#define GURU_SRC_VMX_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

int  vm_pool_init(U32 step);

int  vm_get(U8 *irep_img);
int	 vm_hold(U32 vid);
int	 vm_stop(U32 vid);
int	 vm_run(U32 vid);

int  vm_main_start(U32 trace);

void vm_trace(U32 level);
void vm_show_irep(U8 *irep_img);

#ifdef __cplusplus
}
#endif
#endif

