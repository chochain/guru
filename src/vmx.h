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

cudaError_t guru_vm_setup(guru_ses *ses, U32 step);
cudaError_t guru_vm_run(guru_ses *ses);
cudaError_t guru_vm_release(guru_ses *ses);

#ifdef __cplusplus
}
#endif
#endif

