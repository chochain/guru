/*! @file
  @brief
  mruby bytecode executor.

  <pre>
  Copyright (C) 2015-2017 Kyushu Institute of Technology.
  Copyright (C) 2015-2017 Shimane IT Open-Innovation Center.

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

cudaError_t guru_vm_setup(guru_ses *ses, U32 trace);
cudaError_t guru_vm_run(guru_ses *ses, U32 trace);
cudaError_t guru_vm_release(guru_ses *ses, U32 trace);
cudaError_t guru_vm_trace(U32 level);

#ifdef __cplusplus
}
#endif
#endif

