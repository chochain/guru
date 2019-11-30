/*! @file
  @brief
  GURU - VM debugger interfaces

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  Fetch mruby VM bytecodes, decode and execute.

  </pre>
*/

#ifndef GURU_SRC_VM_DEBUG_H_
#define GURU_SRC_VM_DEBUG_H_

#ifdef __cplusplus
extern "C" {
#endif

void debug_show_irep(guru_irep *irep, char level, char *idx);
void debug_trace(U32 level);

#ifdef __cplusplus
}
#endif
#endif

