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

void debug_init(U32 flag);
void debug_mmu_stat();
void debug_vm_irep(S32 vm_id);
void debug_disasm(S32 vm_id);
void debug_log(const char *msg);

#ifdef __cplusplus
}
#endif
#endif

