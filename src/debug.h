/*! @file
  @brief
  GURU - VM debugger interfaces

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  Fetch mruby VM bytecodes, decode and execute.

  </pre>
*/

#ifndef GURU_SRC_DEBUG_H_
#define GURU_SRC_DEBUG_H_
#include "vm.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
// instructions: packed 32 bit
// ----------------------------
//   A:B:C:OP = 9: 9: 7: 7
//    A:Bx:OP = 9:   16: 7
// A:Bz:Cz:OP = 9: 14:2: 7
//      Ax:OP = 25     : 7
typedef struct {
	union {
		U16 bx;
		struct {
			U16 c : 7, b : 9;
		};
		struct {
			U16 cz: 2, bz: 14;
		};
	};
	U16 a: 9, x: 7;				// hopefully take up 32-bits total (4-byte)
} GAR;

void debug_init(U32 flag);
void debug_mmu_stat();
void debug_vm_irep(guru_vm *vm);
int  debug_disasm(guru_vm *vm);
void debug_log(const char *msg);

#ifdef __cplusplus
}
#endif
#endif

