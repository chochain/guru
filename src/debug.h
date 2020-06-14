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
typedef union {
    U32 rbcode;				// cached bytecode (rotated op to MSBs due to nvcc bit-field limitation)
    struct {
    	U32 ax: 25, op: 7;	// call stack depth
    };
	struct {
		union {
			U16 bx;
			struct {
				U16 c : 7, b : 9;
			};
			struct {
				U16 cz: 2, bz: 14;
			};
		};
		U16 a : 9;
	};
} GAR;

void debug_init(U32 flag);
void debug_mmu_stat();
void debug_vm_irep(guru_vm *vm);
void debug_disasm(guru_vm *vm);
void debug_error(int ec);
void debug_log(const char *msg);

#ifdef __cplusplus
}
#endif

#if GURU_CXX_CODEBASE
class Debug
{
	class Impl;
	Impl  *_impl;

	Debug(U32 flag);
	~Debug();

public:
	static Debug *getInstance(U32 flag);

	void mmu_stat();
	void vm_irep(guru_vm *vm);
	void disasm(guru_vm *vm);
	void err(int ec);
	void log(const char *msg);
};
#endif // GURU_CXX_CODEBASE
#endif // GURU_SRC_DEBUG_H_

