/*! @file
  @brief
  GURU VM prototypes and interfaces (using mruby IREP 1.x)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_VM_H_
#define GURU_SRC_VM_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

// VM state machine (3-bit status)
//
//       +----> FREE (init state)
//       |       |
//       |       v
//       |     READY
//       |       |
//       |       v
//       |      RUN <-----> HOLD
//       |       |
//       |       v
//       +---- STOP
//
#define VM_STATUS_FREE  	0
#define VM_STATUS_READY		1
#define VM_STATUS_STOP  	2
#define VM_STATUS_RUN   	4
#define VM_STATUS_HOLD  	(VM_STATUS_RUN|VM_STATUS_READY)

// instructions: packed 32 bit
// ----------------------------
//   A:B:C:OP = 9: 9: 7: 7
//    A:Bx:OP = 9:   16: 7
// A:Bz:Cz:OP = 9: 14:2: 7
//      Ax:OP = 25     : 7
//================================================================
/*!@brief
  Virtual Machine
*/
typedef struct {				// 16-byte header + 8*4-byte rescue + 256*8-byte regfile
    U16	id;						// allocation control
    U16	run  : 3;				// VM_STATUS_FREE, READY, RUN, HOLD
    U16	step : 1;				// for single-step debug level
    U16 xcp  : 4;				// exception stack depth
    U16 err  : 8;				// error code

    union {
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
    };
    GP  regfile;
    GP  state;					// VM state (callinfo) linked list
} guru_vm;

#define RESCUE_PUSH(vm, pc)	(*(U32*)MEMPTR((vm)->regfile + sizeof(GR)*(VM_REGFILE_SIZE - ++(vm)->xcp)) = (pc))
#define RESCUE_POP(vm)		(*(U32*)MEMPTR((vm)->regfile + sizeof(GR)*(VM_REGFILE_SIZE - (vm)->xcp--)))

#define VM_STATE(vm)	((guru_state*)MEMPTR((vm)->state))
#define VM_IREP(vm)    	((guru_irep*)MEMPTR(VM_STATE(vm)->irep))
#define VM_ISEQ(vm)	 	((U32*)IREP_ISEQ(VM_IREP(vm)))

#define VM_REPS(vm,n)	(&IREP_REPS(VM_IREP(vm))[(n)])
#define VM_VAR(vm,n)	(&IREP_POOL(VM_IREP(vm))[(n)])
#define VM_STR(vm,n)	(&IREP_POOL(VM_IREP(vm))[(n)])
#define VM_SYM(vm,n)    ((IREP_POOL(VM_IREP(vm))[VM_IREP(vm)->p+(n)]).i)

#ifdef __cplusplus
}
#endif

#if GURU_CXX_CODEBASE
class VM
{
public:
    U16	id;						// allocation control
    U16	run  : 3;				// VM_STATUS_FREE, READY, RUN, HOLD
    U16	step : 1;				// for single-step debug level
    U16 depth: 4;				// exception stack depth
    U16 err  : 8;				// error code

    union {
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
    };

    GP  state;					// VM state (callinfo) linked list
    U32 xxx;
    U32 rescue[VM_RESCUE_STACK];
    GR 	regfile[VM_REGFILE_SIZE];	// registers

    __GURU__ void  	init(int i, int step);
    __GURU__ void  	prep(U8 *u8_gr);
    __GURU__ void  	exec();

private:
    __GURU__ void 	_transcode(U8 *u8_gr);
    __GURU__ void 	_ready(GP irep);
};
#endif // GURU_CXX_CODEBASE

#endif // GURU_SRC_VM_H_
