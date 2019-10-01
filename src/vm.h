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

#if GURU_HOST_IMAGE
//================================================================
/*!@brief
  IREP Internal REPresentation
*/
typedef struct RIrep {			// 32-byte
	U16 size;					// size of this IREP block
    U16 nv; 		  			//!< # of local variables
    U16 nr;						//!< # of register used
    U16 r;						//!< # of child IREP blocks (into list below)

    U32 i;						//!< # of bytecodes (by iseq below)
    U16 p;						//!< # of objects in pool (into pool below)
    U16	s;						//!< # of symbols (into sym below)

    struct RIrep  	**reps;		//!< array of child IREP's pointer.
    GV  			*pool; 		//!< array of POOL objects pointer.
    							// ISEQ block follows here, and
    							// REPS block after
} guru_irep;

//================================================================
/*!@brief
  Call information
*/
typedef struct RState {			// 24-byte
    U32 pc;						// program counter
    U32 argc;  					// num of args

    guru_class      *klass;		// current class
    GV      		*regs;		// pointer to current register (in VM register file)
    guru_irep       *irep;		// pointer to current irep block
    struct RState   *prev;		// previous state (call stack)
} guru_state;					// VM context

//================================================================
/* instructions: packed 32 bit      */
/* -------------------------------  */
/*     A:B:C:OP = 9: 9: 7: 7        */
/*      A:Bx:OP = 9:   16: 7        */
/*   A:Bz:Cz:OP = 9: 14:2: 7        */
/*        Ax:OP = 25     : 7        */
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
	U32 a : 9;			// hopefully take up 32-bits total (4-byte)
} GAR;

//================================================================
/*!@brief
  Virtual Machine
*/
typedef struct VM {				// 80-byte
    U32	id;						// allocation control
    U16	run  : 3;				// VM_STATUS_FREE, READY, RUN, HOLD
    U16	step : 1;				// for single-step debug level
    U16 depth: 4;				// reserved
    U16 err;					// reserved

    union {
        U32 bytecode;			// cached bytecode
    	struct {
    		U32 op	 : 7;		// cached opcode
    		U32 opn  : 25;		// call stack depth
    	};
    };
    GAR ar;						// argument struct

    guru_state  *state;			// VM state (callinfo) linked list
    U32			fil[2];			// reserved (for alignment)

    // TODO: pointers (for dynamic sizing), use array now for debugging
    U32			rescue[8];					// ONERR/RESCUE return stack
    U8			rcnt[MAX_REGFILE_SIZE];		// reference counters
    GV 			regfile[MAX_REGFILE_SIZE];	// registers
} guru_vm;

#else	// !GURU_HOST_IMAGE
//
// old MRBC implementation (on HOST with pointers)
//
typedef struct XIrep {
    U16 	 nlv;   		//!< # of local variables
    U16 	 nreg;			//!< # of register used
    U16 	 rlen;			//!< # of child IREP blocks (into list below)
    U16 	 ilen;			//!< # of bytecodes (by iseq below)
    U16 	 plen;			//!< # of objects in pool (into pool below)
    U16	 	 slen;			//!< # of symbols (into sym below)

    U32P     iseq;			//!< ISEQ (code) BLOCK
    U8P      sym;			//!< SYMBOL list

    mrbc_object   **pool; 	//!< array of POOL objects pointer.
    struct XIrep **list;	//!< array of child IREP's pointer.
} mrbc_irep;

typedef struct XState {
    U16        		pc;
    U16        		argc;     	// num of args
    guru_class      *klass;
    GV      *reg;
    mrbc_irep       *irep;
    struct XState  *prev;
} mrbc_state;

typedef struct XVM {
    mrbc_irep       *irep;		// pointer to IREP tree
    mrbc_state      *state;		// VM state (callinfo) linked list
    GV      regfile[MAX_REGS_SIZE];

	U32				id;
    volatile U8 	run;
    volatile U8		err;
} mrbc_vm;
#endif 	// GURU_HOST_IMAGE

#define VM_IREP(vm)    	((vm)->state->irep)
#define VM_REPS(vm,n)	((guru_irep*)VM_IREP(vm)->reps[(n)])
#define VM_VAR(vm,n)	(VM_IREP(vm)->pool[VM_IREP(vm)->s + (n)])
#define VM_STR(vm,n)	(VM_IREP(vm)->pool[VM_IREP(vm)->s + (n)])
#define VM_SYM(vm,n)    (VM_IREP(vm)->pool[(n)].i)

#if GURU_HOST_IMAGE
#define VM_ISEQ(vm)	 	((U32*)U8PADD(VM_IREP(vm), sizeof(guru_irep)))
#define VM_BYTECODE(vm) (_bin_to_u32(VM_ISEQ(vm) + (vm)->state->pc))
#else  // !GURU_HOST_IMAGE
#define VM_ISEQ(vm)	 	 (VM_IREP(vm)->iseq)
#define VM_BYTECODE(vm) (bin_to_u32(U8PADD(VM_ISEQ(vm), sizeof(U32) * (vm)->state->pc)))
#endif

#ifdef __cplusplus
}
#endif
#endif
