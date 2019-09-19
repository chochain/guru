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
#include "value.h"

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
typedef struct RIrep {	// 32-byte
    U32	size;			// size of entire IREP block

    S32 reps;			//!< offset to array of child IREP's pointer.
    S32 pool; 			//!< offset to array of POOL objects pointer.
    S32	sym;			//!< offset to array of SYMBOLs
    S32 iseq;			//!< offset to ISEQ (code) BLOCK

    U32 i;				//!< # of bytecodes (by iseq below)
    U16 c;				//!< # of child IREP blocks (into list below)
    U16 p;				//!< # of objects in pool (into pool below)
    U16	s;				//!< # of symbols (into sym below)
    U8  nv; 		  	//!< # of local variables
    U8  nr;				//!< # of register used
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
typedef struct VM {				// 24 + 32*reg bytes
    U32	id   : 13;				// allocation control
    U32	run  : 3;				// VM_STATUS_FREE, READY, RUN, HOLD
    U32	err	 : 8;				// error code/condition
    U32 depth: 7;				// depth of call stack
    U32	step : 1;				// for single-step debug level

    U32 op	 : 7;				// cached opcode
    U32 opn  : 25;				// call stack depth

    U32 bytecode;				// cached bytecode
    GAR ar;						// argument struct

    guru_state *state;			// VM state (callinfo) linked list
    GV regfile[MAX_REGS_SIZE];	// TODO: change to a pointer
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

//================================================================
/*!@brief
  Get 32bit value from memory big endian.

  @param  s	Pointer of memory.
  @return	32bit unsigned value.
*/
__GURU__ __INLINE__ U32
_bin_to_u32(const void *s)
{
#if GURU_32BIT_ALIGN_REQUIRED
    U8P p = (U8P)s;
    return (U32)(p[0]<<24) | (p[1]<<16) |  (p[2]<<8) | p[3];
#else
    U32 x = *((U32P)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
#endif
}

//================================================================
/*!@brief
  Get 16bit value from memory big endian.

  @param  s	Pointer of memory.
  @return	16bit unsigned value.
*/
__GURU__ __INLINE__ U16
_bin_to_u16(const void *s)
{
#if GURU_32BIT_ALIGN_REQUIRED
    U8P p = (U8P)s;
    return (U16)(p[0]<<8) | p[1];
#else
    U16 x = *((U16P)s);
    return (x << 8) | (x >> 8);
#endif
}

/*!@brief
  Set 16bit big endian value from memory.

  @param  s Input value.
  @param  bin Pointer of memory.
  @return sizeof(U16).
*/
__GURU__ __INLINE__ void
_u16_to_bin(U16 s, U8P bin)
{
    *bin++ = (s >> 8) & 0xff;
    *bin   = s & 0xff;
}

/*!@brief
  Set 32bit big endian value from memory.

  @param  l Input value.
  @param  bin Pointer of memory.
  @return sizeof(U32).
*/
__GURU__ __INLINE__ void
_u32_to_bin(U32 l, U8P bin)
{
    *bin++ = (l >> 24) & 0xff;
    *bin++ = (l >> 16) & 0xff;
    *bin++ = (l >> 8) & 0xff;
    *bin   = l & 0xff;
}

#define VM_IREP(vm)    	((vm)->state->irep)
#define VM_SYM(vm,n)    (U8PADD(VM_IREP(vm), *(U32*)U8PADD(VM_IREP(vm), VM_IREP(vm)->sym  + n*sizeof(U32))))
#define VM_VAR(vm,n)	((U32P)U8PADD(VM_IREP(vm), *(U32*)U8PADD(VM_IREP(vm), VM_IREP(vm)->pool + n*sizeof(U32))))
#define VM_REPS(vm,n)	((guru_irep*)U8PADD(VM_IREP(vm), *(U32*)U8PADD(VM_IREP(vm), VM_IREP(vm)->reps + n*sizeof(U32))))

#if GURU_HOST_IMAGE
#define VM_ISEQ(vm)	 	 (U8PADD(VM_IREP(vm), VM_IREP(vm)->iseq))
#define VM_BYTECODE(vm) (_bin_to_u32(U8PADD(VM_ISEQ(vm), sizeof(U32) * (vm)->state->pc)))
#else  // !GURU_HOST_IMAGE
#define VM_ISEQ(vm)	 	 (VM_IREP(vm)->iseq)
#define VM_BYTECODE(vm) (bin_to_u32(U8PADD(VM_ISEQ(vm), sizeof(U32) * (vm)->state->pc)))
#endif

#ifdef __cplusplus
}
#endif
#endif
