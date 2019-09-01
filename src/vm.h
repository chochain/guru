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

#ifndef GURU_SRC_VM_H_
#define GURU_SRC_VM_H_

#include "guru.h"
#include "value.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  IREP Internal REPresentation
*/
typedef struct RIrep {	// 32-byte
    U32	size;			// size of entire IREP block
    U32 ilen;			//!< # of bytecodes (by iseq below)

    U32 plen	: 16;	//!< # of objects in pool (into pool below)
    U32	slen	: 16;	//!< # of symbols (into sym below)
    U32 rlen	: 16;	//!< # of child IREP blocks (into list below)
    U32 nlv		: 8;   	//!< # of local variables
    U32 nreg	: 8;	//!< # of register used

    U32 iseq;			//!< offset to ISEQ (code) BLOCK
    U32	sym;			//!< offset to array of SYMBOLs
    U32 pool; 			//!< offset to array of POOL objects pointer.
    U32 list;			//!< offset to array of child IREP's pointer.
} guru_irep;

//================================================================
/*!@brief
  Call information
*/

typedef struct RState {			// 24-byte
    U32 pc;						// program counter
    U32 argc;  					// num of args

    guru_class      *klass;		// current class
    GV      		*reg;		// pointer to current register (in VM register file)
    guru_irep       *irep;		// pointer to current irep block
    struct RState   *prev;		// previous state (call stack)
} guru_state;					// VM context

//================================================================
/*!@brief
  Virtual Machine
*/
typedef struct VM {				// 12 + 32*reg bytes
    U32	id   : 14;				// allocation control (0 means free)
    U32	step : 1;				// for single-step debug level
    U32	run  : 1;				// to exit vm loop
    U32	err  : 16;				// error code/condition

    guru_irep  *irep;			// pointer to IREP tree
    guru_state *state;			// VM state (callinfo) linked list

    GV regfile[MAX_REGS_SIZE];
} guru_vm;

#if !GURU_HOST_IMAGE
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
#endif 	// !GURU_HOST_IMAGE
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

#define VM_IREP(vm)      ((vm)->state->irep)

#if GURU_HOST_IMAGE
#define VM_ISEQ(vm)	 	 ((U32P)U8PADD(VM_IREP(vm), VM_IREP(vm)->iseq))
#define GET_BYTECODE(vm) (_bin_to_u32(U8PADD(VM_ISEQ(vm), sizeof(U32) * (vm)->state->pc)))
#else  // !GURU_HOST_IMAGE
#define VM_ISEQ(vm)	 	 (((U32P)VM_IREP(vm)->iseq)
#define GET_BYTECODE(vm) (_bin_to_u32(U8PADD(VM_ISEQ(vm), sizeof(U32) * (vm)->state->pc)))
#endif

#ifdef __cplusplus
}
#endif
#endif
