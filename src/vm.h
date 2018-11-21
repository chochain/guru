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
#include "instance.h"

#ifdef __cplusplus
extern "C" {
#endif

#if GURU_HOST_IMAGE
//================================================================
/*!@brief
  IREP Internal REPresentation
*/
typedef struct RIrep {			// 32-bytes
    uint32_t	 size;			// size of entire IREP block
    uint16_t 	 nlv;   		//!< # of local variables
    uint16_t 	 nreg;			//!< # of register used
    uint16_t 	 rlen;			//!< # of child IREP blocks (into list below)
    uint16_t 	 ilen;			//!< # of bytecodes (by iseq below)
    uint16_t 	 plen;			//!< # of objects in pool (into pool below)
    uint16_t	 slen;			//!< # of symbols (into sym below)

    uint32_t 	 iseq;			//!< offset to ISEQ (code) BLOCK
    uint32_t	 sym;			//!< offset to array of SYMBOLs
    uint32_t     pool; 	   		//!< offset to array of POOL objects pointer.
    uint32_t 	 list;			//!< offset to array of child IREP's pointer.
} guru_irep;

//================================================================
/*!@brief
  Call information
*/

typedef struct RState {
    uint16_t        pc;
    uint16_t        argc;     	// num of args
    mrbc_class      *klass;
    mrbc_value      *reg;
    guru_irep       *irep;
    struct RState   *prev;
} guru_state;

//================================================================
/*!@brief
  Virtual Machine
*/
typedef struct VM {
    guru_irep       *irep;		// pointer to IREP tree
    guru_state      *state;		// VM state (callinfo) linked list
    mrbc_value      regfile[MAX_REGS_SIZE];

    volatile int8_t run;
    volatile int8_t	err;
} guru_vm;

#else // !GURU_HOST_IMAGE

typedef struct RIrep1 {
    uint16_t 	 nlv;   		//!< # of local variables
    uint16_t 	 nreg;			//!< # of register used
    uint16_t 	 rlen;			//!< # of child IREP blocks (into list below)
    uint16_t 	 ilen;			//!< # of bytecodes (by iseq below)
    uint16_t 	 plen;			//!< # of objects in pool (into pool below)
    uint16_t	 slen;			//!< # of symbols (into sym below)

    uint32_t     *iseq;			//!< ISEQ (code) BLOCK
    uint8_t      *sym;			//!< SYMBOL list

    mrbc_object   **pool; 	   	//!< array of POOL objects pointer.
    struct RIrep1 **list;		//!< array of child IREP's pointer.
} mrbc_irep;

typedef struct RState1 {
    uint16_t        pc;
    uint16_t        argc;     	// num of args
    mrbc_class      *klass;
    mrbc_value      *reg;
    mrbc_irep       *irep;
    struct RState   *prev;
} mrbc_state;

typedef struct VM1 {
    mrbc_irep       *irep;		// pointer to IREP tree
    mrbc_state      *state;		// VM state (callinfo) linked list
    mrbc_value      regfile[MAX_REGS_SIZE];

    volatile int8_t run;
    volatile int8_t	err;
} mrbc_vm;

#endif // GURU_HOST_IMAGE
//================================================================
/*!@brief
  Get 32bit value from memory big endian.

  @param  s	Pointer of memory.
  @return	32bit unsigned value.
*/
__GURU__ __INLINE__
uint32_t _bin_to_uint32(const void *s)
{
#if GURU_REQUIRE_32BIT_ALIGNMENT
    uint8_t *p = (uint8_t *)s;
    uint32_t x = *p++;
    x <<= 8;
    x |= *p++;
    x <<= 8;
    x |= *p++;
    x <<= 8;
    x |= *p;
    return x;
#else
    uint32_t x = *((uint32_t *)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
#endif
}

//================================================================
/*!@brief
  Get 16bit value from memory big endian.

  @param  s	Pointer of memory.
  @return	16bit unsigned value.
*/
__GURU__ __INLINE__
uint16_t _bin_to_uint16(const void *s)
{
#if GURU_REQUIRE_32BIT_ALIGNMENT
    uint8_t *p = (uint8_t *)s;
    uint16_t x = *p++ << 8;
    x |= *p;
    return x;
#else
    uint16_t x = *((uint16_t *)s);
    return (x << 8) | (x >> 8);
#endif
}

/*!@brief
  Set 16bit big endian value from memory.

  @param  s Input value.
  @param  bin Pointer of memory.
  @return sizeof(uint16_t).
*/
__GURU__ __INLINE__
void _uint16_to_bin(uint16_t s, uint8_t *bin)
{
    *bin++ = (s >> 8) & 0xff;
    *bin   = s & 0xff;
}

/*!@brief
  Set 32bit big endian value from memory.

  @param  l Input value.
  @param  bin Pointer of memory.
  @return sizeof(uint32_t).
*/
__GURU__ __INLINE__
void _uint32_to_bin(uint32_t l, uint8_t *bin)
{
    *bin++ = (l >> 24) & 0xff;
    *bin++ = (l >> 16) & 0xff;
    *bin++ = (l >> 8) & 0xff;
    *bin   = l & 0xff;
}

cudaError_t guru_vm_init(guru_ses *ses);
cudaError_t guru_vm_run(guru_ses *ses);

#define VM_IREP(vm)      ((vm)->state->irep)

#if GURU_HOST_IMAGE
#define VM_ISEQ(vm)	 	 ((uint32_t *)((uint8_t *)VM_IREP(vm) + VM_IREP(vm)->iseq))
#define GET_BYTECODE(vm) (_bin_to_uint32(VM_ISEQ(vm) + (vm)->state->pc))

#ifdef GURU_DEBUG
	void guru_dump_irep(guru_irep *irep);
	void guru_dump_regfile(guru_vm *vm, int debug);
#endif

#else  // !GURU_HOST_IMAGE

#define VM_ISEQ(vm)	 	 ((uint32_t *)(VM_IREP(vm)->iseq)
#define GET_BYTECODE(vm) (_bin_to_uint32(VM_ISEQ(vm) + (vm)->state->pc))

#ifdef GURU_DEBUG
	void guru_dump_irep(mrbc_irep *irep);
	void guru_dump_regfile(mrbc_vm *vm, int debug);
#endif

#endif	// GURU_HOST_IMAGE

#ifdef __cplusplus
}
#endif
#endif
