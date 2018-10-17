/*! @file
  @brief
  If you want to add your own extension,
  add your code in c_ext.c and c_ext.h. 

  <pre>
  Copyright (C) 2015 Kyushu Institute of Technology.
  Copyright (C) 2015 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef MRBC_SRC_C_EXT_H_
#define MRBC_SRC_C_EXT_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IREP {
    uint16_t 	nlocals;   	//!< # of local variables
    uint16_t 	nregs;		//!< # of register variables
    uint16_t 	rlen;		//!< # of child IREP blocks
    uint16_t 	ilen;		//!< # of irep
    uint16_t 	plen;		//!< # of pool

    uint8_t     *code;		//!< ISEQ (code) BLOCK
    mrbc_object **pools;    //!< array of POOL objects pointer.
    uint8_t     *ptr_to_sym;
    struct IREP **reps;		//!< array of child IREP's pointer.
} mrbc_irep;

typedef struct CALLINFO {
    struct CALLINFO *prev;
    mrbc_irep       *pc_irep;
    uint16_t        pc;
    mrbc_value      *current_regs;
    mrbc_class      *target_class;
    uint8_t         n_args;     // num of args
} mrbc_callinfo;

typedef struct VM {
    mrbc_irep      *irep;

    uint8_t        vm_id; 		// vm_id: (1..vm_config.MAX_VM_COUNT)
    const uint8_t  *mrb;   		// bytecode

    mrbc_irep      *pc_irep;    // PC
    uint16_t       pc;         	// PC

    //  uint16_t   reg_top;
    mrbc_value     regs[MAX_REGS_SIZE];
    mrbc_value     *current_regs;
    mrbc_callinfo  *callinfo_tail;

    mrbc_class     *target_class;

    int32_t        error_code;

    volatile int8_t flag_preemption;
    volatile int8_t flag_need_memfree;
} mrbc_vm;

void guru_init_ext(mrbc_vm *vm, char *fname);

#ifdef __cplusplus
}
#endif
#endif
