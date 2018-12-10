/*! @file
  @brief
  Global configration of mruby/c VM's

  <pre>
  Copyright (C) 2015 Kyushu Institute of Technology.
  Copyright (C) 2015 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef GURU_SRC_VM_CONFIG_H_
#define GURU_SRC_VM_CONFIG_H_

/* min, maximum number of VMs */
#ifndef MIN_VM_COUNT
#define MIN_VM_COUNT 2
#endif
#ifndef MAX_VM_COUNT
#define MAX_VM_COUNT 5
#endif

/* maximum size of registers, which determine how deep call stack can go */
#ifndef MAX_REGS_SIZE
#define MAX_REGS_SIZE 32
#endif

/* maximum size of callinfo (callstack) */
#ifndef MAX_CALL_STACK
#define MAX_CALL_STACK 100
#endif

/* maximum number of objects */
#ifndef MAX_OBJECT_COUNT
#define MAX_OBJECT_COUNT 400
#endif

/* maximum number of classes */
#ifndef MAX_CLASS_COUNT
#define MAX_CLASS_COUNT 20
#endif

/* maximum number of symbols */
#ifndef MAX_SYMBOLS_COUNT
#define MAX_SYMBOLS_COUNT 200
#endif

/* maximum size of global objects */
#ifndef MAX_GLOBAL_OBJECT_SIZE
#define MAX_GLOBAL_OBJECT_SIZE 20
#endif

/* maximum size of consts */
#ifndef MAX_CONST_COUNT
#define MAX_CONST_COUNT 20
#endif


/* Configure environment */
/* 0: NOT USE */
/* 1: USE */

/* USE String. Support String class */
#define GURU_USE_FLOAT  1
#define GURU_USE_STRING 1
#define GURU_USE_MATH   0
#define GURU_USE_ARRAY  1

/* Hardware dependent flags */

/* 32it alignment is required */
/* 0: Byte alignment */
/* 1: 32bit alignment */
#define GURU_REQUIRE_32BIT_ALIGNMENT 	1
#define GURU_HOST_IMAGE					1

#define __GURU_CUDA__
#define GURU_DEBUG
#define BLOCK_MEMORY_SIZE 				(32*1024)
#define GURU_REQUIRE_64BIT_ALIGNMENT 	1
#endif




