/*! @file
  @brief
  GURU - global configration for VM's

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef GURU_SRC_GURU_CONFIG_H_
#define GURU_SRC_GURU_CONFIG_H_

/* debugging flags
 *   guru: general debugging controlled by -t option, default 1
 *   mmu : MMU alloc/free TLSF dump, default 0
 *   cc  : for my development only, default 0
 */
#define GURU_DEBUG			1
#define MMU_DEBUG			0
#define CC_DEBUG			0

/* min, maximum number of VMs */
#define MIN_VM_COUNT 		2
#define MAX_VM_COUNT 		16

/* maximum size of exception stack and registers, which determine how deep call stack can go */
#define VM_RESCUE_STACK 	8
#define VM_REGFILE_SIZE 	128

/* max objects in symbol and global/constant caches allowed */
#define MAX_SYMBOL_COUNT 	256
#define MAX_GLOBAL_COUNT 	64

/* Guru can minimize usage for micro device */
#define GURU_USE_STRING 	1
#define GURU_USE_CONSOLE	0
#define GURU_USE_FLOAT  	1
#define GURU_USE_ARRAY  	1
#define GURU_USE_MATH   	0
/*
 * 32it alignment is required
 * 	  0: Byte alignment
 * 	  1: 32bit alignment
 * Heap size
 *    48K can barely fit utf8_02 test case
 * string buffer size
 *    for strcat and puts
 * host grit image
 *    0 : GRIT created inside GPU
 *    1 :              by CPU
 * CXX codebase
 *    0 : use C only
 *    1 : use C++ set
 */
#define GURU_32BIT_ALIGN_REQUIRED 	1
#define GURU_HEAP_SIZE 				(64*1024)
#define GURU_STRBUF_SIZE			(256-1)
#define GURU_HOST_GRIT_IMAGE		1
#define GURU_CXX_CODEBASE           0
/* CUDA dependent flags */
#define CUDA_MIN_MEMBLOCK_SIZE		0x200
#define CUDA_ENABLE_CDP            	0

#endif // GURU_SRC_GURU_CONFIG_H_




