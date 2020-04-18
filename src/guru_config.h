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

/* min, maximum number of VMs */
#define MIN_VM_COUNT 		2
#define MAX_VM_COUNT 		16

/* maximum size of exception stack and registers, which determine how deep call stack can go */
#define MAX_RESCUE_STACK 	8
#define MAX_REGFILE_SIZE 	128

/* max objects in symbol and global/constant caches allowed */
#define MAX_SYMBOL_COUNT 	256
#define MAX_GLOBAL_COUNT 	64

/* Guru module support */
#define GURU_USE_STRING 	1
#define GURU_USE_CONSOLE	0

#define GURU_USE_FLOAT  	1
#define GURU_USE_ARRAY  	1
#define GURU_USE_MATH   	0

/* 32it alignment is required */
/* 0: Byte alignment */
/* 1: 32bit alignment */
#define GURU_32BIT_ALIGN_REQUIRED 	1
#define GURU_HOST_IMAGE				0
#define GURU_DEBUG					1
#define BLOCK_MEMORY_SIZE 			(128*1024)
#define GURU_64BIT_ALIGN_REQUIRED 	1

#define CC_DEBUG					0

/* CUDA dependent flags */
#define CUDA_MIN_MEMBLOCK_SIZE		0x200
#define CUDA_PROFILE_CDP            0

#endif




