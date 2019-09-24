/*! @file
  @brief
  GURU puts functions for Object, Proc, Nil, False and True class and class specific functions.

  vm_config.h#GURU_USE_CONSOLE can switch between CUDA or internal implementation
  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#ifndef GURU_SRC_PUTS_H_
#define GURU_SRC_PUTS_H_

#include "guru.h"

#if GURU_USE_CONSOLE	// use built-in print functions
#include "console.h"
#include "sprintf.h"
#define PRINTF		guru_printf
#define VPRINTF		guru_vprintf
#else					// use CUDA printf function
#include <stdio.h>
#define PRINTF		printf
#endif

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void guru_puts(GV *v, U32 argc);
__GURU__ void guru_p(GV *v, U32 argc);

#ifdef __cplusplus
}
#endif
#endif
