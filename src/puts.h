/*! @file
  @brief
  GURU puts functions for Object, Proc, Nil, False and True class and class specific functions.

  guru_config.h#GURU_USE_CONSOLE can switch between CUDA or internal implementation
  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#ifndef GURU_SRC_PUTS_H_
#define GURU_SRC_PUTS_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void guru_puts(GR r[], U32 ri);
__GURU__ void guru_p(GR r[], U32 ri);

#ifdef __cplusplus
}
#endif
#endif
