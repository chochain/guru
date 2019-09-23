/*! @file
  @brief
  GURU Integer and Float class

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef GURU_SRC_C_NUMERIC_H_
#define GURU_SRC_C_NUMERIC_H_

#include "vm_config.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void guru_init_class_int();
__GURU__ void guru_init_class_float();

// cross module functions (called by Object#to_s)
__GURU__ void int_to_s(GV v[], U32 argc);

#ifdef __cplusplus
}
#endif
#endif
