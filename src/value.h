/*! @file
  @brief
  Guru value definitions

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef MRBC_SRC_VALUE_H_
#define MRBC_SRC_VALUE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ int        mrbc_compare(const mrbc_value *v1, const mrbc_value *v2);
__GURU__ void       mrbc_dup(mrbc_value *v);
__GURU__ mrbc_int   mrbc_atoi(const char *s, int base);

#ifdef __cplusplus
}
#endif
#endif
