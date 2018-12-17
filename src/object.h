/*! @file
  @brief

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef GURU_SRC_OBJECT_H_
#define GURU_SRC_OBJECT_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GPU__  void		guru_class_init(void);
__GURU__ mrbc_value mrbc_send(mrbc_value v[], mrbc_value *rcv, const char *method, int argc, ...);

#ifdef __cplusplus
}
#endif
#endif
