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
__GURU__ mrbc_value mrbc_send(mrbc_value v[], mrbc_value *rcv, const U8P method, U32 argc, ...);

__GURU__ mrbc_value guru_inspect(mrbc_value v[], mrbc_value *rcv);		// inspect
__GURU__ mrbc_value guru_kind_of(mrbc_value v[], U32 argc);				// whether v1 is a kind of v0

#ifdef __cplusplus
}
#endif
#endif
