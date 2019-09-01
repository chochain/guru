/*! @file
  @brief
  GURU object class

  <pre>
  Copyright (C) 2019- GreenII

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

__GURU__ GV 		guru_inspect(GV v[], GV *rcv);				// inspect
__GURU__ GV 		guru_kind_of(GV v[], U32 argc);				// whether v1 is a kind of v0

// cross module c-function call (a hack for now)
__GURU__ void		c_proc_call(GV v[], U32 argc);

#ifdef __cplusplus
}
#endif
#endif
