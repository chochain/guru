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

__GURU__ GV 		guru_inspect(GV v[], GV *obj);				// inspect obj using v[] as stack
__GURU__ GV 		guru_kind_of(GV v[]);						// whether v1 is a kind of v0
__GURU__ void     	guru_obj_del(GV *v);						// a facade to ostore_del

// cross module c-function call (a hack for now)
__GURU__ void		obj_new(GV v[], U32 vi);
__GURU__ void		prc_call(GV v[], U32 vi);

#ifdef __cplusplus
}
#endif
#endif
