/*! @file
  @brief
  GURU - object store, object implementation

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_OSTORE_H_
#define GURU_SRC_OSTORE_H_
#include "guru.h"
#include "class.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ GV 	ostore_new(guru_class *cls);
__GURU__ void   ostore_del(GV *v);
__GURU__ void   ostore_set(GV *v, GS vid, GV *val);
__GURU__ GV 	ostore_get(GV *v, GS vid);

#ifdef __cplusplus
}
#endif
#endif
