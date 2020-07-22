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

__GURU__ GR 	ostore_new(GP ns);
__GURU__ void   ostore_del(guru_obj *o);
__GURU__ void   ostore_set(guru_obj *o, GS oid, GR *val);
__GURU__ GR 	ostore_get(guru_obj *o, GS oid);
__GURU__ GR		ostore_getcv(guru_obj *o, GS oid);

#ifdef __cplusplus
}
#endif
#endif // GURU_SRC_OSTORE_H_
