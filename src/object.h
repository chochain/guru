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

__GPU__  void		guru_core_init(void);
__GPU__  void		guru_ext_init(void);

__GURU__ void     	guru_obj_del(GR *r);						// a facade to ostore_del

#ifdef __cplusplus
}
#endif
#endif
