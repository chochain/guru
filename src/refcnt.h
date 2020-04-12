/*! @file
  @brief
  GURU value and macro definitions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_REFCNT_H_
#define GURU_SRC_REFCNT_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*guru_del_func)(GV *v);
__GURU__ void guru_del_func_register(GT gt, guru_del_func f);

__GURU__ GV *ref_get(GV *v);
__GURU__ GV	*ref_free(GV *v);

__GURU__ GV *ref_dec(GV *v);
__GURU__ GV *ref_inc(GV *v);

#ifdef __cplusplus
}
#endif
#endif
