/*! @file
  @brief
  GURU Inspection functions for each classes

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_INSPECT_H_
#define GURU_SRC_INSPECT_H_

#include "guru.h"
#include "puts.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void guru_na(const U8 *msg);

__GURU__ void nil_inspect(GV v[], U32 argc);
__GURU__ void nil_to_s(GV v[], U32 argc);
__GURU__ void false_to_s(GV v[], U32 argc);
__GURU__ void true_to_s(GV v[], U32 argc);

__GURU__ void int_chr(GV v[], U32 argc);
__GURU__ void int_to_s(GV v[], U32 argc);
__GURU__ void prc_inspect(GV v[], U32 argc);
__GURU__ void str_inspect(GV v[], U32 argc);

__GURU__ void ary_inspect(GV v[], U32 argc);
__GURU__ void ary_join(GV v[], U32 argc);
__GURU__ void hsh_inspect(GV v[], U32 argc);
__GURU__ void rng_inspect(GV v[], U32 argc);

__GURU__ void obj_to_s(GV v[], U32 argc);

#ifdef __cplusplus
}
#endif
#endif
