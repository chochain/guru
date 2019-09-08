/*! @file
  @brief
  mruby/c Range object

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_C_RANGE_H_
#define GURU_SRC_C_RANGE_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EXCLUDE_END				0x1
#define IS_EXCLUDE_END(rng)		((rng)->flag & EXCLUDE_END)

//================================================================
/*!@brief
  Define Range object (same the handles of other objects)
*/
typedef struct RRange {
    GURU_HDR;
    GV first;
    GV last;
} guru_range;

__GURU__ GV 		guru_range_new(GV *first, GV *last, int exclude_end);
__GURU__ void       guru_range_del(GV *v);
__GURU__ int        guru_range_cmp(const GV *v1, const GV *v2);

__GURU__ void       guru_init_class_range();

#ifdef __cplusplus
}
#endif
#endif
