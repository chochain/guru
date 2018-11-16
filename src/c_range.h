/*! @file
  @brief
  mruby/c Range object

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

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
    MRBC_OBJECT_HEADER;

    mrbc_value first;
    mrbc_value last;
} mrbc_range;

__GURU__ mrbc_value mrbc_range_new(mrbc_value *first, mrbc_value *last, int exclude_end);

__GURU__ void       mrbc_range_delete(mrbc_value *v);
__GURU__ int        mrbc_range_compare(const mrbc_value *v1, const mrbc_value *v2);

__GURU__ void       mrbc_init_class_range();

#ifdef __cplusplus
}
#endif
#endif
