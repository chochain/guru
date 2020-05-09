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

#define RANGE_INCLUDE		0x1
#define IS_INCLUDE(rng)		((rng)->kt & RANGE_INCLUDE)

//================================================================
/*!@brief
  Define Range object (same header of other objects)
  n : used as include flag
*/
typedef struct RRange {		// 48-byte
    GURU_HDR;				// 16-byte
    GR 	first;				// 16-byte
    GR 	last;				// 16-byte
} guru_range;

__GURU__ GR 		guru_range_new(GR *first, GR *last, int exclude_end);
__GURU__ void       guru_range_del(GR *r);
__GURU__ int        guru_range_cmp(const GR *r0, const GR *r1);

__GURU__ void       guru_init_class_range();

#ifdef __cplusplus
}
#endif
#endif
