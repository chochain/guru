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
#define IS_INCLUDE(rng)		((rng)->flag & RANGE_INCLUDE)

//================================================================
/*!@brief
  Define Range object (same header of other objects)
*/
typedef struct RRange {		// 48-byte
    GURU_HDR;				// 8-byte
    U32	flag;
    U32 temp;
    GV 	first;				// 16-byte
    GV 	last;				// 16-byte
} guru_range;

__GURU__ GV 		guru_range_new(GV *first, GV *last, int exclude_end);
__GURU__ void       guru_range_del(GV *v);
__GURU__ int        guru_range_cmp(const GV *v0, const GV *v1);

__GURU__ void       guru_init_class_range();

#ifdef __cplusplus
}
#endif
#endif
