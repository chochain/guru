/*! @file
  @brief
  mruby/c Array class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_C_ARRAY_H_
#define GURU_SRC_C_ARRAY_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  Define Array handle.
*/
typedef struct RArray {
    GURU_HDR;

    U32  size : 16;	//!< data buffer size.
    U32  n    : 16;	//!< # of stored.
    GV 	 *data;		//!< pointer to allocated memory.

} guru_array;

__GURU__ GV 		guru_array_new(int size);
__GURU__ void       guru_array_delete(GV *ary);

__GURU__ int		guru_array_resize(guru_array *h, int size);
__GURU__ int        guru_array_push(GV *ary, GV *set_val);
__GURU__ void       guru_array_clear(GV *ary);
__GURU__ int        guru_array_compare(const GV *v1, const GV *v2);

__GURU__ void       guru_init_class_array();

#ifdef __cplusplus
}
#endif
#endif
