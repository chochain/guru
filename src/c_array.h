/*! @file
  @brief
  GURU Array class

  <pre>
  Copyright (C) 2019- GreenII

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
typedef struct RArray {	// 16-byte
    GURU_HDR;			// reference count
    GR 	 *data;			//!< pointer to allocated memory.
} guru_array, guru_lambda;

__GURU__ GR 		guru_array_new(U32 sz);
__GURU__ void       guru_array_del(GR *ary);

__GURU__ void		guru_array_resize(guru_array *h, U32 new_sz);
__GURU__ void       guru_array_push(GR *ary, GR *set_val);
__GURU__ void       guru_array_clr(GR *ary);
__GURU__ S32        guru_array_cmp(const GR *a0, const GR *a1);

__GURU__ void       guru_init_class_array();

#ifdef __cplusplus
}
#endif
#endif
