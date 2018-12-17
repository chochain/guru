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
    GURU_OBJECT_HEADER;

    uint16_t   size;	//!< data buffer size.
    uint16_t   n;	    //!< # of stored.
    mrbc_value *data;	//!< pointer to allocated memory.

} mrbc_array;

__GURU__ mrbc_value mrbc_array_new(int size);
__GURU__ void       mrbc_array_delete(mrbc_value *ary);

__GURU__ int		mrbc_array_resize(mrbc_array *h, int size);
__GURU__ int        mrbc_array_push(mrbc_value *ary, mrbc_value *set_val);
__GURU__ void       mrbc_array_clear(mrbc_value *ary);
__GURU__ int        mrbc_array_compare(const mrbc_value *v1, const mrbc_value *v2);

__GURU__ void       mrbc_init_class_array();

#ifdef __cplusplus
}
#endif
#endif
