/*! @file
  @brief
  mruby/c String object

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_C_STRING_H_
#define GURU_SRC_C_STRING_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  Define String handle.
*/
__GURU__ mrbc_value mrbc_string_new(const U8 *str);				// U8P will require to many casting
__GURU__ void       mrbc_string_delete(mrbc_value *s);
__GURU__ mrbc_value mrbc_string_add(const mrbc_value *v0, const mrbc_value *v1);
__GURU__ void       mrbc_string_append(const mrbc_value *v0, const mrbc_value *v1);
__GURU__ void       mrbc_string_append_cstr(const mrbc_value *v0, const U8 *str);

__GURU__ void       mrbc_init_class_string(void);

#ifdef __cplusplus
}
#endif
#endif
