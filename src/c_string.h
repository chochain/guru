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
__GURU__ GV 		guru_str_new(const U8 *str);				// U8P will require to many casting
__GURU__ void       guru_str_delete(GV *s);
__GURU__ GV 		guru_str_add(const GV *v0, const GV *v1);
__GURU__ void       guru_str_append(const GV *v0, const GV *v1);
__GURU__ void       guru_str_append_cstr(const GV *v0, const U8 *str);

__GURU__ void       mrbc_init_class_string(void);

#ifdef __cplusplus
}
#endif
#endif
