/*! @file
  @brief
  mruby/c String object

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef MRBC_SRC_C_STRING_H_
#define MRBC_SRC_C_STRING_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  Define String handle.
*/
__GURU__ mrbc_value mrbc_string_new(const void *src, int len);
__GURU__ mrbc_value mrbc_string_new_cstr(const char *src);
__GURU__ mrbc_value mrbc_string_new_alloc(void *buf, int len);
__GURU__ void       mrbc_string_delete(mrbc_value *str);
__GURU__ void       mrbc_string_clear_vm_id(mrbc_value *str);
__GURU__ mrbc_value mrbc_string_dup(mrbc_value *s1);
__GURU__ mrbc_value mrbc_string_add(const mrbc_value *s1, const mrbc_value *s2);

__GURU__ int        mrbc_string_append(mrbc_value *s1, const mrbc_value *s2);
__GURU__ int        mrbc_string_append_cstr(mrbc_value *s1, const char *s2);
__GURU__ int        mrbc_string_index(const mrbc_value *src, const mrbc_value *pattern, int offset);
__GURU__ int        mrbc_string_strip(mrbc_value *src, int mode);
__GURU__ int        mrbc_string_chomp(mrbc_value *src);

__GURU__ void       mrbc_init_class_string(void);

#ifdef __cplusplus
}
#endif
#endif
