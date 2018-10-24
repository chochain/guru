/*! @file
  @brief

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef MRBC_SRC_CLASS_H_
#define MRBC_SRC_CLASS_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ mrbc_class *mrbc_get_class_by_object(mrbc_object *obj);
__GURU__ mrbc_class *mrbc_get_class_by_name(const char *name);
__GURU__ mrbc_proc  *mrbc_get_class_method(mrbc_value recv, mrbc_sym sym_id);

__GURU__ mrbc_class *mrbc_define_class(const char *name, mrbc_class *super);
__GURU__ mrbc_proc  *mrbc_proc_alloc(const char *name);
__GURU__ void       mrbc_define_method(mrbc_class *cls, const char *name, mrbc_func_t cfunc);

// common c-functions
__GURU__ void       c_proc_call(mrbc_value v[], int argc);
__GURU__ void       c_nop(mrbc_value v[], int argc);

#ifdef __cplusplus
}
#endif
#endif
