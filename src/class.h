/*! @file
  @brief

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef GURU_SRC_CLASS_H_
#define GURU_SRC_CLASS_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

// external methods
__GURU__ mrbc_class *mrbc_define_class(const U8P name, mrbc_class *super);
__GURU__ mrbc_proc  *mrbc_define_method(mrbc_class *cls, const U8P name, mrbc_func_t cfunc);

__GURU__ mrbc_class *mrbc_get_class_by_object(guru_obj *obj);
__GURU__ mrbc_class *mrbc_get_class_by_name(const U8P name);									// cannot use U8P, lots of casting
__GURU__ mrbc_proc  *mrbc_get_proc_by_symid(mrbc_value rcv, mrbc_sym sid);

// cross module c-functions
__GURU__ mrbc_proc 	*mrbc_alloc_proc(const U8P name);
__GURU__ void		c_object_new(mrbc_value v[], U32 argc);

#ifdef __cplusplus
}
#endif
#endif
