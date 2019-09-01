/*! @file
  @brief
  GURU class building functions

  <pre>
  Copyright (C) 2019- GreenII

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
__GURU__ guru_class *guru_define_class(const U8P name, guru_class *super);
__GURU__ guru_proc  *guru_define_method(guru_class *cls, const U8P name, guru_fptr cfunc);
__GURU__ guru_proc 	*guru_alloc_proc(const U8P name);

__GURU__ guru_class *class_by_obj(guru_obj *obj);
__GURU__ guru_proc  *proc_by_sid(GV rcv, GS sid);

// cross module c-function (a hack)
__GURU__ void 		c_object_new(GV v[], U32 argc);

#ifdef __cplusplus
}
#endif
#endif
