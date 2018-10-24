/*! @file
  @brief
  mrubyc memory management.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in mruby/c.

  </pre>
*/

#ifndef MRBC_SRC_INSTANCE_H_
#define MRBC_SRC_INSTANCE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ mrbc_value mrbc_instance_new(mrbc_class *cls, int size);
__GURU__ void       mrbc_instance_delete(mrbc_value *v);
__GURU__ void       mrbc_instance_setiv(mrbc_object *obj, mrbc_sym sym_id, mrbc_value *v);
__GURU__ mrbc_value mrbc_instance_getiv(mrbc_object *obj, mrbc_sym sym_id);

#ifdef __cplusplus
}
#endif
#endif
