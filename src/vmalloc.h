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

#ifndef MRBC_SRC_VMALLOC_H_
#define MRBC_SRC_VMALLOC_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

// << from value.hu
__GURU__ mrbc_object *mrbc_obj_alloc(mrbc_vtype tt);
__GURU__ mrbc_proc *mrbc_rproc_alloc(const char *name);

__GURU__ mrbc_value mrbc_instance_new(mrbc_class *cls, int size);
__GURU__ void       mrbc_instance_delete(mrbc_value *v);
__GURU__ void       mrbc_instance_setiv(mrbc_object *obj, mrbc_sym sym_id, mrbc_value *v);
__GURU__ mrbc_value mrbc_instance_getiv(mrbc_object *obj, mrbc_sym sym_id);

__GURU__ void       mrbc_dec_ref_counter(mrbc_value *v);
__GURU__ void 		mrbc_release(mrbc_value *v);

#ifdef __cplusplus
}
#endif
#endif
