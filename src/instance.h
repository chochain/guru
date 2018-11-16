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

#ifndef GURU_SRC_INSTANCE_H_
#define GURU_SRC_INSTANCE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*! Define Key-Value data.
*/
typedef struct RKeyValueData {
    mrbc_sym   sym_id;	    //!< symbol ID as key.
    mrbc_value value;	    //!< stored value.
} mrbc_kv_data;

//================================================================
/*! Define Key-Value handle.
*/
typedef struct RKeyValue {
    uint16_t     size;		//!< data buffer size.
    uint16_t     n;	    	//!< # of stored.
    mrbc_kv_data *data;		//!< pointer to allocated memory.
} mrbc_kv;

__GURU__ mrbc_value mrbc_instance_new(mrbc_class *cls, int size);
__GURU__ void       mrbc_instance_delete(mrbc_value *v);
__GURU__ void       mrbc_instance_setiv(mrbc_object *obj, mrbc_sym sym_id, mrbc_value *v);
__GURU__ mrbc_value mrbc_instance_getiv(mrbc_object *obj, mrbc_sym sym_id);

#ifdef __cplusplus
}
#endif
#endif
