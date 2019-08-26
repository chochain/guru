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

#ifndef GURU_SRC_STORE_H_
#define GURU_SRC_STORE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*! Define instance data.
*/
typedef struct RStoreData {
    mrbc_sym   sym_id;	    //!< symbol ID as key.
    mrbc_value value;	    //!< stored value.
} mrbc_store_data;

//================================================================
/*! Define instance data handle.
*/
typedef struct RStore {
    uint32_t     size : 16;	//!< data buffer size.
    uint32_t     n    : 16;	//!< # of stored.
    mrbc_store_data *data;	//!< pointer to allocated memory.
} mrbc_store;

__GURU__ mrbc_value mrbc_store_new(mrbc_class *cls, U32 size);
__GURU__ void       mrbc_store_delete(mrbc_value *v);
__GURU__ void       mrbc_store_set(mrbc_object *obj, mrbc_sym sid, mrbc_value *v);
__GURU__ mrbc_value mrbc_store_get(mrbc_object *obj, mrbc_sym sid);

#ifdef __cplusplus
}
#endif
#endif
