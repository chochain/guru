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
    guru_sym   	sym_id;	    //!< symbol ID as key.
    GV 			value;	    //!< stored value.
} guru_store_data;

//================================================================
/*! Define instance data handle.
*/
typedef struct RStore {
    uint32_t     size : 16;	//!< data buffer size.
    uint32_t     n    : 16;	//!< # of object stored.
    guru_store_data *data;	//!< pointer to allocated memory.
} guru_store;

__GURU__ guru_obj guru_store_new(guru_class *cls, U32 size);
__GURU__ void     guru_store_delete(GV *v);
__GURU__ void     guru_store_set(guru_obj *obj, guru_sym sid, GV *v);
__GURU__ guru_obj guru_store_get(guru_obj *obj, guru_sym sid);

#ifdef __cplusplus
}
#endif
#endif
