/*! @file
  @brief
  GURU - object store

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

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
    GS  sid;	    		//!< symbol ID as key.
    GV 	val;	    		//!< stored value.
} guru_store_data;

//================================================================
/*! Define instance data handle.
*/
typedef struct RStore {
    U32  size : 16;			//!< data buffer size.
    U32  n    : 16;			//!< # of object stored.
    guru_store_data *data;	//!< pointer to allocated memory.
} guru_store;

__GURU__ guru_obj guru_store_new(guru_class *cls, U32 size);
__GURU__ void     guru_store_delete(GV *v);
__GURU__ void     guru_store_set(guru_obj *obj, GS sid, GV *v);
__GURU__ guru_obj guru_store_get(guru_obj *obj, GS sid);

#ifdef __cplusplus
}
#endif
#endif
