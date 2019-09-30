/*! @file
  @brief
  GURU - object store

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_OSTORE_H_
#define GURU_SRC_OSTORE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*! Define instance data.
*/
typedef struct RStoreData {
    GS  sid;	    		//!< symbol ID as key. u16
    U16 acl;				// reserved
    GV 	val;	    		//!< stored value.
} guru_odata;

//================================================================
/*! Define instance data handle.
*/
typedef struct RStore {
    U32  rc;
    U16  size;				//!< data buffer size.
    U16  n;					//!< # of object stored.
    guru_odata *data;		//!< pointer to allocated memory.
} guru_ostore;

__GURU__ guru_obj ostore_new(guru_class *cls, U32 size);
__GURU__ void     ostore_del(GV *v);
__GURU__ void     ostore_set(guru_obj *obj, GS sid, GV *v);
__GURU__ guru_obj ostore_get(guru_obj *obj, GS sid);

#ifdef __cplusplus
}
#endif
#endif
