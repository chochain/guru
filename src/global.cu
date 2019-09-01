/*! @file
  @brief
  GURU global objects.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>
#include "value.h"
#include "global.h"
/*

  GLobal objects are stored in '_global' array.
  '_global' array is decending order by sid.
  In case of searching a global object, binary search is used.
  In case of adding a global object, insertion sort is used.

*/
typedef enum {
    GURU_GLOBAL_OBJECT = 1,
    GURU_CONST_OBJECT,
} _gtype;

typedef struct _gobj_ {
    GS 			sid;
    guru_obj 	obj;
    _gtype 		gt	:8;
} _gobj;

// max of global object in _global[]
__GURU__ U32 	_mutex_gobj;

__GURU__ U32 	_global_idx;
__GURU__ _gobj 	_global[MAX_GLOBAL_COUNT];

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
__GURU__ U32
_get_idx(GS sid, _gtype gt)
{
    for (U32 i=0; i < _global_idx ; i++) {
        _gobj *obj = &_global[i];
        if (obj->sid == sid && obj->gt == gt) return i;
    }
    return MAX_GLOBAL_COUNT;
}

__GURU__ guru_obj
_get_obj(GS sid, _gtype gt)
{
    U32 idx = _get_idx(sid, gt);

    if (idx==MAX_GLOBAL_COUNT) return GURU_NIL_NEW();

    ref_inc(&_global[idx].obj);

    return _global[idx].obj;				// pointer to global object
}

__GURU__ void
_add_obj(GS sid, guru_obj *obj, _gtype gt)
{
    int idx = _get_idx(sid, gt);

    MUTEX_LOCK(_mutex_gobj);

    if (idx < MAX_GLOBAL_COUNT) {
        ref_clr(&(_global[idx].obj));
     }
    else {
        idx = ++_global_idx;
         assert(idx < MAX_GLOBAL_COUNT);	// maybe raise ex
    }
    _global[idx].sid = sid;
    _global[idx].obj = *obj;				// deep copy
    _global[idx].gt  = gt;

    MUTEX_FREE(_mutex_gobj);

    ref_inc(obj);
}

/* add */
/* TODO: Check reference count */
__GURU__ void
global_object_add(GS sid, guru_obj *obj)
{
    _add_obj(sid, obj, GURU_GLOBAL_OBJECT);
}

__GURU__ void
const_object_add(GS sid, guru_obj *obj)
{
    _add_obj(sid, obj, GURU_CONST_OBJECT);
}

/* get */
__GURU__ guru_obj
global_object_get(GS sid)
{
    return _get_obj(sid, GURU_GLOBAL_OBJECT);
}

/* add const */
__GURU__ guru_obj
const_object_get(GS sid)
{
    return _get_obj(sid, GURU_CONST_OBJECT);
}
//
__GPU__ void
guru_global_init(void)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	_mutex_gobj = 0;
	_global_idx = 0;
}




