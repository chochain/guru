/*! @file
  @brief
  GURU global objects.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>
#include "global.h"
/*

  GLobal objects are stored in '_global' array.
  '_global' array is decending order by sid.
  In case of searching a global object, binary search is used.
  In case of adding a global object, insertion sort is used.

*/
typedef enum {
    GURU_CONST_OBJECT  = 0,				// for classes
    GURU_GLOBAL_OBJECT					// Ruby global objects
} _gtype;

typedef struct {						// 32-bit
    GS 			sid;
    _gtype 		gt;
} _gidx;

#define _LOCK		{ MUTEX_LOCK(_mutex_gobj); }
#define _UNLOCK		{ MUTEX_FREE(_mutex_gobj); }

// max of global object in _global[]
__GURU__ U32 	_mutex_gobj;

__GURU__ U32    _global_sz;
__GURU__ _gidx 	_global_idx[MAX_GLOBAL_COUNT];
__GURU__ GV 	_global[MAX_GLOBAL_COUNT];
__GURU__ GV		_nil { .gt = GT_NIL, .acl=0, .fil=0 };

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
__GURU__ S32
_idx(GS sid, _gtype gt)
{
	_gidx *p = _global_idx;
    for (U32 i=0; i <_global_sz ; i++, p++) {
        if (p->sid == sid && p->gt == gt) return i;
    }
    return -1;
}

__GURU__ GV *
_get(GS sid, _gtype gt)
{
    S32 i = _idx(sid, gt);

    if (i < 0) return &_nil;		// not found

    return &_global[i];				// pointer to global object
}

__GURU__ void
_set(GS sid, GV *v, _gtype gt)
{
    S32 i = _idx(sid, gt);

    _LOCK;

    if (i < 0) {							// not found
        i = _global_sz++;
    }
    assert(i < MAX_GLOBAL_COUNT);			// maybe raise ex

    _global_idx[i].sid = sid;
    _global_idx[i].gt  = gt;
    _global[i] = *v;

    _UNLOCK;
}

/* add */
/* TODO: Check reference count */
__GURU__ void
global_set(GS sid, GV *v)
{
    _set(sid, v, GURU_GLOBAL_OBJECT);
}

__GURU__ void
const_set(GS sid, GV *v)
{
    _set(sid, v, GURU_CONST_OBJECT);
}

/* get */
__GURU__ GV *
global_get(GS sid)
{
    return _get(sid, GURU_GLOBAL_OBJECT);
}

/* add const */
__GURU__ GV *
const_get(GS sid)
{
    return _get(sid, GURU_CONST_OBJECT);
}
//
__GPU__ void
guru_global_init(void)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	_mutex_gobj = 0;
	_global_sz  = 0;
}
