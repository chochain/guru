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
#include "puts.h"
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
    GS 			xid;
    _gtype 		gt;
} _gidx;

#define _LOCK		{ MUTEX_LOCK(_mutex_gobj); }
#define _UNLOCK		{ MUTEX_FREE(_mutex_gobj); }

// max of global object in _global[]
__GURU__ U32 	_mutex_gobj = 0;
__GURU__ U32    _global_sz  = 0;
__GURU__ _gidx 	_global_idx[MAX_GLOBAL_COUNT];
__GURU__ GV 	_global[MAX_GLOBAL_COUNT];
__GURU__ GV		_nil { .gt = GT_NIL, .acl=0 };

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
__GPU__ void
__idx(S32 *idx, GS xid, _gtype gt)
{
	S32    i = threadIdx.x;
	_gidx *p = _global_idx + i;

	if (i<_global_sz && p->xid == xid && p->gt == gt) *idx = i;

	__syncthreads();		// sync all thread in the block (make sure idx is captured)
}

__GURU__ S32
_find_idx(GS xid, _gtype gt)
{
	static S32 idx;			// warn: scoped outside of function
	idx = -1;
	__idx<<<1, 32*(1+(_global_sz>>5))>>>(&idx, xid, gt);
	DEVSYNC();

	return idx;
}

__GURU__ GV *
_get(GS xid, _gtype gt)
{
	S32 i = _find_idx(xid, gt);
    if (i < 0) return &_nil;		// not found

    return &_global[i];				// pointer to global object
}

__GURU__ void
_set(GS xid, GV *v, _gtype gt)
{
    S32 i = _find_idx(xid, gt);

    _LOCK;
    if (i<0) {
    	i = _global_sz++;
    	assert(i<MAX_GLOBAL_COUNT);	// maybe raise ex
    }
    else {
//    	assert(i<0);
    	assert(1==1);
    }
    _global_idx[i].xid = xid;
    _global_idx[i].gt  = gt;
    _global[i] = *v;
    _UNLOCK;

#if CC_DEBUG
    printf("G[%d]=", i);	guru_puts(v, 1);
#endif // CC_DEBUG
}

/* add */
/* TODO: Check reference count */
__GURU__ void
global_set(GS xid, GV *v)
{
    _set(xid, v, GURU_GLOBAL_OBJECT);
}

__GURU__ void
const_set(GS xid, GV *v)
{
    _set(xid, v, GURU_CONST_OBJECT);
}

/* get */
__GURU__ GV *
global_get(GS xid)
{
    return _get(xid, GURU_GLOBAL_OBJECT);
}

/* add const */
__GURU__ GV *
const_get(GS xid)
{
    return _get(xid, GURU_CONST_OBJECT);
}
