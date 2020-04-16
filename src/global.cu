/*! @file
  @brief
  GURU global objects.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
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
__GURU__ GV		_NIL = { .gt = GT_NIL, .acl=0 };

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
#if CUDA_PROFILE_CDP
__GPU__ void
__idx(S32 *idx, GS xid, _gtype gt)
{
	S32    i = threadIdx.x;
	_gidx *p = _global_idx + i;

	if (i<_global_sz && p->xid == xid && p->gt == gt) *idx = i;
}
#else
__GURU__ S32
__idx(GS xid, _gtype gt)
{
	_gidx *p = _global_idx;
	for (int i=0; i<_global_sz; i++, p++) {
		if (p->xid==xid && p->gt==gt) return i;
	}
	return -1;
}
#endif // CUDA_PROFILE_CDP

__GURU__ S32
_find_idx(GS xid, _gtype gt)
{
	static S32 idx;					// warning: outside of function scope
#if CUDA_PROFILE_CDP
	idx = -1;
	__idx<<<1, 32*(1+(_global_sz>>5))>>>(&idx, xid, gt);
	SYNC_CHK();						// make sure idx is captured
#else
	idx = __idx(xid, gt);
#endif // CUDA_PROFILE_CDP
	return idx;
}

__GURU__ GV *
_get(GS xid, _gtype gt)
{
	S32 i = _find_idx(xid, gt);
    if (i < 0) return &_NIL;		// not found

    return &_global[i];				// pointer to global object
}

__GURU__ void
_set(GS xid, GV *v, _gtype gt)
{
    S32 i = _find_idx(xid, gt);

    _LOCK;
    if (i<0) {
    	i = _global_sz++;
    	ASSERT(i<MAX_GLOBAL_COUNT);	// maybe raise ex
    }
    else {
//    	ASSERT(i<0);
    	ASSERT(1==1);
    }
    _global_idx[i].xid = xid;
    _global_idx[i].gt  = gt;
    _global[i] = *v;
    _UNLOCK;

#if CC_DEBUG
    PRINTF("G[%d]=", i);	guru_puts(v, 1);
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
