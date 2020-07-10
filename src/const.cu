/*! @file
  @brief
  GURU global objects.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "global.h"

#if CC_DEBUG
#include "puts.h"
#endif // CC_DEBUG
/*
  GLobal objects are stored in '_global' array.
  '_global' array is decending order by sid.
  In case of searching a global object, binary search is used.
  In case of adding a global object, insertion sort is used.
*/
typedef struct {						// 32-bit
    GP		key;						// cache key (0: global)
    GS 		xid;
} _gidx;

#define _LOCK		{ MUTEX_LOCK(_mutex_gobj); }
#define _UNLOCK		{ MUTEX_FREE(_mutex_gobj); }

// max of global object in _global[]
__GURU__ U32 	_mutex_gobj = 0;
__GURU__ U32    _global_sz  = 0;
__GURU__ _gidx 	_global_idx[MAX_GLOBAL_COUNT];
__GURU__ GR 	_global[MAX_GLOBAL_COUNT];
__GURU__ GR		_NIL = { .gt=GT_NIL, .acl=0 };

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
#if CUDA_ENABLE_CDP
__GPU__ void
__idx(S32 *idx, GP key, GS xid)
{
	S32    i = threadIdx.x;
	_gidx *p = _global_idx + i;

	if (i<_global_sz && p->key==key && p->xid==xid) *idx = i;
}
#else
__GURU__ S32
__idx(GP key, GS xid)
{
	_gidx *p = _global_idx;
	for (int i=0; i<_global_sz; i++, p++) {
		if (p->key==key && p->xid==xid) return i;
	}
	return -1;
}
#endif // CUDA_ENABLE_CDP

__GURU__ S32
_find_idx(GP key, GS xid)
{
	static S32 idx;					// warning: outside of function scope
#if CUDA_ENABLE_CDP
	idx = -1;
	__idx<<<1, 32*(1+(_global_sz>>5))>>>(&idx, key, xid);
	GPU_CHK();						// make sure idx is captured
#else
	idx = __idx(key, xid);
#endif // CUDA_ENABLE_CDP
	return idx;
}

__GURU__ GR *
_get(GP key, GS xid)
{
	S32 i = _find_idx(key, xid);
    if (i < 0) return &_NIL;		// not found

    return &_global[i];				// pointer to global object
}

__GURU__ void
_set(GP key, GS xid, GR *r)
{
    S32 i = _find_idx(key, xid);

    _LOCK;
    if (i<0) {
    	i = _global_sz++;
    	ASSERT(i<MAX_GLOBAL_COUNT);	// maybe raise ex
    }
    _global_idx[i].key = key;
    _global_idx[i].xid = xid;
    _global[i] = *r;
    _UNLOCK;

#if CC_DEBUG
    PRINTF("G[%d]=", i);	guru_puts(r, 1);
#endif // CC_DEBUG
}

/* add */
/* TODO: Check reference count */
__GURU__ void
gv_set(GS xid, GR *r)
{
    _set(0, xid, r);
}

/* get */
__GURU__ GR *
gv_get(GS xid)
{
    return _get(0, xid);
}

__GURU__ void
const_set(GP key, GS xid, GR *r)
{
    _set(key, xid, r);
}

/* add const */
__GURU__ GR *
const_get(GP key, GS xid)
{
    return _get(key, xid);
}
