/*! @file
  @brief
  GURU constant objects (cache).

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "const.h"

#if CC_DEBUG
#include "puts.h"
#endif // CC_DEBUG
/*
  GLobal objects are stored in '_const' array.
  '_const' array is decending order by sid.
  In case of searching a constant object, binary search is used.
  In case of adding a constant object, insertion sort is used.
*/
typedef struct {						// 32-bit
    GP		ns;							// cache namespace (0: global)
    GS 		xid;
} _gidx;

#define _LOCK		{ MUTEX_LOCK(_mutex_gobj); }
#define _UNLOCK		{ MUTEX_FREE(_mutex_gobj); }

// max of global object in _const[]
__GURU__ U32 	_mutex_gobj = 0;
__GURU__ U32    _const_sz  = 0;
__GURU__ _gidx 	_const_idx[MAX_CONST_COUNT];
__GURU__ GR 	_const[MAX_CONST_COUNT];
__GURU__ GR		_NIL = { .gt=GT_NIL, .acl=0 };

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
#if CUDA_ENABLE_CDP
__GPU__ void
__idx(S32 *idx, GP ns, GS xid)
{
	S32    i = threadIdx.x;
	_gidx *p = _const_idx + i;

	if (i<_const_sz && p->ns==ns && p->xid==xid) *idx = i;
}
#else
__GURU__ S32
__idx(GP ns, GS xid)
{
	_gidx *p = _const_idx;
	for (int i=0; i<_const_sz; i++, p++) {
		if (p->ns==ns && p->xid==xid) return i;
	}
	return -1;
}
#endif // CUDA_ENABLE_CDP

__GURU__ S32
_find_idx(GP ns, GS xid)
{
	static S32 idx;					// warning: outside of function scope
#if CUDA_ENABLE_CDP
	idx = -1;
	__idx<<<1, 32*(1+(_const_sz>>5))>>>(&idx, ns, xid);
	GPU_CHK();						// make sure idx is captured
#else
	idx = __idx(ns, xid);
#endif // CUDA_ENABLE_CDP
	return idx;
}

__GURU__ GR *
_get(GP ns, GS xid)
{
	S32 i = _find_idx(ns, xid);
    if (i < 0) return &_NIL;		// not found

    return &_const[i];				// pointer to global object
}

__GURU__ void
_set(GP ns, GS xid, GR *r)
{
    S32 i = _find_idx(ns, xid);

    _LOCK;
    if (i<0) {
    	i = _const_sz++;
    	ASSERT(i<MAX_CONST_COUNT);	// maybe raise ex
    }
    _const_idx[i].ns  = ns;
    _const_idx[i].xid = xid;
    _const[i] = *r;
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
const_set(GP ns, GS xid, GR *r)
{
	r->acl &= ~ACL_HAS_REF;			// set to constant

    _set(ns, xid, r);
}

/* add const */
__GURU__ GR *
const_get(GP ns, GS xid)
{
    return _get(ns, xid);
}
