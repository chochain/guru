/*! @file
  @brief
  GURU - object store

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "alloc.h"
#include "value.h"
#include "store.h"

//================================================================
/*! binary search

  @param  st	pointer to instance store handle.
  @param  sid	symbol ID.
  @return		result. It's not necessarily found.
*/
__GURU__ S32
_bsearch(guru_store *st, GS sid)
{
    int left  = 0;
    int right = st->n - 1;
    if (right<=0) return -1;

    guru_store_data *d = st->data;
    while (left < right) {
        int mid = (left + right) >> 1; 			// div by 2
        if ((d+mid)->sid < sid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

//================================================================
/*! resize buffer

  @param  st	pointer to instance store handle.
  @param  size	size.
  @return		0: success, 1: failed
*/
__GURU__ U32
_resize(guru_store *st, U32 size)
{
    guru_store_data *d2 = (guru_store_data *)guru_realloc(st->data, sizeof(guru_store_data) * size);

    st->data = d2;
    st->size = size;

    return 0;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size.
  @return instance store handle
*/
__GURU__ guru_store *
_new(U32 size)
{
    guru_store *st = (guru_store *)guru_alloc(sizeof(guru_store));

    st->data = (guru_store_data *)guru_alloc(sizeof(guru_store_data) * size);
    st->size = size;
    st->n    = 0;

    return st;
}

//================================================================
/*! destructor

  @param  st	pointer to instance store handle.
*/
__GURU__ void
_del(guru_store *st)
{
    guru_store_data *d = st->data;
    for (U32 i=0; i<st->n; i++, d++) {		// free logical
        ref_clr(&d->val);            // CC: was dec_ref 20181101
    }
    st->n = 0;

    guru_free(st->data);					// free physical
    guru_free(st);
}

//================================================================
/*! setter

  @param  st		pointer to instance store handle.
  @param  sid	symbol ID.
  @param  set_val	set value.
  @return			0: success, 1:failed
*/
__GURU__ S32
_set(guru_store *st, GS sid, GV *val)
{
    S32 idx = _bsearch(st, sid);
    guru_store_data *d = st->data + idx;
    if (idx < 0) {
    	idx = 0;
        goto INSERT_VALUE;
    }
    // replace value ?
    if (d->sid == sid) {
        ref_clr(&d->val);      								// CC: was dec_refc 20181101
        d->val = *val;
        return 0;
    }
    if (d->sid < sid) idx++;

INSERT_VALUE:
    if (st->n >= st->size) {								// need resize?
        if (_resize(st, st->size + 5)) return -1;			// ENOMEM
    }
    d = st->data + idx;
    if (idx < st->n) {										// need more data?
        int size = sizeof(guru_store_data) * (st->n - idx);
        MEMCPY(d+1, d, size);
    }
    d->sid = sid;
    d->val = *val;
    st->n++;

    return 0;
}

//================================================================
/*! getter

  @param  st	pointer to instance store handle.
  @param  sid	symbol ID.
  @return		pointer to GV or NULL.
*/
__GURU__ GV*
_get(guru_store *st, GS sid)
{
    S32 idx = _bsearch(st, sid);
    if (idx < 0) return NULL;

    guru_store_data *d = st->data + idx;
    if (d->sid != sid) return NULL;

    return &d->val;
}

//================================================================
/*! guru_store constructor

  @param  vm    Pointer to VM.
  @param  cls	Pointer to Class (guru_class).
  @param  size	size of additional data.
  @return       guru_store object.
*/
__GURU__ GV
guru_store_new(guru_class *cls, U32 size)
{
    GV v = { .gt = GT_OBJ };
    v.self = (guru_var *)guru_alloc(sizeof(guru_store));

    v.self->ivar = _new(0);			// allocate internal kv handle
    v.self->cls  = cls;

    return v;
}

//================================================================
/*! instance variable destructor

  @param  v	pointer to target value
*/
__GURU__ void
guru_store_del(GV *v)
{
    _del(v->self->ivar);
    guru_free(v->self);
}

//================================================================
/*! instance variable setter

  @param  obj	pointer to target.
  @param  sid	key symbol ID.
  @param  v		pointer to value.
*/
__GURU__ void
guru_store_set(guru_obj *obj, GS sid, GV *v)
{
    _set(obj->self->ivar, sid, v);
    ref_inc(v);
}

//================================================================
/*! instance variable getter

  @param  obj	pointer to target.
  @param  sid	key symbol ID.
  @return		value.
*/
__GURU__ guru_obj
guru_store_get(guru_obj *obj, GS sid)
{
    GV *v = _get(obj->self->ivar, sid);

    if (!v) return GURU_NIL_NEW();

    ref_inc(v);
    return *v;
}
