/*! @file
  @brief
  mrubyc memory management.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in Guru.

  </pre>
*/
#include "alloc.h"
#include "value.h"
#include "store.h"

//================================================================
/*! binary search

  @param  st		pointer to instance store handle.
  @param  sym_id	symbol ID.
  @return		result. It's not necessarily found.
*/
__GURU__ S32
_bsearch(mrbc_store *st, guru_sym sid)
{
    int left  = 0;
    int right = st->n - 1;
    if (right<=0) return -1;

    mrbc_store_data *d = st->data;
    while (left < right) {
        int mid = (left + right) >> 1; 			// div by 2
        if ((d+mid)->sym_id < sid) {
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
  @return	mrbc_error_code.
*/
__GURU__ S32
_resize(mrbc_store *st, U32 size)
{
    mrbc_store_data *d2 = (mrbc_store_data *) mrbc_realloc(st->data, sizeof(mrbc_store_data) * size);
    if (!d2) return -1;		// ENOMEM

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
__GURU__ mrbc_store *
_new(U32 size)
{
    mrbc_store *st = (mrbc_store *)mrbc_alloc(sizeof(mrbc_store));
    if (!st) return NULL;	// ENOMEM

    st->data = (mrbc_store_data *)mrbc_alloc(sizeof(mrbc_store_data) * size);
    if (!st->data) {		// ENOMEM
        mrbc_free(st);
        return NULL;
    }
    st->size = size;
    st->n    = 0;

    return st;
}

//================================================================
/*! destructor

  @param  st	pointer to instance store handle.
*/
__GURU__ void
_delete(mrbc_store *st)
{
    mrbc_store_data *d = st->data;
    for (U32 i=0; i<st->n; i++, d++) {		// free logical
        ref_clr(&d->value);            // CC: was dec_ref 20181101
    }
    st->n = 0;

    mrbc_free(st->data);					// free physical
    mrbc_free(st);
}

//================================================================
/*! setter

  @param  st		pointer to instance store handle.
  @param  sym_id	symbol ID.
  @param  set_val	set value.
  @return		mrbc_error_code.
*/
__GURU__ S32
_set(mrbc_store *st, guru_sym sid, GV *val)
{
    S32 idx = _bsearch(st, sid);
    mrbc_store_data *d = st->data + idx;
    if (idx < 0) {
    	idx = 0;
        goto INSERT_VALUE;
    }
    // replace value ?
    if (d->sym_id == sid) {
        ref_clr(&d->value);      // CC: was dec_refc 20181101
        d->value = *val;
        return 0;
    }
    if (d->sym_id < sid) idx++;

INSERT_VALUE:
    if (st->n >= st->size) {								// need resize?
        if (_resize(st, st->size + 5) != 0) return -1;		// ENOMEM
    }
    d = st->data + idx;
    if (idx < st->n) {										// need more data?
        int size = sizeof(mrbc_store_data) * (st->n - idx);
        MEMCPY((U8P)(d+1), (U8P)d, size);
    }
    d->sym_id = sid;
    d->value  = *val;
    st->n++;

    return 0;
}

//================================================================
/*! getter

  @param  st		pointer to instance store handle.
  @param  sym_id	symbol ID.
  @return		pointer to GV or NULL.
*/
__GURU__ GV*
_get(mrbc_store *st, guru_sym sid)
{
    S32 idx = _bsearch(st, sid);
    if (idx < 0) return NULL;

    mrbc_store_data *d = st->data + idx;
    if (d->sym_id != sid) return NULL;

    return &d->value;
}

//================================================================
/*! mrbc_store constructor

  @param  vm    Pointer to VM.
  @param  cls	Pointer to Class (guru_class).
  @param  size	size of additional data.
  @return       mrbc_store object.
*/
__GURU__ GV
mrbc_store_new(guru_class *cls, U32 size)
{
    GV v = {.gt = GT_OBJ};
    v.self = (guru_var *)mrbc_alloc(sizeof(mrbc_store) + size);
    if (v.self == NULL) return v;	// ENOMEM

    v.self->ivar = _new(0);			// allocate internal kv handle
    if (v.self->ivar == NULL) {		// ENOMEM
        mrbc_free(v.self);
        v.self = NULL;
        return v;
    }

    v.self->refc = 1;
    v.self->gt   = GT_OBJ;	// for debug only.
    v.self->cls  = cls;

    return v;
}

//================================================================
/*! instance variable destructor

  @param  v	pointer to target value
*/
__GURU__ void
mrbc_store_delete(GV *v)
{
    _delete(v->self->ivar);
    mrbc_free(v->self);
}

//================================================================
/*! instance variable setter

  @param  obj		pointer to target.
  @param  sym_id	key symbol ID.
  @param  v		pointer to value.
*/
__GURU__ void
mrbc_store_set(guru_obj *obj, guru_sym sid, GV *v)
{
    _set(obj->self->ivar, sid, v);
    ref_inc(v);
}

//================================================================
/*! instance variable getter

  @param  obj		pointer to target.
  @param  sym_id	key symbol ID.
  @return		value.
*/
__GURU__ guru_obj
mrbc_store_get(guru_obj *obj, guru_sym sid)
{
    GV *v = _get(obj->self->ivar, sid);

    if (!v) return GURU_NIL_NEW();

    ref_inc(v);
    return *v;
}
