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
#include "instance.h"

//================================================================
/*! binary search

  @param  kv		pointer to key-value handle.
  @param  sym_id	symbol ID.
  @return		result. It's not necessarily found.
*/
__GURU__ int
_bsearch(mrbc_kv *kv, mrbc_sym sym_id)
{
    int left  = 0;
    int right = kv->n - 1;
    if (right < 0) return -1;

    mrbc_kv_data *d = kv->data;
    while (left < right) {
        int mid = (left + right) >> 1; 			// div by 2
        if ((d+mid)->sym_id < sym_id) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

//================================================================
/*! resize buffer

  @param  kv	pointer to key-value handle.
  @param  size	size.
  @return	mrbc_error_code.
*/
__GURU__ int
_resize(mrbc_kv *kv, int size)
{
    mrbc_kv_data *d2 = (mrbc_kv_data *) mrbc_realloc(kv->data, sizeof(mrbc_kv_data) * size);
    if (!d2) return -1;		// ENOMEM

    kv->data = d2;
    kv->size = size;

    return 0;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size.
  @return 	Key-Value handle.
*/
__GURU__ mrbc_kv*
_new(int size)
{
    mrbc_kv *kv = (mrbc_kv *)mrbc_alloc(sizeof(mrbc_kv));
    if (!kv) return NULL;	// ENOMEM

    kv->data = (mrbc_kv_data *)mrbc_alloc(sizeof(mrbc_kv_data) * size);
    if (!kv->data) {		// ENOMEM
        mrbc_free(kv);
        return NULL;
    }
    kv->size = size;
    kv->n    = 0;

    return kv;
}

//================================================================
/*! destructor

  @param  kv	pointer to key-value handle.
*/
__GURU__ void
_delete(mrbc_kv *kv)
{
    mrbc_kv_data *d = kv->data;
    for (int i=0; i<kv->n; i++, d++) {		// free logical
        mrbc_release(&d->value);            // CC: was dec_ref 20181101
    }
    kv->n = 0;

    mrbc_free(kv->data);					// free physical
    mrbc_free(kv);
}

//================================================================
/*! setter

  @param  kv		pointer to key-value handle.
  @param  sym_id	symbol ID.
  @param  set_val	set value.
  @return		mrbc_error_code.
*/
__GURU__ int
_set(mrbc_kv *kv, mrbc_sym sym_id, mrbc_value *val)
{
    int idx = _bsearch(kv, sym_id);
    if (idx < 0) {
        idx = 0;
        goto INSERT_VALUE;
    }
    // replace value ?
    mrbc_kv_data *d = kv->data + idx;
    if (d->sym_id == sym_id) {
        mrbc_release(&d->value);      // CC: was dec_refc 20181101
        d->value = *val;
        return 0;
    }
    if (d->sym_id < sym_id) idx++;

INSERT_VALUE:
    if (kv->n >= kv->size) {								// need resize?
        if (_resize(kv, kv->size + 5) != 0) return -1;		// ENOMEM
    }
    d = kv->data + idx;
    if (idx < kv->n) {										// need more data?
        int size = sizeof(mrbc_kv_data) * (kv->n - idx);
        MEMCPY((uint8_t *)(d+1), (const uint8_t *)d, size);
    }
    d->sym_id = sym_id;
    d->value  = *val;
    kv->n++;

    return 0;
}

//================================================================
/*! getter

  @param  kv		pointer to key-value handle.
  @param  sym_id	symbol ID.
  @return		pointer to mrbc_value or NULL.
*/
__GURU__ mrbc_value*
_get(mrbc_kv *kv, mrbc_sym sym_id)
{
    int idx = _bsearch(kv, sym_id);
    if (idx < 0) return NULL;

    mrbc_kv_data *d = kv->data + idx;
    if (d->sym_id != sym_id) return NULL;

    return &d->value;
}

//================================================================
/*! mrbc_instance constructor

  @param  vm    Pointer to VM.
  @param  cls	Pointer to Class (mrbc_class).
  @param  size	size of additional data.
  @return       mrbc_instance object.
*/
__GURU__ mrbc_value
mrbc_instance_new(mrbc_class *cls, int size)
{
    mrbc_value v = {.tt = MRBC_TT_OBJECT};
    v.self = (mrbc_instance *)mrbc_alloc(sizeof(mrbc_instance) + size);
    if (v.self == NULL) return v;	// ENOMEM

    v.self->ivar = _new(0);			// allocate internal kv handle
    if (v.self->ivar == NULL) {		// ENOMEM
        mrbc_free(v.self);
        v.self = NULL;
        return v;
    }

    v.self->refc = 1;
    v.self->tt   = MRBC_TT_OBJECT;	// for debug only.
    v.self->cls  = cls;

    return v;
}

//================================================================
/*! mrbc_instance destructor

  @param  v	pointer to target value
*/
__GURU__ void
mrbc_instance_delete(mrbc_value *v)
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
mrbc_instance_setiv(mrbc_object *obj, mrbc_sym sym_id, mrbc_value *v)
{
    _set(obj->self->ivar, sym_id, v);
    mrbc_retain(v);
}


//================================================================
/*! instance variable getter

  @param  obj		pointer to target.
  @param  sym_id	key symbol ID.
  @return		value.
*/
__GURU__ mrbc_value
mrbc_instance_getiv(mrbc_object *obj, mrbc_sym sym_id)
{
    mrbc_value *v = _get(obj->self->ivar, sym_id);
    return (v) ? *v : mrbc_nil_value();
}










