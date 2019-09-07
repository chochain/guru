/*! @file
  @brief
  GURU Array class

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>

#include "vm_config.h"
#include "guru.h"
#include "alloc.h"
#include "static.h"

#include "opcode.h"
#include "object.h"
#include "c_array.h"
#include "c_string.h"

#include "puts.h"

/*
  function summary

 (constructor)
    guru_array_new

 (destructor)
    guru_array_delete

 (setter)
  --[name]-------------[arg]---[ret]-------------------------------------------
    guru_array_set		*T		int
    guru_array_push		*T		int
    guru_array_unshift	*T		int
    guru_array_insert	*T		int

 (getter)
  --[name]-------------[arg]---[ret]---[note]----------------------------------
    guru_array_get		T		Data remains in the container
    guru_array_pop		T		Data does not remain in the container
    guru_array_shift	T		Data does not remain in the container
    guru_array_remove	T		Data does not remain in the container

 (others)
    guru_array_resize
    guru_array_clear
    guru_array_compare
    guru_array_minmax
*/

//================================================================
/*! get size
 */
__GURU__ U32
_reindex(guru_array *h, S32 idx, U32 inc)
{
    if (idx < 0) {					// index from tail of array
        idx += h->n + inc;
        assert(idx>=0);
    }

    U32 nsz = 0;
    if (idx >= h->size) {			// need resize?
        nsz = idx + inc;
    }
    else if (h->n >= h->size) {
        nsz = h->n + 4;				// pre allocate
    }
    if (nsz) {
    	guru_array_resize(h, nsz);
    }
    return idx;
}

//================================================================
/*! setter

  @param  ary		pointer to target value
  @param  idx		index
  @param  set_val	set value
*/
__GURU__ void
_set(GV *ary, S32 idx, GV *val)
{
    guru_array *h = ary->array;

    U32 ndx = _reindex(h, idx, 0);				// adjust index if needed

    if (ndx < h->n) {
        ref_dec(&h->data[ndx]);					// release existing data
    }
    else {
        for (U32 i=h->n; i<ndx; i++) {			// lazy fill here, instead of when resized
            h->data[i] = GURU_NIL_NEW();		// prep newly allocated cells
        }
        h->n = ndx+1;
    }
    h->data[ndx] = *ref_inc(val);				// keep the reference to the value
}

__GURU__ void
_push(GV *ary, GV *set_val)
{
    guru_array *h = ary->array;

    if (h->n >= h->size) {
        U32 sz = h->size + 6;
        guru_array_resize(h, sz);
    }
    h->data[h->n++] = *ref_inc(set_val);
}
//================================================================
/*! pop a data from tail.

  @param  ary	pointer to target value
  @return		tail data or Nil
*/
__GURU__ GV
_pop(GV *ary)
{
    guru_array *h = ary->array;

    if (h->n <= 0) return GURU_NIL_NEW();

    return *ref_dec(&h->data[--h->n]);
}

//================================================================
/*! insert a data

  @param  ary		pointer to target value
  @param  idx		index
  @param  set_val	set value
  @return			error_code
*/
__GURU__ int
_insert(GV *ary, S32 idx, GV *set_val)
{
    guru_array *h = ary->array;

    U32 ndx = _reindex(h, idx, 1);

    if (ndx < h->n) {										// move data
    	U32 blksz = sizeof(GV)*(h->n - idx);
        MEMCPY(h->data + ndx + 1, h->data + ndx, blksz);	// rshift
    }

    h->data[ndx] = *ref_inc(set_val);						// set data
    h->n++;

    if (ndx >= h->n) {										// clear empty cells
        for (U32 i = h->n-1; i < ndx; i++) {
            h->data[i] = GURU_NIL_NEW();
        }
        h->n = ndx;
    }
    return 0;
}

//================================================================
/*! insert a data to the first.

  @param  ary		pointer to target value
  @param  set_val	set value
  @return			error_code
*/
__GURU__ int
_unshift(GV *ary, GV *set_val)
{
    return _insert(ary, 0, set_val);
}

//================================================================
/*! removes the first data and returns it.

  @param  ary		pointer to target value
  @return		first data or Nil
*/
__GURU__ GV
_shift(GV *ary)
{
    guru_array *h = ary->array;

    if (h->n <= 0) return GURU_NIL_NEW();

    GV *v = ref_dec(&h->data[0]);
    MEMCPY(h->data, h->data + 1, sizeof(GV)*(--h->n));		// lshift

    return *v;
}

//================================================================
/*! getter (no change to ref count)

  @param  ary		pointer to target value
  @param  idx		index
  @return			GV data at index position or Nil.
*/
__GURU__ GV
_get(GV *ary, S32 idx)
{
    guru_array *h = ary->array;
    U32 ndx = (idx < 0) ? h->n + idx : idx;

    return (ndx < h->n) ? h->data[ndx] : GURU_NIL_NEW();
}

//================================================================
/*! remove a data

  @param  ary		pointer to target value
  @param  idx		index
  @return			GV data at index position or Nil.
*/
__GURU__ GV
_remove(GV *ary, S32 idx)
{
    guru_array *h = ary->array;
    U32 ndx = (idx < 0) ? h->n + idx : idx;

    if (ndx >= h->n) return GURU_NIL_NEW();

    GV *v = ref_dec(&h->data[ndx]);
    if (ndx < --h->n) {										// shrink by 1
    	U32 blksz = sizeof(GV) * (h->n - ndx);
        MEMCPY(v, v+1, blksz);								// lshift
    }
    return *v;
}

//================================================================
/*! get min, max value

  @param  ary			pointer to target value
  @param  pp_min_value	returns minimum GV
  @param  pp_max_value	returns maxmum GV
*/
__GURU__ void
_minmax(GV *ary, GV **pp_min_value, GV **pp_max_value)
{
    guru_array *h = ary->array;

    if (h->n==0) {
        *pp_min_value = NULL;
        *pp_max_value = NULL;
        return;
    }
    GV *p_min_value = h->data;
    GV *p_max_value = h->data;
    GV *p           = h->data;
    for (U32 i = 1; i < h->n; i++, p++) {
        if (guru_cmp(p, p_min_value) < 0) p_min_value = p;
        if (guru_cmp(p, p_max_value) > 0) p_max_value = p;
    }
    *pp_min_value = p_min_value;
    *pp_max_value = p_max_value;
}

//================================================================
/*! (method) join
 */
__GURU__ void
_join(GV v[], U32 argc, GV *src, GV *ret, GV *sep)
{
	guru_array *h = src->array;
    if (h->n==0) return;

    U32 i = 0;
    GV  s1;
    while (1) {
        if (h->data[i].gt==GT_ARRAY) {		// recursive
            _join(v, argc, &h->data[i], ret, sep);
        }
        else {
            s1 = guru_inspect(v+argc, &h->data[i]);
            guru_str_append(ret, &s1);
        }
        if (++i >= h->n) break;				// normal return.
        guru_str_append(ret, sep);
    }
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size
  @return 		array object
*/
__GURU__ GV
guru_array_new(U32 sz)
{
    GV ret = {.gt = GT_ARRAY};

    guru_array *h   = (guru_array *)guru_alloc(sizeof(guru_array));		// handle
    void       *ptr = guru_alloc(sizeof(GV) * sz);

    h->gt 	= GT_ARRAY;
    h->rc   = 1;			// assume handle is referenced
    h->size = sz;
    h->n  	= 0;
    h->data = (GV *)ptr;

    ret.array = h;

    return ret;
}

//================================================================
/*! destructor

  @param  ary	pointer to target value
*/
__GURU__ void
guru_array_delete(GV *ary)
{
    guru_array *h = ary->array;
    GV *p = h->data;
    for (U32 i=0; i < h->n; i++, p++) {
    	ref_dec(p);						// no more referenced by the array
    }
    guru_free(h->data);
    guru_free(h);
}

//================================================================
/*! resize buffer

  @param  ary	pointer to target value
  @param  size	size
  @return	error_code
*/
__GURU__ int
guru_array_resize(guru_array *h, U32 new_sz)
{
	assert(new_sz > h->size);

    void *ptr = guru_realloc(h->data, sizeof(GV) * new_sz);

    h->size = new_sz;
    h->data = (GV *)ptr;			// lazy fill later

    return 0;
}

//================================================================
/*! push a data to tail

  @param  ary		pointer to target value
  @param  set_val	set value
  @return			error_code
*/
__GURU__ int
guru_array_push(GV *ary, GV *set_val)
{
	_push(ary, set_val);
    return 0;
}

//================================================================
/*! clear all

  @param  ary		pointer to target value
*/
__GURU__ void
guru_array_clear(GV *ary)
{
    guru_array *h = ary->array;
    GV *p = h->data;
    for (U32 i=0; i < h->n; i++, p++) {
    	ref_dec(p);
    }
    h->n = 0;
}

//================================================================
/*! compare

  @param  v1	Pointer to GV
  @param  v2	Pointer to another GV
  @retval 0	v1==v2
  @retval plus	v1 >  v2
  @retval minus	v1 <  v2
*/
__GURU__ int
guru_array_compare(const GV *v0, const GV *v1)
{
	guru_array *h0 = v0->array;
	guru_array *h1 = v1->array;
	GV *d0 = h0->data;
	GV *d1 = h1->data;

	for (U32 i=0; i < h0->n && i < h1->n; i++) {
        int res = guru_cmp(d0++, d1++);
        if (res != 0) return res;
    }
    return h0->n - h1->n;
}

//================================================================
/*! method new
 */
__GURU__ void
c_array_new(GV v[], U32 argc)
{
	GV ret;
    if (argc==0) {											// in case of new()
        ret = guru_array_new(0);
    }
    else if (argc==1 && v[1].gt==GT_INT && v[1].i >= 0) {	// new(num)
        ret = guru_array_new(v[1].i);

        GV nil = GURU_NIL_NEW();
        if (v[1].i > 0) {
            _set(&ret, v[1].i - 1, &nil);
        }
    }
    else if (argc==2 && v[1].gt==GT_INT && v[1].i >= 0) {	// new(num, value)
        ret = guru_array_new(v[1].i);
        for (U32 i=0; i < v[1].i; i++) {
            _set(&ret, i, &v[2]);
        }
    }
    else {
    	ret = GURU_NIL_NEW();
        PRINTF("ArgumentError\n");	// raise?
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (operator) +
 */
__GURU__ void
c_array_add(GV v[], U32 argc)
{
    assert(ARG_GT(1) == GT_ARRAY);			// array only (for now)

    guru_array *h0 = v[0].array;
    guru_array *h1 = v[1].array;

    U32 h0sz = sizeof(GV) * h0->n;
    U32 h1sz = sizeof(GV) * h1->n;

    GV ret = guru_array_new(h0sz + h1sz);	// new array with ref count already set to 1

    MEMCPY(ret.array->data,         h0->data, h0sz);
    MEMCPY(ret.array->data + h0->n, h1->data, h1sz);

    ret.array->n = h0->n + h1->n;			// reset element count

    // free both source arrays
    ref_dec(v);
    ref_dec(v+1);

    RETURN_VAL(ret);
}

//================================================================
/*! (operator) []
 */
__GURU__ void
c_array_get(GV v[], U32 argc)
{
	GV ret;
    if (argc==1 && v[1].gt==GT_INT) {					// self[n] -> object | nil
        ret = _get(v, v[1].i);
    }
    else if (argc==2 &&			 						// self[idx, len] -> Array | nil
    		v[1].gt==GT_INT &&
    		v[2].gt==GT_INT) {
        U32 len = v->array->n;
        S32 idx = v[1].i;
        U32 ndx = (idx < 0) ? idx + len : idx;

        S32 sz = (v[2].i < (len - ndx)) ? v[2].i : (len - ndx);
        if (sz < 0) RETURN_VAL(ret);

        ret = guru_array_new(sz);
        for (U32 i=0; i < sz; i++) {
            GV val = _get(v, v[1].i + i);
            guru_array_push(&ret, &val);
        }
    }
    else {
        guru_na("case of Array#[]");
    	ret = GURU_NIL_NEW();
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (operator) []=
 */
__GURU__ void
c_array_set(GV v[], U32 argc)
{
	GT gt1 = v[1].gt;
	GT gt2 = v[2].gt;
    if (argc==2 && gt1==GT_INT) {		// self[n] = val
        _set(v, v[1].i, &v[2]);
    }
    else if (argc==3 &&					// self[n, len] = valu
    		gt1==GT_INT &&
    		gt2==GT_INT) {
    	guru_na("array[i,n]");
    }
    else {
        guru_na("case of Array#[]=");
    }
}

//================================================================
/*! (method) clear
 */
__GURU__ void
c_array_clear(GV v[], U32 argc)
{
    guru_array_clear(v);
}

//================================================================
/*! (method) delete_at
 */
__GURU__ void
c_array_delete_at(GV v[], U32 argc)
{
	S32 n = ARG_INT(1);

    RETURN_VAL(_remove(v, n));
}

//================================================================
/*! (method) empty?
 */
__GURU__ void
c_array_empty(GV v[], U32 argc)
{
    RETURN_BOOL(v->array->n==0);
}

//================================================================
/*! (method) size,length,count
 */
__GURU__ void
c_array_size(GV v[], U32 argc)
{
    RETURN_INT(v->array->n);
}

//================================================================
/*! (method) index
 */
__GURU__ void
c_array_index(GV v[], U32 argc)
{
    guru_array *h = v->array;
    GV *p = h->data;
    for (U32 i=0; i < h->n; i++, p++) {
        if (guru_cmp(p, v+1)==0) {
            RETURN_INT(i);
        }
    }
    RETURN_NIL();
}

//================================================================
/*! (method) first
 */
__GURU__ void c_array_first(GV v[], U32 argc)
{
    RETURN_VAL(_get(v, 0));
}

//================================================================
/*! (method) last
 */
__GURU__ void
c_array_last(GV v[], U32 argc)
{
    RETURN_VAL(_get(v, -1));
}

//================================================================
/*! (method) push
 */
__GURU__ void
c_array_push(GV v[], U32 argc)
{
    guru_array_push(v, v+1);	// raise? ENOMEM
    v[1].gt = GT_EMPTY;
}

//================================================================
/*! (method) pop
 */
__GURU__ void
c_array_pop(GV v[], U32 argc)
{
    if (argc==0) {							// pop() -> object | nil
        RETURN_VAL(_pop(v));
    }
    else if (argc==1 && v[1].gt==GT_INT) {	// pop(n) -> Array | nil
        guru_na("pop(n)");					// TODO: loop
    }
    else {
    	guru_na("case of Array#pop");
    }
}

//================================================================
/*! (method) unshift
 */
__GURU__ void
c_array_unshift(GV v[], U32 argc)
{
    _unshift(v, v+1);						// raise? IndexError or ENOMEM
    v[1].gt = GT_EMPTY;
}

//================================================================
/*! (method) shift
 */
__GURU__ void
c_array_shift(GV v[], U32 argc)
{
    if (argc==0) {							// shift() -> object | nil
        RETURN_VAL(_shift(v));
    }
    else if (argc==1 && v[1].gt==GT_INT) {	// shift() -> Array | nil
        guru_na("shift(n)");				// TODO: loop
    }
    else {
    	guru_na("case of Array#shift");
    }
}

//================================================================
/*! (method) dup
 */
__GURU__ void
c_array_dup(GV v[], U32 argc)
{
    guru_array *h0 = v[0].array;

    GV ret = guru_array_new(h0->n);		// create new array
    GV *p0 = h0->data;
    for (U32 i=0; i < h0->n; i++, p0++) {
    	_set(&ret, i, p0);				// shallow copy
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) min
 */
__GURU__ void
c_array_min(GV v[], U32 argc)
{
    // Subset of Array#min, not support min(n).
    GV *min, *max;

    _minmax(v, &min, &max);
    if (min) {
        RETURN_VAL(*min);
    }
    RETURN_NIL();
}

//================================================================
/*! (method) max
 */
__GURU__ void
c_array_max(GV v[], U32 argc)
{
    // Subset of Array#max, not support max(n).
    GV *min, *max;

    _minmax(v, &min, &max);
    if (max) {
        RETURN_VAL(*max);
    }
    RETURN_NIL();
}

//================================================================
/*! (method) minmax
 */
__GURU__ void
c_array_minmax(GV v[], U32 argc)
{
    GV nil = GURU_NIL_NEW();
    GV ret = guru_array_new(2);
    GV *min, *max;

    _minmax(v, &min, &max);
    if (min==NULL) min = &nil;
    if (max==NULL) max = &nil;

    _set(&ret, 0, min);
    _set(&ret, 1, max);

    RETURN_VAL(ret);
}

#if GURU_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__ void
c_array_inspect(GV v[], U32 argc)
{
	GV ret = guru_str_new("[");
    GV vi, s1;
    for (U32 i=0, n=v->array->n; i < n; i++) {
        if (i != 0) guru_str_append_cstr(&ret, ", ");
        vi = _get(v, i);
        s1 = guru_inspect(v+argc, &vi);
        guru_str_append(&ret, &s1);
    }
    guru_str_append_cstr(&ret, "]");

    RETURN_VAL(ret);
}

__GURU__ void
c_array_join(GV v[], U32 argc)
{
    GV ret = guru_str_new(NULL);
    GV sep = (argc==0)						// separator
    		? guru_str_new("")
    		: guru_inspect(v+argc, v+1);
    _join(v, argc, v, &ret, &sep);

    RETURN_VAL(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void
guru_init_class_array()
{
    guru_class *c = guru_class_array = guru_add_class("Array", guru_class_object);

    guru_add_proc(c, "new",       c_array_new);
    guru_add_proc(c, "+",         c_array_add);
    guru_add_proc(c, "[]",        c_array_get);
    guru_add_proc(c, "at",        c_array_get);
    guru_add_proc(c, "[]=",       c_array_set);
    guru_add_proc(c, "<<",        c_array_push);
    guru_add_proc(c, "clear",     c_array_clear);
    guru_add_proc(c, "delete_at", c_array_delete_at);
    guru_add_proc(c, "empty?",    c_array_empty);
    guru_add_proc(c, "size",      c_array_size);
    guru_add_proc(c, "length",    c_array_size);
    guru_add_proc(c, "count",     c_array_size);
    guru_add_proc(c, "index",     c_array_index);
    guru_add_proc(c, "first",     c_array_first);
    guru_add_proc(c, "last",      c_array_last);
    guru_add_proc(c, "push",      c_array_push);
    guru_add_proc(c, "pop",       c_array_pop);
    guru_add_proc(c, "shift",     c_array_shift);
    guru_add_proc(c, "unshift",   c_array_unshift);
    guru_add_proc(c, "dup",       c_array_dup);
    guru_add_proc(c, "min",       c_array_min);
    guru_add_proc(c, "max",       c_array_max);
    guru_add_proc(c, "minmax",    c_array_minmax);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect",   c_array_inspect);
    guru_add_proc(c, "to_s",      c_array_inspect);
    guru_add_proc(c, "join",      c_array_join);
#endif
}
