/*! @file
  @brief
  mruby/c Array class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

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
__GURU__ S32
_adjust_index(guru_array *h, S32 idx, U32 inc)
{
    if (idx < 0) {
        idx += h->n + inc;
        assert(idx>=0);
    }
    int nsz = 0;
    if (idx >= h->size) {	// need resize?
        nsz = idx + inc;
    }
    else if (h->n >= h->size) {
        nsz = h->n + 4;		// pre allocate
    }
    if (nsz && guru_array_resize(h, nsz) != 0) return -1;

    return idx;
}

//================================================================
/*! setter

  @param  ary		pointer to target value
  @param  idx		index
  @param  set_val	set value
  @return			mrbc_error_code
*/
__GURU__ int
_set(GV *ary, int idx, GV *val)
{
    guru_array *h = ary->array;

    int ndx = _adjust_index(h, idx, 0);			// adjust index if needed
    if (ndx<0) return -1;						// allocation error

    if (ndx < h->n) {
        ref_clr(&h->data[ndx]);			// release existing data
    }
    else {
        for (U32 i=h->n; i<ndx; i++) {			// lazy fill here, instead of when resized
            h->data[i] = GURU_NIL_NEW();		// prep newly allocated cells
        }
        h->n = ndx+1;
    }
    h->data[ndx] = *val;

    return 0;
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

    return h->data[--h->n];
}

//================================================================
/*! insert a data

  @param  ary		pointer to target value
  @param  idx		index
  @param  set_val	set value
  @return			mrbc_error_code
*/
__GURU__ int
_insert(GV *ary, int idx, GV *set_val)
{
    guru_array *h = ary->array;

    int size = _adjust_index(h, idx, 1);
    if (size < 0) return -1;

    if (idx < h->n) {			// move data
    	int blksz = sizeof(GV)*(h->n - idx);
        MEMCPY((uint8_t *)(h->data + idx + 1),(uint8_t *)(h->data + idx), blksz);	// shift
    }

    h->data[idx] = *set_val;	// set data
    h->n++;

    if (size >= h->n) {			// clear empty cells if needed
        for (U32 i = h->n-1; i < size; i++) {
            h->data[i] = GURU_NIL_NEW();
        }
        h->n = size;
    }
    return 0;
}

//================================================================
/*! insert a data to the first.

  @param  ary		pointer to target value
  @param  set_val	set value
  @return			mrbc_error_code
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

    GV ret = h->data[0];
    MEMCPY((uint8_t *)h->data, (uint8_t *)(h->data+1), sizeof(GV)*--h->n);

    return ret;
}

//================================================================
/*! getter

  @param  ary		pointer to target value
  @param  idx		index
  @return			GV data at index position or Nil.
*/
__GURU__ GV
_get(GV *ary, int idx)
{
    guru_array *h = ary->array;

    if (idx < 0) idx = h->n + idx;
    if (idx < 0 || idx >= h->n) return GURU_NIL_NEW();

    return h->data[idx];
}

//================================================================
/*! remove a data

  @param  ary		pointer to target value
  @param  idx		index
  @return			GV data at index position or Nil.
*/
__GURU__ GV
_remove(GV *ary, int idx)
{
    guru_array *h = ary->array;

    if (idx < 0) idx = h->n + idx;
    if (idx < 0 || idx >= h->n) return GURU_NIL_NEW();

    GV *p  = h->data + idx;
    GV ret = *p;
    if (idx < --h->n) {										// shrink by 1
    	int blksz = sizeof(GV) * (h->n - idx);
        MEMCPY((uint8_t *)p, (uint8_t *)(p+1), blksz);		// shift forward
    }
    return ret;
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
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size
  @return 		array object
*/
__GURU__ GV
guru_array_new(int size)
{
    GV ret = {.gt = GT_ARRAY};
    guru_array *h = (guru_array *)mrbc_alloc(sizeof(guru_array));		// handle
    if (!h) return ret;		// ENOMEM

    GV *data = (GV *)mrbc_alloc(sizeof(GV) * size);	// buffer
    if (!data) {			// ENOMEM
        mrbc_free(h);
        return ret;
    }
    h->refc = 1;			// handle is referenced
    h->gt 	= GT_ARRAY;
    h->size = size;
    h->n  	= 0;
    h->data = data;

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
    	ref_clr(p);
    }
    mrbc_free(h->data);
    mrbc_free(h);
}

//================================================================
/*! resize buffer

  @param  ary	pointer to target value
  @param  size	size
  @return	mrbc_error_code
*/
__GURU__ int
guru_array_resize(guru_array *h, int size)
{
	assert(size > h->size);

    GV *d2 = (GV *)mrbc_realloc(h->data, sizeof(GV) * size);
    if (!d2) return -1;

    h->data = d2;			// lazy fill later
    h->size = size;

    return 0;
}

//================================================================
/*! push a data to tail

  @param  ary		pointer to target value
  @param  set_val	set value
  @return		mrbc_error_code
*/
__GURU__ int
guru_array_push(GV *ary, GV *set_val)
{
    guru_array *h = ary->array;

    if (h->n >= h->size) {
        int size = h->size + 6;
        if (guru_array_resize(h, size) != 0) {
            return -1;
        }
    }
    h->data[h->n++] = *set_val;

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
    	ref_clr(p);                      // CC: was dec_refc 20181101
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
    for (U32 i=0; ; i++) {
        if (i >= h0->n || i >= h1->n) break;

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
    if (argc==0) {													// in case of new()
        ret = guru_array_new(0);
        if (ret.array==NULL) return;		// ENOMEM
    }
    else if (argc==1 && v[1].gt==GT_INT && v[1].i >= 0) {	// new(num)
        ret = guru_array_new(v[1].i);
        if (ret.array==NULL) return;		// ENOMEM

        GV nil = GURU_NIL_NEW();
        if (v[1].i > 0) {
            _set(&ret, v[1].i - 1, &nil);
        }
    }
    else if (argc==2 && v[1].gt==GT_INT && v[1].i >= 0) {	// new(num, value)
        ret = guru_array_new(v[1].i);
        if (ret.array==NULL) return;		// ENOMEM

        for (U32 i=0; i < v[1].i; i++) {
            ref_inc(&v[2]);
            _set(&ret, i, &v[2]);
        }
    }
    else {
    	ret = GURU_NIL_NEW();
        PRINTF("ArgumentError\n");	// raise?
    }
    SET_RETURN(ret);
}

//================================================================
/*! (operator) +
 */
__GURU__ void
c_array_add(GV v[], U32 argc)
{
    if (GET_GT_ARG(1) != GT_ARRAY) {
        PRINTF("TypeError\n");		// raise?
        return;
    }
    guru_array *h0 = v[0].array;
    guru_array *h1 = v[1].array;

    int h0sz = sizeof(GV) * h0->n;
    int h1sz = sizeof(GV) * h1->n;

    GV ret = guru_array_new(h0sz + h1sz);
    if (ret.array==NULL) return;		// ENOMEM

    MEMCPY((U8P)(ret.array->data),        (U8P)h0->data, h0sz);
    MEMCPY((U8P)(ret.array->data) + h0sz, (U8P)h1->data, h1sz);

    GV *p = ret.array->data;
    int         n = ret.array->n = h0->n + h1->n;
    for (U32 i=0; i<n; i++, p++) {
    	ref_inc(p);
    }
    ref_clr(v+1);					// dec_refc v[1], free if not needed

    SET_RETURN(ret);
}

//================================================================
/*! (operator) []
 */
__GURU__ void
c_array_get(GV v[], U32 argc)
{
	GV ret;
    if (argc==1 && v[1].gt==GT_INT) {				// self[n] -> object | nil
        ret = _get(v, v[1].i);
        ref_inc(&ret);
    }
    else if (argc==2 &&			 						// self[idx, len] -> Array | nil
    		v[1].gt==GT_INT &&
    		v[2].gt==GT_INT) {
        int len = v->array->n;
        int idx = v[1].i;
        if (idx < 0) idx += len;
        if (idx < 0) goto DONE;

        int size = (v[2].i < (len - idx)) ? v[2].i : (len - idx);
        // min(v[2].i, (len - idx))
        if (size < 0) goto DONE;

        ret = guru_array_new(size);
        if (ret.array==NULL) return;		// ENOMEM

        for (U32 i=0; i < size; i++) {
            GV val = _get(v, v[1].i + i);
            ref_inc(&val);
            guru_array_push(&ret, &val);
        }
    }
    else {
        guru_na("case of Array#[]");
    	ret = GURU_NIL_NEW();
    }
DONE:
    SET_RETURN(ret);
}

//================================================================
/*! (operator) []=
 */
__GURU__ void
c_array_set(GV v[], U32 argc)
{
    if (argc==2 && v[1].gt==GT_INT) {		// self[n] = val
        _set(v, v[1].i, &v[2]);		// raise? IndexError or ENOMEM
    }
    else if (argc==3 &&							// self[n, len] = valu
    		v[1].gt==GT_INT &&
    		v[2].gt==GT_INT) {
        // TODO: not implement yet.
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
	int n = GET_INT_ARG(1);

    SET_RETURN(_remove(v, n));
}

//================================================================
/*! (method) empty?
 */
__GURU__ void
c_array_empty(GV v[], U32 argc)
{
    SET_BOOL_RETURN(v->array->n==0);
}

//================================================================
/*! (method) size,length,count
 */
__GURU__ void
c_array_size(GV v[], U32 argc)
{
    SET_INT_RETURN(v->array->n);
}

//================================================================
/*! (method) index
 */
__GURU__ void
c_array_index(GV v[], U32 argc)
{
    GV *ret = v+1;
    
    guru_array *h = v->array;
    GV *p = h->data;
    for (U32 i=0; i < h->n; i++, p++) {
        if (guru_cmp(p, ret)==0) {
            SET_INT_RETURN(i);
            return;
        }
    }
    SET_NIL_RETURN();
}

//================================================================
/*! (method) first
 */
__GURU__ void c_array_first(GV v[], U32 argc)
{
    GV ret = _get(v, 0);
    ref_inc(&ret);
	SET_RETURN(ret);
}

//================================================================
/*! (method) last
 */
__GURU__ void
c_array_last(GV v[], U32 argc)
{
    GV ret = _get(v, -1);
    ref_inc(&ret);
	SET_RETURN(ret);
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
    if (argc==0) {									// pop() -> object | nil
        SET_RETURN(_pop(v));
    }
    else if (argc==1 && v[1].gt==GT_INT) {	// pop(n) -> Array | nil
        // TODO: not implement yet.
    }
    else {
    	PRINTF("case of Array#pop");
    }
}

//================================================================
/*! (method) unshift
 */
__GURU__ void
c_array_unshift(GV v[], U32 argc)
{
    _unshift(v, v+1);								// raise? IndexError or ENOMEM
    v[1].gt = GT_EMPTY;
}

//================================================================
/*! (method) shift
 */
__GURU__ void
c_array_shift(GV v[], U32 argc)
{
    if (argc==0) {									// shift() -> object | nil
        SET_RETURN(_shift(v));
    }
    else if (argc==1 && v[1].gt==GT_INT) {		// shift() -> Array | nil
        // TODO: not implement yet.
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
    GV ret = guru_array_new(h0->n);
    guru_array *h1 = ret.array;
    if (!h1) return;		// ENOMEM

    int n = h1->n = h0->n;
    MEMCPY((U8P)h1->data, (U8P)h0->data, n*sizeof(GV));

    GV *p = h1->data;
    for (U32 i=0; i<n; i++, p++) {
        ref_inc(p);
    }
    SET_RETURN(ret);
}

//================================================================
/*! (method) min
 */
__GURU__ void
c_array_min(GV v[], U32 argc)
{
    // Subset of Array#min, not support min(n).

    GV *p_min_value, *p_max_value;

    _minmax(v, &p_min_value, &p_max_value);
    if (p_min_value==NULL) SET_NIL_RETURN();
    else {
        ref_inc(p_min_value);
        SET_RETURN(*p_min_value);
    }
}

//================================================================
/*! (method) max
 */
__GURU__ void
c_array_max(GV v[], U32 argc)
{
    // Subset of Array#max, not support max(n).

    GV *p_min_value, *p_max_value;

    _minmax(v, &p_min_value, &p_max_value);
    if (p_max_value==NULL) SET_NIL_RETURN();
    else {
        ref_inc(p_max_value);
        SET_RETURN(*p_max_value);
    }
}

//================================================================
/*! (method) minmax
 */
__GURU__ void
c_array_minmax(GV v[], U32 argc)
{
    // Subset of Array#minmax, not support minmax(n).

    GV *p_min_value, *p_max_value;
    GV nil = GURU_NIL_NEW();
    GV ret = guru_array_new(2);

    _minmax(v, &p_min_value, &p_max_value);
    if (p_min_value==NULL) p_min_value = &nil;
    if (p_max_value==NULL) p_max_value = &nil;

    ref_inc(p_min_value);
    ref_inc(p_max_value);
    _set(&ret, 0, p_min_value);
    _set(&ret, 1, p_max_value);

    SET_RETURN(ret);
}

#if GURU_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__ void
c_array_inspect(GV v[], U32 argc)
{
	GV ret  = guru_str_new("[");
    if (!ret.str) {
    	SET_NIL_RETURN();
    	return;
    }
    GV vi, s1;
    int n = v->array->n;
    for (U32 i=0; i < n; i++) {
        if (i != 0) guru_str_append_cstr(&ret, ", ");
        vi = _get(v, i);
        s1 = guru_inspect(v+argc, &vi);
        guru_str_append(&ret, &s1);
        ref_clr(&s1);
    }
    guru_str_append_cstr(&ret, "]");

    SET_RETURN(ret);
}

//================================================================
/*! (method) join
 */
__GURU__ void
c_array_join_1(GV v[], U32 argc,
                    GV *src, GV *ret, GV *separator)
{
	guru_array *h = src->array;
    if (h->n==0) return;

    int i   = 0;
    GV s1;
    while (1) {
        if (h->data[i].gt==GT_ARRAY) {
            c_array_join_1(v, argc, &h->data[i], ret, separator);
        }
        else {
            s1 = guru_inspect(v+argc, &h->data[i]);
            guru_str_append(ret, &s1);
            ref_clr(&s1);					// free locally allocated memory
        }
        if (++i >= h->n) break;					// normal return.
        guru_str_append(ret, separator);
    }
}

__GURU__ void
c_array_join(GV v[], U32 argc)
{
    GV ret = guru_str_new(NULL);
    if (!ret.str) {
        SET_NIL_RETURN();
        return;
    }
    GV separator = (argc==0)
    		? guru_str_new("")
    		: guru_inspect(v+argc, v+1);

    c_array_join_1(v, argc, v, &ret, &separator);
    ref_clr(&separator);		            // release locally allocated memory

    SET_RETURN(ret);
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
