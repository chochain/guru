/*! @file
  @brief
  mruby/c Array class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#include "vm_config.h"
#include <string.h>
#include <assert.h>

#include "guru.h"
#include "alloc.h"
#include "static.h"

#include "console.h"
#include "sprintf.h"

#include "opcode.h"
#include "class.h"

#include "object.h"
#include "c_array.h"
#include "c_string.h"

/*
  function summary

 (constructor)
    mrbc_array_new

 (destructor)
    mrbc_array_delete

 (setter)
  --[name]-------------[arg]---[ret]-------------------------------------------
    mrbc_array_set		*T		int
    mrbc_array_push		*T		int
    mrbc_array_unshift	*T		int
    mrbc_array_insert	*T		int

 (getter)
  --[name]-------------[arg]---[ret]---[note]----------------------------------
    mrbc_array_get		T		Data remains in the container
    mrbc_array_pop		T		Data does not remain in the container
    mrbc_array_shift	T		Data does not remain in the container
    mrbc_array_remove	T		Data does not remain in the container

 (others)
    mrbc_array_resize
    mrbc_array_clear
    mrbc_array_compare
    mrbc_array_minmax
*/

//================================================================
/*! get size
 */
__GURU__ S32
_adjust_index(mrbc_array *h, S32 idx, U32 inc)
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
    if (nsz && mrbc_array_resize(h, nsz) != 0) return -1;

    return idx;
}

//================================================================
/*! setter

  @param  ary		pointer to target value
  @param  idx		index
  @param  set_val	set value
  @return		mrbc_error_code
*/
__GURU__ int
_set(mrbc_value *ary, int idx, mrbc_value *val)
{
    mrbc_array *h = ary->array;

    int ndx = _adjust_index(h, idx, 0);			// adjust index if needed
    if (ndx<0) return -1;						// allocation error

    if (ndx < h->n) {
        mrbc_release(&h->data[ndx]);			// release existing data
    }
    else {
        for (U32 i=h->n; i<ndx; i++) {			// lazy fill here, instead of when resized
            h->data[i] = mrbc_nil_value();		// prep newly allocated cells
        }
        h->n = ndx+1;
    }
    h->data[ndx] = *val;

    return 0;
}

//================================================================
/*! pop a data from tail.

  @param  ary		pointer to target value
  @return		tail data or Nil
*/
__GURU__ mrbc_value
_pop(mrbc_value *ary)
{
    mrbc_array *h = ary->array;

    if (h->n <= 0) return mrbc_nil_value();

    return h->data[--h->n];
}

//================================================================
/*! insert a data

  @param  ary		pointer to target value
  @param  idx		index
  @param  set_val	set value
  @return		mrbc_error_code
*/
__GURU__ int
_insert(mrbc_value *ary, int idx, mrbc_value *set_val)
{
    mrbc_array *h = ary->array;

    int size = _adjust_index(h, idx, 1);
    if (size < 0) return -1;

    if (idx < h->n) {			// move data
    	int blksz = sizeof(mrbc_value)*(h->n - idx);
        MEMCPY((uint8_t *)(h->data + idx + 1),(uint8_t *)(h->data + idx), blksz);	// shift
    }

    h->data[idx] = *set_val;	// set data
    h->n++;

    if (size >= h->n) {			// clear empty cells if needed
        for (U32 i = h->n-1; i < size; i++) {
            h->data[i] = mrbc_nil_value();
        }
        h->n = size;
    }
    return 0;
}

//================================================================
/*! insert a data to the first.

  @param  ary		pointer to target value
  @param  set_val	set value
  @return		mrbc_error_code
*/
__GURU__ int
_unshift(mrbc_value *ary, mrbc_value *set_val)
{
    return _insert(ary, 0, set_val);
}

//================================================================
/*! removes the first data and returns it.

  @param  ary		pointer to target value
  @return		first data or Nil
*/
__GURU__ mrbc_value
_shift(mrbc_value *ary)
{
    mrbc_array *h = ary->array;

    if (h->n <= 0) return mrbc_nil_value();

    mrbc_value ret = h->data[0];
    MEMCPY((uint8_t *)h->data, (uint8_t *)(h->data+1), sizeof(mrbc_value)*--h->n);

    return ret;
}

//================================================================
/*! getter

  @param  ary		pointer to target value
  @param  idx		index
  @return		mrbc_value data at index position or Nil.
*/
__GURU__ mrbc_value
_get(mrbc_value *ary, int idx)
{
    mrbc_array *h = ary->array;

    if (idx < 0) idx = h->n + idx;
    if (idx < 0 || idx >= h->n) return mrbc_nil_value();

    return h->data[idx];
}

//================================================================
/*! remove a data

  @param  ary		pointer to target value
  @param  idx		index
  @return			mrbc_value data at index position or Nil.
*/
__GURU__ mrbc_value
_remove(mrbc_value *ary, int idx)
{
    mrbc_array *h = ary->array;

    if (idx < 0) idx = h->n + idx;
    if (idx < 0 || idx >= h->n) return mrbc_nil_value();

    mrbc_value *p  = h->data + idx;
    mrbc_value ret = *p;
    if (idx < --h->n) {										// shrink by 1
    	int blksz = sizeof(mrbc_value) * (h->n - idx);
        MEMCPY((uint8_t *)p, (uint8_t *)(p+1), blksz);		// shift forward
    }
    return ret;
}

//================================================================
/*! get min, max value

  @param  ary		pointer to target value
  @param  pp_min_value	returns minimum mrbc_value
  @param  pp_max_value	returns maxmum mrbc_value
*/
__GURU__ void
_minmax(mrbc_value *ary, mrbc_value **pp_min_value, mrbc_value **pp_max_value)
{
    mrbc_array *h = ary->array;

    if (h->n==0) {
        *pp_min_value = NULL;
        *pp_max_value = NULL;
        return;
    }
    mrbc_value *p_min_value = h->data;
    mrbc_value *p_max_value = h->data;
    mrbc_value *p           = h->data;
    for (U32 i = 1; i < h->n; i++, p++) {
        if (mrbc_compare(p, p_min_value) < 0) p_min_value = p;
        if (mrbc_compare(p, p_max_value) > 0) p_max_value = p;
    }
    *pp_min_value = p_min_value;
    *pp_max_value = p_max_value;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size
  @return 	array object
*/
__GURU__ mrbc_value
mrbc_array_new(int size)
{
    mrbc_value ret = {.tt = GURU_TT_ARRAY};
    mrbc_array *h 	 = (mrbc_array *)mrbc_alloc(sizeof(mrbc_array));		// handle
    if (!h) return ret;		// ENOMEM

    mrbc_value *data = (mrbc_value *)mrbc_alloc(sizeof(mrbc_value) * size);	// buffer
    if (!data) {			// ENOMEM
        mrbc_free(h);
        return ret;
    }
    h->refc = 1;			// handle is referenced
    h->tt 	= GURU_TT_ARRAY;
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
mrbc_array_delete(mrbc_value *ary)
{
    mrbc_array *h = ary->array;
    mrbc_value *p = h->data;
    for (U32 i=0; i < h->n; i++, p++) {
    	mrbc_release(p);
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
mrbc_array_resize(mrbc_array *h, int size)
{
	assert(size > h->size);

    mrbc_value *d2 = (mrbc_value *)mrbc_realloc(h->data, sizeof(mrbc_value) * size);
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
mrbc_array_push(mrbc_value *ary, mrbc_value *set_val)
{
    mrbc_array *h = ary->array;

    if (h->n >= h->size) {
        int size = h->size + 6;
        if (mrbc_array_resize(h, size) != 0) {
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
mrbc_array_clear(mrbc_value *ary)
{
    mrbc_array *h = ary->array;
    mrbc_value *p = h->data;
    for (U32 i=0; i < h->n; i++, p++) {
    	mrbc_release(p);                      // CC: was dec_refc 20181101
    }
    h->n = 0;
}

//================================================================
/*! compare

  @param  v1	Pointer to mrbc_value
  @param  v2	Pointer to another mrbc_value
  @retval 0	v1==v2
  @retval plus	v1 >  v2
  @retval minus	v1 <  v2
*/
__GURU__ int
mrbc_array_compare(const mrbc_value *v0, const mrbc_value *v1)
{
	mrbc_array *h0 = v0->array;
	mrbc_array *h1 = v1->array;
	mrbc_value *d0 = h0->data;
	mrbc_value *d1 = h1->data;
    for (U32 i=0; ; i++) {
        if (i >= h0->n || i >= h1->n) break;

        int res = mrbc_compare(d0++, d1++);
        if (res != 0) return res;
    }
    return h0->n - h1->n;
}

//================================================================
/*! method new
 */
__GURU__ void
c_array_new(mrbc_value v[], U32 argc)
{
	mrbc_value ret;
    if (argc==0) {													// in case of new()
        ret = mrbc_array_new(0);
        if (ret.array==NULL) return;		// ENOMEM
    }
    else if (argc==1 && v[1].tt==GURU_TT_FIXNUM && v[1].i >= 0) {	// new(num)
        ret = mrbc_array_new(v[1].i);
        if (ret.array==NULL) return;		// ENOMEM

        mrbc_value nil = mrbc_nil_value();
        if (v[1].i > 0) {
            _set(&ret, v[1].i - 1, &nil);
        }
    }
    else if (argc==2 && v[1].tt==GURU_TT_FIXNUM && v[1].i >= 0) {	// new(num, value)
        ret = mrbc_array_new(v[1].i);
        if (ret.array==NULL) return;		// ENOMEM

        for (U32 i=0; i < v[1].i; i++) {
            mrbc_retain(&v[2]);
            _set(&ret, i, &v[2]);
        }
    }
    else {
    	ret = mrbc_nil_value();
        console_str("ArgumentError\n");	// raise?
    }
    SET_RETURN(ret);
}

//================================================================
/*! (operator) +
 */
__GURU__ void
c_array_add(mrbc_value v[], U32 argc)
{
    if (GET_TT_ARG(1) != GURU_TT_ARRAY) {
        console_str("TypeError\n");		// raise?
        return;
    }
    mrbc_array *h0 = v[0].array;
    mrbc_array *h1 = v[1].array;

    int h0sz = sizeof(mrbc_value) * h0->n;
    int h1sz = sizeof(mrbc_value) * h1->n;

    mrbc_value ret = mrbc_array_new(h0sz + h1sz);
    if (ret.array==NULL) return;		// ENOMEM

    MEMCPY((uint8_t *)(ret.array->data),        (const uint8_t *)h0->data, h0sz);
    MEMCPY((uint8_t *)(ret.array->data) + h0sz, (const uint8_t *)h1->data, h1sz);

    mrbc_value *p = ret.array->data;
    int         n = ret.array->n = h0->n + h1->n;
    for (U32 i=0; i<n; i++, p++) {
    	mrbc_retain(p);
    }
    mrbc_release(v+1);					// dec_refc v[1], free if not needed

    SET_RETURN(ret);
}

//================================================================
/*! (operator) []
 */
__GURU__ void
c_array_get(mrbc_value v[], U32 argc)
{
	mrbc_value ret;
    if (argc==1 && v[1].tt==GURU_TT_FIXNUM) {			// self[n] -> object | nil
        ret = _get(v, v[1].i);
        mrbc_retain(&ret);
    }
    else if (argc==2 &&			 						// self[idx, len] -> Array | nil
    		v[1].tt==GURU_TT_FIXNUM &&
    		v[2].tt==GURU_TT_FIXNUM) {
        int len = v->array->n;
        int idx = v[1].i;
        if (idx < 0) idx += len;
        if (idx < 0) goto DONE;

        int size = (v[2].i < (len - idx)) ? v[2].i : (len - idx);
        // min(v[2].i, (len - idx))
        if (size < 0) goto DONE;

        ret = mrbc_array_new(size);
        if (ret.array==NULL) return;		// ENOMEM

        for (U32 i = 0; i < size; i++) {
            mrbc_value val = _get(v, v[1].i + i);
            mrbc_retain(&val);
            mrbc_array_push(&ret, &val);
        }
    }
    else {
        console_na("case of Array#[]");
    	ret = mrbc_nil_value();
    }
DONE:
    SET_RETURN(ret);
}

//================================================================
/*! (operator) []=
 */
__GURU__ void
c_array_set(mrbc_value v[], U32 argc)
{
    if (argc==2 && v[1].tt==GURU_TT_FIXNUM) {	// self[n] = val
        _set(v, v[1].i, &v[2]);		// raise? IndexError or ENOMEM
    }
    else if (argc==3 &&							// self[n, len] = valu
    		v[1].tt==GURU_TT_FIXNUM &&
    		v[2].tt==GURU_TT_FIXNUM) {
        // TODO: not implement yet.
    }
    else {
        console_na("case of Array#[]=");
    }
}

//================================================================
/*! (method) clear
 */
__GURU__ void
c_array_clear(mrbc_value v[], U32 argc)
{
    mrbc_array_clear(v);
}

//================================================================
/*! (method) delete_at
 */
__GURU__ void
c_array_delete_at(mrbc_value v[], U32 argc)
{
	int n = GET_INT_ARG(1);

    SET_RETURN(_remove(v, n));
}

//================================================================
/*! (method) empty?
 */
__GURU__ void
c_array_empty(mrbc_value v[], U32 argc)
{
    SET_BOOL_RETURN(v->array->n==0);
}

//================================================================
/*! (method) size,length,count
 */
__GURU__ void
c_array_size(mrbc_value v[], U32 argc)
{
    SET_INT_RETURN(v->array->n);
}

//================================================================
/*! (method) index
 */
__GURU__ void
c_array_index(mrbc_value v[], U32 argc)
{
    mrbc_value *ret = v+1;
    
    mrbc_array *h = v->array;
    mrbc_value *p = h->data;
    for (U32 i = 0; i < h->n; i++, p++) {
        if (mrbc_compare(p, ret)==0) {
            SET_INT_RETURN(i);
            return;
        }
    }
    SET_NIL_RETURN();
}

//================================================================
/*! (method) first
 */
__GURU__ void c_array_first(mrbc_value v[], U32 argc)
{
    mrbc_value ret = _get(v, 0);
    mrbc_retain(&ret);
	SET_RETURN(ret);
}

//================================================================
/*! (method) last
 */
__GURU__ void
c_array_last(mrbc_value v[], U32 argc)
{
    mrbc_value ret = _get(v, -1);
    mrbc_retain(&ret);
	SET_RETURN(ret);
}

//================================================================
/*! (method) push
 */
__GURU__ void
c_array_push(mrbc_value v[], U32 argc)
{
    mrbc_array_push(v, v+1);	// raise? ENOMEM
    v[1].tt = GURU_TT_EMPTY;
}

//================================================================
/*! (method) pop
 */
__GURU__ void
c_array_pop(mrbc_value v[], U32 argc)
{
    if (argc==0) {									// pop() -> object | nil
        SET_RETURN(_pop(v));
    }
    else if (argc==1 && v[1].tt==GURU_TT_FIXNUM) {	// pop(n) -> Array | nil
        // TODO: not implement yet.
    }
    else {
    	console_str("case of Array#pop");
    }
}

//================================================================
/*! (method) unshift
 */
__GURU__ void
c_array_unshift(mrbc_value v[], U32 argc)
{
    _unshift(v, v+1);	// raise? IndexError or ENOMEM
    v[1].tt = GURU_TT_EMPTY;
}

//================================================================
/*! (method) shift
 */
__GURU__ void
c_array_shift(mrbc_value v[], U32 argc)
{
    if (argc==0) {									// shift() -> object | nil
        SET_RETURN(_shift(v));
    }
    else if (argc==1 && v[1].tt==GURU_TT_FIXNUM) {	// shift() -> Array | nil
        // TODO: not implement yet.
    }
    else {
    	console_na("case of Array#shift");
    }
}

//================================================================
/*! (method) dup
 */
__GURU__ void
c_array_dup(mrbc_value v[], U32 argc)
{
    mrbc_array *h0 = v[0].array;
    mrbc_value ret = mrbc_array_new(h0->n);
    mrbc_array *h1 = ret.array;
    if (!h1) return;		// ENOMEM

    int n = h1->n = h0->n;
    MEMCPY((uint8_t *)h1->data, (const uint8_t *)h0->data, n*sizeof(mrbc_value));

    mrbc_value *p = h1->data;
    for (U32 i=0; i<n; i++, p++) {
        mrbc_retain(p);
    }
    SET_RETURN(ret);
}

//================================================================
/*! (method) min
 */
__GURU__ void
c_array_min(mrbc_value v[], U32 argc)
{
    // Subset of Array#min, not support min(n).

    mrbc_value *p_min_value, *p_max_value;

    _minmax(v, &p_min_value, &p_max_value);
    if (p_min_value==NULL) SET_NIL_RETURN();
    else {
        mrbc_retain(p_min_value);
        SET_RETURN(*p_min_value);
    }
}

//================================================================
/*! (method) max
 */
__GURU__ void
c_array_max(mrbc_value v[], U32 argc)
{
    // Subset of Array#max, not support max(n).

    mrbc_value *p_min_value, *p_max_value;

    _minmax(v, &p_min_value, &p_max_value);
    if (p_max_value==NULL) SET_NIL_RETURN();
    else {
        mrbc_retain(p_max_value);
        SET_RETURN(*p_max_value);
    }
}

//================================================================
/*! (method) minmax
 */
__GURU__ void
c_array_minmax(mrbc_value v[], U32 argc)
{
    // Subset of Array#minmax, not support minmax(n).

    mrbc_value *p_min_value, *p_max_value;
    mrbc_value nil = mrbc_nil_value();
    mrbc_value ret = mrbc_array_new(2);

    _minmax(v, &p_min_value, &p_max_value);
    if (p_min_value==NULL) p_min_value = &nil;
    if (p_max_value==NULL) p_max_value = &nil;

    mrbc_retain(p_min_value);
    mrbc_retain(p_max_value);
    _set(&ret, 0, p_min_value);
    _set(&ret, 1, p_max_value);

    SET_RETURN(ret);
}

#if GURU_USE_STRING
__GURU__ void
_rfc(mrbc_value *str, mrbc_value *v)
{
	U8 buf[8];
	guru_sprintf(buf, "^%d_", v->self->refc);
	mrbc_string_append_cstr(str, buf);
}
//================================================================
/*! (method) inspect
 */
__GURU__ void
c_array_inspect(mrbc_value v[], U32 argc)
{
	mrbc_value ret  = mrbc_string_new((U8P)"[");
    if (!ret.str) {
    	SET_NIL_RETURN();
    	return;
    }
    mrbc_value vi, s1;
    int n = v->array->n;
    for (U32 i = 0; i < n; i++) {
        if (i != 0) mrbc_string_append_cstr(&ret, (U8P)", ");
        vi = _get(v, i);
        s1 = guru_inspect(v+argc, &vi);
        mrbc_string_append(&ret, &s1);
        mrbc_release(&s1);
    }
    mrbc_string_append_cstr(&ret, (U8P)"]");

    SET_RETURN(ret);
}

//================================================================
/*! (method) join
 */
__GURU__ void
c_array_join_1(mrbc_value v[], U32 argc,
                    mrbc_value *src, mrbc_value *ret, mrbc_value *separator)
{
	mrbc_array *h = src->array;
    if (h->n==0) return;

    int i   = 0;
    mrbc_value s1;
    while (1) {
        if (h->data[i].tt==GURU_TT_ARRAY) {
            c_array_join_1(v, argc, &h->data[i], ret, separator);
        }
        else {
            s1 = guru_inspect(v+argc, &h->data[i]);
            mrbc_string_append(ret, &s1);
            mrbc_release(&s1);					// free locally allocated memory
        }
        if (++i >= h->n) break;					// normal return.
        mrbc_string_append(ret, separator);
    }
}

__GURU__ void
c_array_join(mrbc_value v[], U32 argc)
{
    mrbc_value ret = mrbc_string_new(NULL);
    if (!ret.str) {
        SET_NIL_RETURN();
        return;
    }
    mrbc_value separator = (argc==0)
    		? mrbc_string_new((U8P)"")
    		: guru_inspect(v+argc, v+1);

    c_array_join_1(v, argc, v, &ret, &separator);
    mrbc_release(&separator);		            // release locally allocated memory

    SET_RETURN(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void
mrbc_init_class_array()
{
    mrbc_class *c = mrbc_class_array = guru_add_class("Array", mrbc_class_object);

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
