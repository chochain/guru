/*! @file
  @brief
  GURU Array and Lambda classes

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "guru.h"
#include "util.h"
#include "mmu.h"

#include "base.h"
#include "static.h"
#include "c_array.h"

#include "inspect.h"

/*
  function summary

 (constructor)
    guru_array_new

 (destructor)
    guru_array_del

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
    guru_array_clr
    guru_array_cmp
    guru_array_minmax
*/

//================================================================
/*! get size
 */
__GURU__ void
_resize(guru_array *h, U32 nsz)
{
    U32 sz = 0;
    if (nsz >= h->sz) {						// need resize?
        sz = nsz;
    }
    else if (h->n >= h->sz) {
        sz = h->n + 4;						// auto allocate extra 4 elements
    }
    if (sz==0) return;

    h->data = h->data
    	? guru_gr_realloc(h->data, sz)
        : guru_gr_alloc(sz);
    h->sz = sz;
    for (int i=h->n; i<sz; i++) {			// DEBUG: lazy fill here, instead of when resized
    	h->data[i] = EMPTY;
    }
}

//================================================================
/*! setter

  @param  ary		pointer to target value
  @param  idx		index
  @param  set_val	set value
*/
__GURU__ void
_set(GR *ary, S32 idx, GR *val)
{
    guru_array *h = GR_ARY(ary);
    U32 ndx = (idx < 0) ? h->sz + idx : idx;

    if (ndx >= h->sz) {
        _resize(h, ndx + 4);					// adjust array size
    }
    if (ndx < h->n) {
        ref_dec(&h->data[ndx]);					// release existing data
    }
    else {
    	while (h->n < ndx+1) {
    		h->data[h->n++] = NIL;				// filling the blanks
    	}
    }
    h->data[ndx] = *ref_inc(val);				// keep the reference to the value
}

__GURU__ void
_push(GR *ary, GR *set_val)
{
    guru_array *h = GR_ARY(ary);

    if (h->n >= h->sz) {
        _resize(h, h->sz + 6);
    }
    h->data[h->n++] = *ref_inc(set_val);
}
//================================================================
/*! pop a data from tail.

  @param  ary	pointer to target value
  @return		tail data or Nil
*/
__GURU__ GR
_pop(GR *ary)
{
    guru_array *h = GR_ARY(ary);

    if (h->n <= 0) return NIL;

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
_insert(GR *ary, S32 idx, GR *set_val)
{
    guru_array *h = GR_ARY(ary);
    U32 ndx = 1 + (idx < 0) ? h->sz+idx : idx;
    _resize(h, ndx);

    if (ndx < h->n) {										// move data
    	U32 sz = sizeof(GR)*(h->n - ndx);
        MEMCPY(h->data + ndx + 1, h->data + ndx, sz);		// rshift (copy backward, does this work?)
    }

    h->data[ndx] = *ref_inc(set_val);						// set data
    h->n++;

    if (ndx >= h->n) {										// clear empty cells
        for (int i = h->n-1; i < ndx; i++) {
            h->data[i] = NIL;
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
_unshift(GR *ary, GR *set_val)
{
    return _insert(ary, 0, set_val);
}

//================================================================
/*! removes the first data and returns it.

  @param  ary		pointer to target value
  @return		first data or Nil
*/
__GURU__ GR
_shift(GR *ary)
{
    guru_array *h = GR_ARY(ary);

    if (h->n <= 0) return NIL;

    GR *v = ref_dec(&h->data[0]);
    MEMCPY(h->data, h->data + 1, sizeof(GR)*(--h->n));		// lshift

    return *v;
}

//================================================================
/*! getter (no change to ref count)

  @param  ary		pointer to target value
  @param  idx		index
  @return			GR data at index position or Nil.
*/
__GURU__ GR
_get(GR *r, U32 ri, S32 n1, S32 n2)
{
	guru_array *h = GR_ARY(r);

	n1 += (n1 < 0) ? h->n : 0;
	if (ri<2) return (n1 < h->n) ? h->data[n1] : NIL;		// single element

	n2 += (n2 < 0) ? h->n : 0;								// sliced array

	U32 da = h->n - n1,										// remaining elements
		dn = n2-n1+1,
		sz = (dn > da) ? da : dn;
	GR ret = guru_array_new(sz);
    for (int i=n1; i <= n2 && i < h->n; i++) {
    	_push(&ret, &h->data[i]);
    }
    return ret;
}

//================================================================
/*! remove a data

  @param  ary		pointer to target value
  @param  idx		index
  @return			GR data at index position or Nil.
*/
__GURU__ GR
_remove(GR *ary, S32 idx)
{
    guru_array *h = GR_ARY(ary);
    U32 ndx = (idx < 0) ? h->n + idx : idx;

    if (ndx >= h->n) return NIL;
    h->n--;

    GR  *r  = ref_dec(&h->data[ndx]);			// release the object
    GR  ret = *r;								// copy return value before it's overwritten
    U32 nx = h->n - ndx;						// number of elements to move
    if (nx) MEMCPY(r, r+1, nx*sizeof(GR));		// lshift

    return ret;									// return the deleted item
}

//================================================================
/*! get min, max value

  @param  ary			pointer to target value
  @param  pp_min_value	returns minimum GR
  @param  pp_max_value	returns maxmum GR
*/
__GURU__ void
_minmax(GR *ary, GR **pp_min_value, GR **pp_max_value)
{
    guru_array *h = GR_ARY(ary);

    if (h->n==0) {
        *pp_min_value = NULL;
        *pp_max_value = NULL;
        return;
    }
    GR *p_min_value = h->data;
    GR *p_max_value = h->data;
    GR *p           = h->data;
    for (int i = 1; i < h->n; i++, p++) {
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
__GURU__ GR
guru_array_new(U32 sz)
{
    guru_array *h = (guru_array *)guru_alloc(sizeof(guru_array));		// handle

    h->rc   = 1;
    h->n  	= 0;
    h->sz   = sz;
    h->data = sz ? guru_gr_alloc(sz) : NULL;							// empty array?

    GR r { GT_ARRAY, ACL_HAS_REF, 0, MEMOFF(h) };

    return r;
}

//================================================================
/*! destructor

  @param  ary	pointer to target value
*/
__GURU__ void
guru_array_del(GR *ary)
{
    guru_array 	*h = GR_ARY(ary);
    GR 			*p = h->data;
    for (int i=0; i < h->n; i++, ref_dec(p++));		// released element from the array

    if (h->data) guru_free(h->data);				// release data block
    guru_free(h);									// release header block
}

//================================================================
/*! resize buffer

  @param  ary	pointer to target value
  @param  size	size
  @return	error_code
*/
__GURU__ void
guru_array_resize(guru_array *h, U32 new_sz)
{
	_resize(h, new_sz);
}

//================================================================
/*! push a data to tail

  @param  ary		pointer to target value
  @param  set_val	set value
  @return			error_code
*/
__GURU__ void
guru_array_push(GR *ary, GR *set_val)
{
	_push(ary, set_val);
}

//================================================================
/*! clear all

  @param  ary		pointer to target value
*/
__GURU__ void
guru_array_clr(GR *ary)
{
    guru_array *h = GR_ARY(ary);
    GR *p = h->data;
    for (int i=0; i < h->n; i++, p++) {
    	ref_dec(p);
    }
    h->n = 0;
}

//================================================================
/*! compare

  @param  v1	Pointer to GR
  @param  v2	Pointer to another GR
  @retval 0	v1==v2
  @retval plus	v1 >  v2
  @retval minus	v1 <  v2
*/
__GURU__ S32
guru_array_cmp(const GR *a0, const GR *a1)
{
	guru_array *h0 = GR_ARY(a0);
	guru_array *h1 = GR_ARY(a1);

	S32 dif = (h0->n - h1->n);
	if (dif) return dif;

	GR *d0 = h0->data;
	GR *d1 = h1->data;

	for (int i=0; i < h0->n && i < h1->n; i++) {
        S32 res = guru_cmp(d0++, d1++);
        if (res != 0) return res;
    }
    return 0;
}

//================================================================
/*! method new
 */
__CFUNC__
ary_new(GR r[], U32 ri)
{
	GR ret;
    if (ri==0) {											// in case of new()
        ret = guru_array_new(0);
    }
    else if (ri==1 && r[1].gt==GT_INT && r[1].i >= 0) {		// new(num)
        ret = guru_array_new(r[1].i);

        GR nil = NIL;
        if (r[1].i > 0) {
            _set(&ret, r[1].i - 1, &nil);
        }
    }
    else if (ri==2 && r[1].gt==GT_INT && r[1].i >= 0) {		// new(num, value)
        ret = guru_array_new(r[1].i);
        for (int i=0; i < r[1].i; i++) {
            _set(&ret, i, &r[2]);
        }
    }
    else {
    	ret = NIL;
    	NA("ArgumentError");
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (operator) +
 */
__CFUNC__
ary_add(GR r[], U32 ri)
{
    ASSERT(r[0].gt==GT_ARRAY && r[1].gt==GT_ARRAY);		// array only (for now)

    guru_array 	*h0 = GR_ARY(r), 	*h1 = GR_ARY(r+1);
    U32 		n0  = h0->n, 		n1  = h1->n;

    GR ret = guru_array_new(n0 + n1);		// new array with ref count already set to 1
    GR *ra = GR_ARY(&ret)->data;

    MEMCPY(ra,      h0->data, sizeof(GR) * n0);
    MEMCPY(ra + n0, h1->data, sizeof(GR) * n1);

    GR_ARY(&ret)->n = n0 + n1;				// reset element count

    RETURN_VAL(ret);						// both array will be released by caller's _wipe_stack
}

//================================================================
/*! (operator) -
 */
__CFUNC__
ary_sub(GR r[], U32 ri)
{
    ASSERT(r[0].gt==GT_ARRAY && r[1].gt==GT_ARRAY);		// array only (for now)

    guru_array 	*h0 = GR_ARY(r), 	*h1 = GR_ARY(r+1);
    U32 		n0  = h0->n,	  	n1  = h1->n;

    GR ret = guru_array_new(n0);			// TODO: shrink after adding elements

    GR *v0 = h0->data;
    for (int i=0; i < n0; i++, v0++) {
    	GR *v1 = h1->data;					// scan thrugh v1 array to find matching elements
    	U32 j;
    	for (j=0; j < n1 && guru_cmp(v0, v1++); j++);
    	if (j>=n1) _push(&ret, v0);			// v0 does not belong to any of v1
    }
    RETURN_VAL(ret);						// both array will be released by caller's _wipe_stack
}

//================================================================
/*! (operator) []
 */
__CFUNC__
ary_get(GR r[], U32 ri)
{
	GR ret;
	if (r->gt==GT_ARRAY) {
		ret = _get(r, ri, r[1].i, r[1].i+r[2].i-1);
	}
	else {
        ret = guru_array_new(ri);
        for (int i=0; i < ri; i++) {
            _set(&ret, i, r+1+i);
        }
	}
	RETURN_VAL(ret);
}

//================================================================
/*! (operator) []=
 */
__CFUNC__
ary_set(GR r[], U32 ri)
{
	GT gt1 = r[1].gt;
	GT gt2 = r[2].gt;
    if (ri==2 && gt1==GT_INT) {			// self[n] = val
        _set(r, r[1].i, &r[2]);
    }
    else if (ri==3 &&					// self[n, len] = valu
    		gt1==GT_INT &&
    		gt2==GT_INT) {
    	NA("array[i,n]");
    }
    else {
        NA("case of Array#[]=");
    }
}

//================================================================
/*! (method) clear
 */
__CFUNC__
ary_clr(GR r[], U32 ri)
{
    guru_array_clr(r);
}

//================================================================
/*! (method) delete_at
 */
__CFUNC__
ary_del_at(GR r[], U32 ri)
{
	S32 n = r[1].i;

    RETURN_VAL(_remove(r, n));
}

//================================================================
/*! (method) empty?
 */
__CFUNC__
ary_empty(GR r[], U32 ri)
{
    RETURN_BOOL(GR_ARY(r)->n==0);
}

//================================================================
/*! (method) size,length,count
 */
__CFUNC__
ary_size(GR r[], U32 ri)
{
    RETURN_INT(GR_ARY(r)->n);
}

__GURU__ S32
_index(GR r[])
{
    guru_array *h = GR_ARY(r);
    GR *p = h->data;
    for (int i=0; i < h->n; i++, p++) {
        if (guru_cmp(p, r+1)==0) {
            return i;
        }
    }
    return -1;
}

//================================================================
/*! (method) index
 */
__CFUNC__
ary_index(GR r[], U32 ri)
{
	S32 i = _index(r);
	if (i>=0) { RETURN_INT(i); }
	else      { RETURN_NIL();  }
}

//================================================================
/*! (method) first
 */
__CFUNC__
ary_first(GR r[], U32 ri)
{
	U32 n = r[1].gt==GT_INT;
	GR  ret = _get(r, n ? 2 : 1, 0, n ? r[1].i-1 : 0);
	RETURN_VAL(ret);
}

//================================================================
/*! (method) last
 */
__CFUNC__
ary_last(GR r[], U32 ri)
{
	U32 n = r[1].gt==GT_INT;
	GR  ret = _get(r, n ? 2 : 1, n ? -r[1].i : -1, -1);
	RETURN_VAL(ret);
}

//================================================================
/*! (method) push
 */
__CFUNC__
ary_push(GR r[], U32 ri)
{
    guru_array_push(r, r+1);				// raise? ENOMEM
}

//================================================================
/*! (method) pop
 */
__CFUNC__
ary_pop(GR r[], U32 ri)
{
    if (ri==0) {							// pop() -> object | nil
        RETURN_VAL(_pop(r));
    }
    else if (ri==1 && r[1].gt==GT_INT) {	// pop(n) -> Array | nil
        NA("pop(n)");						// TODO: loop
    }
    else {
    	NA("case of Array#pop");
    }
}

//================================================================
/*! (method) pop
 */
__CFUNC__
ary_reverse(GR r[], U32 ri)
{
	guru_array *a = GR_ARY(r);
	GR ret = guru_array_new(a->n);

	GR *d  = a->data + a->n - 1;
	for (int i=0; i<a->n; i++, d--) {
    	guru_array_push(&ret, d);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) unshift
 */
__CFUNC__
ary_unshift(GR r[], U32 ri)
{
    _unshift(r, r+1);						// raise? IndexError or ENOMEM
}

//================================================================
/*! (method) shift
 */
__CFUNC__
ary_shift(GR r[], U32 ri)
{
    if (ri==0) {							// shift() -> object | nil
        RETURN_VAL(_shift(r));
    }
    else if (ri==1 && r[1].gt==GT_INT) {	// shift() -> Array | nil
        NA("shift(n)");						// TODO: loop
    }
    else {
    	NA("case of Array#shift");
    }
}

//================================================================
/*! (method) dup
 */
__CFUNC__
ary_dup(GR r[], U32 ri)
{
    guru_array *h0 = GR_ARY(r);

    GR ret = guru_array_new(h0->n);		// create new array
    GR *p0 = h0->data;
    for (int i=0; i < h0->n; i++, p0++) {
    	_set(&ret, i, p0);				// shallow copy
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) include?
 */
__CFUNC__
ary_include(GR r[], U32 ri)
{
    if (_index(r)<0) RETURN_FALSE()
    else 		     RETURN_TRUE();
}


//================================================================
/*! (method) min
 */
__CFUNC__
ary_min(GR r[], U32 ri)
{
    // Subset of Array#min, not support min(n).
    GR *min, *max;

    _minmax(r, &min, &max);

    if (min) RETURN_VAL(*min);

    RETURN_NIL();
}

//================================================================
/*! (method) max
 */
__CFUNC__
ary_max(GR r[], U32 ri)
{
    // Subset of Array#max, not support max(n).
    GR *min, *max;

    _minmax(r, &min, &max);
    if (max) RETURN_VAL(*max);

    RETURN_NIL();
}

//================================================================
/*! (method) minmax
 */
__CFUNC__
ary_minmax(GR r[], U32 ri)
{
    GR nil = NIL;
    GR ret = guru_array_new(2);
    GR *min, *max;

    _minmax(r, &min, &max);
    if (min==NULL) min = &nil;
    if (max==NULL) max = &nil;

    _set(&ret, 0, min);
    _set(&ret, 1, max);

    RETURN_VAL(ret);
}

//================================================================
/*! initialize
 */
__GURU__ __const__ Vfunc ary_vtbl[] = {
	{ "new",       ary_new		},
	{ "+",         ary_add		},
	{ "-",		   ary_sub      },
	{ "[]",        ary_get		},
	{ "at",        ary_get		},
	{ "size",      ary_size		},
	{ "length",    ary_size		},
	{ "count",     ary_size		},
	{ "index",     ary_index	},
	{ "first",     ary_first	},
	{ "last",      ary_last		},
	{ "empty?",    ary_empty	},
	{ "min",       ary_min		},
	{ "max",       ary_max		},
	{ "minmax",    ary_minmax	},

	{ "[]=",       ary_set		},
	{ "<<",        ary_push		},
	{ "clear",     ary_clr		},
	{ "delete_at", ary_del_at	},
	{ "push",      ary_push		},
	{ "pop",       ary_pop		},
	{ "shift",     ary_shift	},
	{ "unshift",   ary_unshift	},
	{ "dup",       ary_dup		},
	{ "include?",  ary_include  },
	{ "reverse",   ary_reverse  },

	// reference to string, the following functions are implemented in inspect.cu
	{ "join",      ary_join		},
	{ "inspect",   gr_to_s		},
	{ "to_s",      gr_to_s		},
};

__GURU__ void
guru_init_class_array()
{
    guru_rom_add_class(GT_ARRAY, "Array", GT_OBJ, ary_vtbl, VFSZ(ary_vtbl));
    guru_register_func(GT_ARRAY, NULL, guru_array_del, guru_array_cmp);
}
