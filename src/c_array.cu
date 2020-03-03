/*! @file
  @brief
  GURU Array and Lambda classes

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "vm_config.h"
#include "guru.h"
#include "mmu.h"
#include "class.h"
#include "util.h"
#include "value.h"

#include "vm.h"
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
_resize(guru_array *h, U32 ndx)
{
    U32 n = 0;
    if (ndx >= h->sz) {						// need resize?
        n = ndx;
    }
    else if (h->n >= h->sz) {
        n = h->n + 4;						// auto allocate extra 4 elements
    }
    if (n==0) return;

    h->data = h->data
    	? guru_gv_realloc(h->data, n)
        : guru_gv_alloc(n);
    h->sz = n;
    for (U32 i=h->n; i<n; i++) {			// DEBUG: lazy fill here, instead of when resized
    	h->data[i] = EMPTY();
    }
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
    U32 ndx = (idx < 0) ? h->sz + idx : idx;

    if (ndx >= h->sz) {
        _resize(h, ndx + 4);					// adjust array size
    }
    if (ndx < h->n) {
        ref_dec(&h->data[ndx]);					// release existing data
    }
    else {
    	while (h->n < ndx+1) {
    		h->data[h->n++] = NIL();			// filling the blanks
    	}
    }
    h->data[ndx] = *ref_inc(val);				// keep the reference to the value
}

__GURU__ void
_push(GV *ary, GV *set_val)
{
    guru_array *h = ary->array;

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
__GURU__ GV
_pop(GV *ary)
{
    guru_array *h = ary->array;

    if (h->n <= 0) return NIL();

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
    U32 ndx = 1 + (idx < 0) ? h->sz+idx : idx;
    _resize(h, ndx);

    if (ndx < h->n) {										// move data
    	U32 sz = sizeof(GV)*(h->n - ndx);
        MEMCPY(h->data + ndx + 1, h->data + ndx, sz);		// rshift (copy backward, does this work?)
    }

    h->data[ndx] = *ref_inc(set_val);						// set data
    h->n++;

    if (ndx >= h->n) {										// clear empty cells
        for (U32 i = h->n-1; i < ndx; i++) {
            h->data[i] = NIL();
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

    if (h->n <= 0) return NIL();

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
_get(GV *v, U32 vi, S32 n1, S32 n2)
{
	guru_array *h = v->array;

	n1 += (n1 < 0) ? h->n : 0;
	if (vi<2) return (n1 < h->n) ? h->data[n1] : NIL();		// single element

	n2 += (n2 < 0) ? h->n : 0;								// sliced array

	U32 da = h->n - n1,										// remaining elements
		dn = n2-n1+1,
		sz = (dn > da) ? da : dn;
	GV ret = guru_array_new(sz);
    for (U32 i=n1; i <= n2 && i < h->n; i++) {
    	_push(&ret, &h->data[i]);
    }
    return ret;
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

    if (ndx >= h->n) return NIL();
    h->n--;

    GV  *r  = ref_dec(&h->data[ndx]);			// release the object
    GV  ret = *r;								// copy return value before it's overwritten
    U32 nx = h->n - ndx;						// number of elements to move
    if (nx) MEMCPY(r, r+1, nx*sizeof(GV));		// lshift

    return ret;									// return the deleted item
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
guru_array_new(U32 sz)
{
    GV v; { v.gt=GT_ARRAY; v.acl=ACL_HAS_REF; }

    guru_array *h = (guru_array *)guru_alloc(sizeof(guru_array));		// handle

    h->rc   = 1;
    h->n  	= 0;
    h->sz   = sz;
    h->data = sz ? guru_gv_alloc(sz) : NULL;							// empty array?

    v.array = h;

    return v;
}

//================================================================
/*! destructor

  @param  ary	pointer to target value
*/
__GURU__ void
guru_array_del(GV *ary)
{
    guru_array 	*h = ary->array;
    GV 			*p = h->data;
    for (U32 i=0; i < h->n; i++, ref_dec(p++));		// released element from the array

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
guru_array_push(GV *ary, GV *set_val)
{
	_push(ary, set_val);
}

//================================================================
/*! clear all

  @param  ary		pointer to target value
*/
__GURU__ void
guru_array_clr(GV *ary)
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
__GURU__ S32
guru_array_cmp(const GV *a0, const GV *a1)
{
	guru_array *h0 = a0->array;
	guru_array *h1 = a1->array;

	S32 dif = (h0->n - h1->n);
	if (dif) return dif;

	GV *d0 = h0->data;
	GV *d1 = h1->data;

	for (U32 i=0; i < h0->n && i < h1->n; i++) {
        S32 res = guru_cmp(d0++, d1++);
        if (res != 0) return res;
    }
    return 0;
}

//================================================================
/*! method new
 */
__CFUNC__
ary_new(GV v[], U32 vi)
{
	GV ret;
    if (vi==0) {											// in case of new()
        ret = guru_array_new(0);
    }
    else if (vi==1 && v[1].gt==GT_INT && v[1].i >= 0) {		// new(num)
        ret = guru_array_new(v[1].i);

        GV nil = NIL();
        if (v[1].i > 0) {
            _set(&ret, v[1].i - 1, &nil);
        }
    }
    else if (vi==2 && v[1].gt==GT_INT && v[1].i >= 0) {		// new(num, value)
        ret = guru_array_new(v[1].i);
        for (U32 i=0; i < v[1].i; i++) {
            _set(&ret, i, &v[2]);
        }
    }
    else {
    	ret = NIL();
    	guru_na("ArgumentError");
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (operator) +
 */
__CFUNC__
ary_add(GV v[], U32 vi)
{
    ASSERT(v[0].gt==GT_ARRAY && v[1].gt==GT_ARRAY);		// array only (for now)

    guru_array 	*h0 = v[0].array, 	*h1 = v[1].array;
    U32 		n0  = h0->n, 		n1  = h1->n;

    GV ret = guru_array_new(n0 + n1);		// new array with ref count already set to 1
    GV *ra = ret.array->data;

    MEMCPY(ra,      h0->data, sizeof(GV) * n0);
    MEMCPY(ra + n0, h1->data, sizeof(GV) * n1);

    ret.array->n = n0 + n1;				// reset element count

    RETURN_VAL(ret);					// both array will be released by caller's _wipe_stack
}

//================================================================
/*! (operator) -
 */
__CFUNC__
ary_sub(GV v[], U32 vi)
{
    ASSERT(v[0].gt==GT_ARRAY && v[1].gt==GT_ARRAY);		// array only (for now)

    guru_array 	*h0 = v[0].array, 	*h1 = v[1].array;
    U32 		n0  = h0->n,	  	n1  = h1->n;

    GV ret = guru_array_new(n0);		// TODO: shrink after adding elements

    GV *v0 = v[0].array->data;
    for (U32 i=0; i < n0; i++, v0++) {
    	GV *v1 = v[1].array->data;		// scan thrugh v1 array to find matching elements
    	U32 j;
    	for (j=0; j < n1 && guru_cmp(v0, v1++); j++);
    	if (j>=n1) _push(&ret, v0);
    }
    RETURN_VAL(ret);					// both array will be released by caller's _wipe_stack
}

//================================================================
/*! (operator) []
 */
__CFUNC__
ary_get(GV v[], U32 vi)
{
	GV ret;
	if (v->gt==GT_ARRAY) {
		ret = _get(v, vi, v[1].i, v[1].i+v[2].i-1);
	}
	else {
        ret = guru_array_new(vi);
        for (U32 i=0; i < vi; i++) {
            _set(&ret, i, v+1+i);
        }
	}
	RETURN_VAL(ret);
}

//================================================================
/*! (operator) []=
 */
__CFUNC__
ary_set(GV v[], U32 vi)
{
	GT gt1 = v[1].gt;
	GT gt2 = v[2].gt;
    if (vi==2 && gt1==GT_INT) {			// self[n] = val
        _set(v, v[1].i, &v[2]);
    }
    else if (vi==3 &&					// self[n, len] = valu
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
__CFUNC__
ary_clr(GV v[], U32 vi)
{
    guru_array_clr(v);
}

//================================================================
/*! (method) delete_at
 */
__CFUNC__
ary_del_at(GV v[], U32 vi)
{
	S32 n = v[1].i;

    RETURN_VAL(_remove(v, n));
}

//================================================================
/*! (method) empty?
 */
__CFUNC__
ary_empty(GV v[], U32 vi)
{
    RETURN_BOOL(v->array->n==0);
}

//================================================================
/*! (method) size,length,count
 */
__CFUNC__
ary_size(GV v[], U32 vi)
{
    RETURN_INT(v->array->n);
}

__GURU__ S32
_index(GV v[])
{
    guru_array *h = v->array;
    GV *p = h->data;
    for (U32 i=0; i < h->n; i++, p++) {
        if (guru_cmp(p, v+1)==0) {
            return i;
        }
    }
    return -1;
}

//================================================================
/*! (method) index
 */
__CFUNC__
ary_index(GV v[], U32 vi)
{
	S32 i = _index(v);
	if (i>=0) { RETURN_INT(i); }
	else      { RETURN_NIL();  }
}

//================================================================
/*! (method) first
 */
__CFUNC__
ary_first(GV v[], U32 vi)
{
	U32 n = v[1].gt==GT_INT;
	GV  ret = _get(v, n ? 2 : 1, 0, n ? v[1].i-1 : 0);
	RETURN_VAL(ret);
}

//================================================================
/*! (method) last
 */
__CFUNC__
ary_last(GV v[], U32 vi)
{
	U32 n = v[1].gt==GT_INT;
	GV  ret = _get(v, n ? 2 : 1, n ? -v[1].i : -1, -1);
	RETURN_VAL(ret);
}

//================================================================
/*! (method) push
 */
__CFUNC__
ary_push(GV v[], U32 vi)
{
    guru_array_push(v, v+1);				// raise? ENOMEM
}

//================================================================
/*! (method) pop
 */
__CFUNC__
ary_pop(GV v[], U32 vi)
{
    if (vi==0) {							// pop() -> object | nil
        RETURN_VAL(_pop(v));
    }
    else if (vi==1 && v[1].gt==GT_INT) {	// pop(n) -> Array | nil
        guru_na("pop(n)");					// TODO: loop
    }
    else {
    	guru_na("case of Array#pop");
    }
}

//================================================================
/*! (method) pop
 */
__CFUNC__
ary_reverse(GV v[], U32 vi)
{
	guru_array *a = v->array;
	GV ret = guru_array_new(a->n);

	GV *r  = a->data + a->n - 1;
    for (U32 i=0; i<a->n; i++, r--) {
    	guru_array_push(&ret, r);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) unshift
 */
__CFUNC__
ary_unshift(GV v[], U32 vi)
{
    _unshift(v, v+1);						// raise? IndexError or ENOMEM
}

//================================================================
/*! (method) shift
 */
__CFUNC__
ary_shift(GV v[], U32 vi)
{
    if (vi==0) {							// shift() -> object | nil
        RETURN_VAL(_shift(v));
    }
    else if (vi==1 && v[1].gt==GT_INT) {	// shift() -> Array | nil
        guru_na("shift(n)");				// TODO: loop
    }
    else {
    	guru_na("case of Array#shift");
    }
}

//================================================================
/*! (method) dup
 */
__CFUNC__
ary_dup(GV v[], U32 vi)
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
/*! (method) include?
 */
__CFUNC__
ary_include(GV v[], U32 vi)
{
    if (_index(v)<0) RETURN_FALSE()
    else 		     RETURN_TRUE();
}


//================================================================
/*! (method) min
 */
__CFUNC__
ary_min(GV v[], U32 vi)
{
    // Subset of Array#min, not support min(n).
    GV *min, *max;

    _minmax(v, &min, &max);

    if (min) RETURN_VAL(*min);

    RETURN_NIL();
}

//================================================================
/*! (method) max
 */
__CFUNC__
ary_max(GV v[], U32 vi)
{
    // Subset of Array#max, not support max(n).
    GV *min, *max;

    _minmax(v, &min, &max);
    if (max) RETURN_VAL(*max);

    RETURN_NIL();
}

//================================================================
/*! (method) minmax
 */
__CFUNC__
ary_minmax(GV v[], U32 vi)
{
    GV nil = NIL();
    GV ret = guru_array_new(2);
    GV *min, *max;

    _minmax(v, &min, &max);
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

	{ "join",      gv_join		},
	{ "inspect",   gv_to_s		},
	{ "to_s",      gv_to_s		},
};

__GURU__ void
guru_init_class_array()
{
    guru_rom_set_class(GT_ARRAY, "Array", GT_OBJ, ary_vtbl, VFSZ(ary_vtbl));
}
