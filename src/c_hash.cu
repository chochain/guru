/*! @file
  @brief
  GURU Hash class

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "guru.h"
#include "util.h"
#include "mmu.h"

#include "base.h"
#include "static.h"
#include "c_array.h"
#include "c_hash.h"

#include "inspect.h"

/*
  function summary

 (constructor)
    guru_hash_new

 (destructor)
    guru_hash_del

 (setter)
  --[name]-------------[arg]---[ret]-------
    guru_hash_set		*K,*V	int

 (getter)
  --[name]-------------[arg]---[ret]---[note]------------------------
    guru_hash_get		*K		T		Data remains in the container
    guru_hash_remove	*K		T		Data does not remain in the container
    guru_hash_i_next		   *T		Data remains in the container
*/


//================================================================
/*! get size
 */
__GURU__ __INLINE__ int
_size(const GR *kv) {
    return GR_HSH(kv)->n >> 1;
}

__GURU__ __INLINE__ GR*
_data(const GR *kv) {
	return GR_HSH(kv)->data;
}

//================================================================
/*! search key

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	pointer to found key or NULL(not found).
*/
__GURU__ GR*
_search(const GR *kv, const GR *key)
{
    GR  *p = GR_HSH(kv)->data;
    U32  n = _size(kv);

    for (int i=0; i < n; i++, p+=2) {
        if (guru_cmp(p, key)==0) return p;
    }
    return NULL;
}

//================================================================
/*! setter

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @param  val	pointer to value
  @return		error_code
*/
__GURU__ void
_set(GR *kv, GR *key, GR *val)
{
    GR *r = _search(kv, key);
    if (r==NULL) {					// key not found, create new kv pair
        guru_array_push(kv, key);	// push into array tail (ref counter += 1)
        guru_array_push(kv, val);
    }
    else {
    	ref_dec(r);					// release previous kv elements
    	ref_dec(r+1);
        *(r)   = *ref_inc(key);
        *(r+1) = *ref_inc(val);
    }
}

//================================================================
/*! getter

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	GR data at key position or Nil.
*/
__GURU__ GR
_get(GR *kv, GR *key)
{
    GR *r = _search(kv, key);

    return r ? *(r+1) : NIL;
}

//================================================================
/*! remove a data

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	removed data or Nil
*/
__GURU__ GR
_remove(GR *kv, GR *key)
{
    GR *r = _search(kv, key);
    if (r==NULL) return NIL;

    ref_dec(r);						// CC: was dec_refc 20181101
    GR ret = *(r+1);				// value
    guru_hash  *h  = GR_HSH(kv);
    h->n -= 2;

    MEMCPY(r, (r+2), U8POFF(h->data + h->n, r));

    // TODO: re-index hash table if need.

    return ret;
}

//================================================================
/*! clear all

  @param  hash	pointer to target hash
*/
__GURU__ void
_clr(GR *kv)
{
    guru_array_clr(kv);

    // TODO: re-index hash table if need.
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size
  @return 	hash object
*/
__GURU__ GR
guru_hash_new(int sz)
{
    /*
      Allocate handle and data buffer.
    */
    guru_hash *h = (guru_hash *)guru_alloc(sizeof(guru_hash));

    h->rc   = 1;
    h->n  	= 0;
    h->sz	= sz<<1;		// double the array size for (k,v) pairs
    h->data = sz ? guru_gr_alloc(sz<<1) : NULL;

    GR r { GT_HASH, ACL_HAS_REF, 0, MEMOFF(h) };

    return r;
}

//================================================================
/*! duplicate

  @param  vm	pointer to VM.
  @param  src	pointer to target hash.
*/
__GURU__ GR
_hash_dup(const GR *kv)
{
	int n   = _size(kv);
    GR  ret = guru_hash_new(n);

    GR  *d  = _data(&ret);
    GR  *s  = _data(kv);
    U32 n2  = GR_HSH(&ret)->n = n<<1;	// n pairs (k,v)
    for (int i=0; i < n2; i++) {
    	*d++ = *ref_inc(s++);			// referenced by the new hash now
    }
    return ret;
}

//================================================================
/*! destructor

  @param  hash	pointer to target value
*/
__GURU__ void
guru_hash_del(GR *kv)
{
    guru_array_del(kv);		// free content
}


//================================================================
/*! compare

  @param  r0	Pointer to GR
  @param  r1 	Pointer to another GR
  @retval 0	r0==r1
  @retval 1	r0 != r1
*/
__GURU__ S32
guru_hash_cmp(const GR *r0, const GR *r1)
{
	int n0 = _size(r0);
    if (n0 != _size(r1)) 			return 1;	// size different

    GR *p0 = _data(r0);
    for (int i=0; i < n0; i++, p0+=2) {			// walk the hash element by element
        GR *p1 = _search(r1, p0);				// check key
        if (p1==NULL) 				return 1;	// no key found
        if (guru_cmp(p0+1, p1+1)) 	return 1;	// compare data
    }
    return 0;									// matched
}

//================================================================
/*! (method) new
 */
__CFUNC__
hsh_new(GR r[], S32 ri)
{
	GR ret;
    if (ri==0) {											// in case of new()
        ret = guru_hash_new(0);
    }
    else if (ri==1 && r[1].gt==GT_INT && r[1].i >= 0) {		// new(num)
        ret = guru_hash_new(r[1].i);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (operator) []
 */
__CFUNC__
hsh_get(GR r[], S32 ri)
{
	ASSERT(ri==1);
	GR ret = _get(r, r+1);

    RETURN_VAL(ret);
}

//================================================================
/*! (operator) []=
 */
__CFUNC__
hsh_set(GR r[], S32 ri)
{
	ASSERT(ri==2);
    _set(r, r+1, r+2);		// k + v

    *(r+1) = EMPTY;
    *(r+2) = EMPTY;
}


//================================================================
/*! (method) clear
 */
__CFUNC__
hsh_clr(GR r[], S32 ri)
{
    _clr(r);
}

//================================================================
/*! (method) dup
 */
__CFUNC__
hsh_dup(GR r[], S32 ri)
{
    RETURN_VAL(_hash_dup(r));
}

//================================================================
/*! (method) delete
 */
__CFUNC__
hsh_del(GR r[], S32 ri)
{
    // TODO : now, support only delete(key) -> object
    // TODO: re-index hash table if need.
	RETURN_VAL(_remove(r, r+1));
}

//================================================================
/*! (method) empty?
 */
__CFUNC__
hsh_empty(GR r[], S32 ri)
{
    RETURN_BOOL(_size(r)==0);
}

//================================================================
/*! (method) has_key?
 */
__CFUNC__
hsh_has_key(GR r[], S32 ri)
{
    RETURN_BOOL(_search(r, r+1)!=NULL);
}

//================================================================
/*! (method) has_value?
 */
__CFUNC__
hsh_has_value(GR r[], S32 ri)
{
    GR  *p = _data(r);
    U32 n  = _size(r);
    for (int i=0; i<n; i++, p+=2) {
        if (guru_cmp(p+1, r+1)==0) {	// value to value
            RETURN_BOOL(1);
        }
    }
    RETURN_BOOL(0);
}

//================================================================
/*! (method) key
 */
__CFUNC__
hsh_key(GR r[], S32 ri)
{
    GR  *p = _data(r);
    U32 n  = _size(r);
    for (int i=0; i<n; i++, p+=2) {
        if (guru_cmp(p+1, r+1)==0) {
            RETURN_VAL(*p);
        }
    }
    RETURN_NIL();
}

//================================================================
/*! (method) keys
 */
__CFUNC__
hsh_keys(GR r[], S32 ri)
{
    GR *p  = _data(r);
    int n  = _size(r);
    GR ret = guru_array_new(n);

    for (int i=0; i<n; i++, p+=2) {
        guru_array_push(&ret, p);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) size,length,count
 */
__CFUNC__
hsh_size(GR r[], S32 ri)
{
    RETURN_INT(_size(r));
}

//================================================================
/*! (method) merge
 */
__CFUNC__
hsh_merge(GR r[], S32 ri)		// non-destructive merge
{
	ASSERT((r+1)->gt==GT_HASH);	// other types not supported yet

    GR  ret = _hash_dup(r);
    U32 n   = _size(r+1);
    GR *p   = _data(r+1);
    for (int i=0; i < n; i++, p+=2) {
        _set(&ret, p, p+1);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) merge!
 */
__CFUNC__
hsh_merge_self(GR r[], S32 ri)
{
	ASSERT((r+1)->gt==GT_HASH);	// other types not supported yet

	GR *p  = _data(r+1);
    U32 n  = _size(r+1);
    for (int i=0; i<n; i++, p+=2) {
        _set(r, p, p+1);
    }
}

//================================================================
/*! (method) values
 */
__CFUNC__
hsh_values(GR r[], S32 ri)
{
    GR *p  = _data(r);
    int n  = _size(r);
    GR ret = guru_array_new(n);

    for (int i=0; i<n; i++, p+=2) {
        guru_array_push(&ret, p+1);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! initialize
 */
__GURU__ __const__ Vfunc hsh_vtbl[] = {
	{ "new",	    hsh_new		},
	{ "[]",			hsh_get		},
	{ "[]=",	    hsh_set		},
	{ "clear",		hsh_clr		},
	{ "dup",	    hsh_dup 	},
	{ "delete",	    hsh_del 	},
	{ "empty?",	    hsh_empty	},
	{ "has_key?",	hsh_has_key },
	{ "has_value?",	hsh_has_value	},
	{ "key",	    hsh_key		},
	{ "keys",	    hsh_keys	},
	{ "size",	    hsh_size	},
	{ "length",	    hsh_size	},
	{ "count",	    hsh_size	},
	{ "merge",	    hsh_merge	},
	{ "merge!",	    hsh_merge_self	},
	{ "values",	    hsh_values 	},

	{ "inspect",	gr_to_s 	},
	{ "to_s",	    gr_to_s		}
};
__GURU__ void
guru_init_class_hash()
{
    guru_rom_add_class(GT_HASH, "Hash", GT_OBJ, hsh_vtbl, VFSZ(hsh_vtbl));
    guru_register_func(GT_HASH, NULL, guru_hash_del, guru_hash_cmp);
}
