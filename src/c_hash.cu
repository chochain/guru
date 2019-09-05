/*! @file
  @brief
  GURU Hash class

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>

#include "vm_config.h"
#include "guru.h"
#include "alloc.h"
#include "static.h"

#include "vm.h"
#include "object.h"

#include "c_hash.h"
#include "c_array.h"
#include "c_string.h"

#include "puts.h"

/*
  function summary

 (constructor)
    guru_hash_new

 (destructor)
    guru_hash_delete

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
_size(const GV *kv) {
    return kv->hash->n >> 1;
}

__GURU__ __INLINE__ GV*
_data(const GV *kv) {
	return kv->hash->data;
}

__GURU__ __INLINE__ int
_resize(GV *kv, int size)
{
	return guru_array_resize(kv->array, size<<1);
}

//================================================================
/*! search key

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	pointer to found key or NULL(not found).
*/
__GURU__ GV*
_search(const GV v[], const GV *key)
{
#ifndef GURU_HASH_SEARCH_LINER
#define GURU_HASH_SEARCH_LINER
#endif
    GV *p = v->hash->data;
    int         n = _size(v);

#ifdef GURU_HASH_SEARCH_LINER
    for (U32 i=0; i < n; i++, p+=2) {
        if (guru_cmp(p, key)==0) return p;
    }
    return NULL;
#endif

#ifdef GURU_HASH_SEARCH_LINER_ITERATOR
    for (U32 i=0; i < n; i++, p+=2) {
        if (guru_cmp(p, key)==0) return p;
    }
    return NULL;
#endif
}

//================================================================
/*! setter

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @param  val	pointer to value
  @return		error_code
*/
__GURU__ int
_set(GV *kv, GV *key, GV *val)
{
    GV *v = _search(kv, key);
    int ret = 0;
    if (v==NULL) {				// key not found, create new kv pair
        if ((ret = guru_array_push(kv, key)) != 0) return ret;
        ret = guru_array_push(kv, val);
    }
    else {
    	ref_clr(v);       // CC: was dec_refc 20181101
    	ref_clr(v+1);     // CC: was dec_refc 20181101
        *(v)   = *key;
        *(v+1) = *val;
    }
    return ret;
}

//================================================================
/*! getter

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	GV data at key position or Nil.
*/
__GURU__ GV
_get(GV *kv, GV *key)
{
    GV *v = _search(kv, key);

    return v ? *(v+1) : GURU_NIL_NEW();
}

//================================================================
/*! remove a data

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	removed data or Nil
*/
__GURU__ GV
_remove(GV *kv, GV *key)
{
    GV *v = _search(kv, key);
    if (v==NULL) return GURU_NIL_NEW();

    ref_clr(v);						// CC: was dec_refc 20181101
    GV ret = *(v+1);				// value
    guru_hash  *h  = kv->hash;
    h->n -= 2;

    MEMCPY(v, (v+2), U8POFF(h->data + h->n, v));

    // TODO: re-index hash table if need.

    return ret;
}

//================================================================
/*! clear all

  @param  hash	pointer to target hash
*/
__GURU__ void
_clear(GV *kv)
{
    guru_array_clear(kv);

    // TODO: re-index hash table if need.
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size
  @return 	hash object
*/
__GURU__ GV
guru_hash_new(int size)
{
    GV ret = {.gt = GT_HASH};
    /*
      Allocate handle and data buffer.
    */
    guru_hash *h = (guru_hash *)guru_alloc(sizeof(guru_hash));
    if (!h) return ret;	// ENOMEM

    GV *data = (GV *)guru_alloc(sizeof(GV) * (size<<1));
    if (!data) {			// ENOMEM
        guru_free(h);
        return ret;
    }
    h->gt  	= GT_HASH;
    h->rc   = 1;
    h->size	= size<<1;
    h->n  	= 0;
    h->data = data;

    ret.hash = h;

    return ret;
}

//================================================================
/*! duplicate

  @param  vm	pointer to VM.
  @param  src	pointer to target hash.
*/
__GURU__ GV
_hash_dup(const GV *kv)
{
	int        n   = _size(kv);
    GV ret = guru_hash_new(n);
    if (ret.hash==NULL) return ret;			// ENOMEM

    GV *d = _data(&ret);
    GV *s = _data(kv);
    int        n2 = ret.hash->n = n<<1;		// n pairs (k,v)
    for (U32 i=0; i < n2; i++) {
    	ref_inc(s);						// one extra ref
    	*d++ = *s++;
    }
    return ret;
}

//================================================================
/*! destructor

  @param  hash	pointer to target value
*/
__GURU__ void
guru_hash_delete(GV *kv)
{
    guru_array_delete(kv);		// free content
}


//================================================================
/*! compare

  @param  v1	Pointer to GV
  @param  v2	Pointer to another GV
  @retval 0	v1==v2
  @retval 1	v1 != v2
*/
__GURU__ int
guru_hash_compare(const GV *v0, const GV *v1)
{
	int n0 = _size(v0);
    if (n0 != _size(v1)) return 1;

    GV *p0 = _data(v0);
    for (U32 i=0; i < n0; i++, p0+=2) {
        GV *p1 = _search(v1, p0);		// check key
        if (p1==NULL) return 1;
        if (guru_cmp(p0+1, p1+1)) return 1;	// check data
    }
    return 0;
}

//================================================================
/*! (method) new
 */
__GURU__ void
c_hash_new(GV v[], U32 argc)
{
	RETURN_VAL(guru_hash_new(0));
}

//================================================================
/*! (operator) []
 */
__GURU__ void
c_hash_get(GV v[], U32 argc)
{
    if (argc != 1) {
    	assert(argc!=1);
        return;	// raise ArgumentError.
    }
    GV ret = _get(v, v+1);
    ref_inc(&ret);

    RETURN_VAL(ret);
}

//================================================================
/*! (operator) []=
 */
__GURU__ void
c_hash_set(GV v[], U32 argc)
{
    if (argc != 2) {
    	assert(argc!=2);
        return;				// raise ArgumentError.
    }
    _set(v, v+1, v+2);		// k + v

    (v+1)->gt = GT_EMPTY;
    (v+2)->gt = GT_EMPTY;
}


//================================================================
/*! (method) clear
 */
__GURU__ void
c_hash_clear(GV v[], U32 argc)
{
    _clear(v);
}

//================================================================
/*! (method) dup
 */
__GURU__ void
c_hash_dup(GV v[], U32 argc)
{
    RETURN_VAL(_hash_dup(v));
}

//================================================================
/*! (method) delete
 */
__GURU__ void
c_hash_delete(GV v[], U32 argc)
{
    // TODO : now, support only delete(key) -> object
    // TODO: re-index hash table if need.
	RETURN_VAL(_remove(v, v+1));
}

//================================================================
/*! (method) empty?
 */
__GURU__ void
c_hash_empty(GV v[], U32 argc)
{
    RETURN_BOOL(_size(v)==0);
}

//================================================================
/*! (method) has_key?
 */
__GURU__ void
c_hash_has_key(GV v[], U32 argc)
{
    RETURN_BOOL(_search(v, v+1)!=NULL);
}

//================================================================
/*! (method) has_value?
 */
__GURU__ void
c_hash_has_value(GV v[], U32 argc)
{
    GV *p = _data(v);
    int         n = _size(v);
    for (U32 i=0; i<n; i++, p+=2) {
        if (guru_cmp(p+1, v+1)==0) {	// value to value
            RETURN_BOOL(1);
        }
    }
    RETURN_BOOL(0);
}

//================================================================
/*! (method) key
 */
__GURU__ void
c_hash_key(GV v[], U32 argc)
{
    GV *p = _data(v);
    int         n = _size(v);
    for (U32 i=0; i<n; i++, p+=2) {
        if (guru_cmp(p+1, v+1)==0) {
            ref_inc(p);
            RETURN_VAL(*p);
            return;
        }
    }
    RETURN_NIL();
}

//================================================================
/*! (method) keys
 */
__GURU__ void
c_hash_keys(GV v[], U32 argc)
{
    GV *p  = _data(v);
    int         n  = _size(v);
    GV ret = guru_array_new(n);

    for (U32 i=0; i<n; i++, p+=2) {
        guru_array_push(&ret, p);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) size,length,count
 */
__GURU__ void
c_hash_size(GV v[], U32 argc)
{
    RETURN_INT(_size(v));
}

//================================================================
/*! (method) merge
 */
__GURU__ void
c_hash_merge(GV v[], U32 argc)		// non-destructive merge
{
    GV ret = _hash_dup(v);
    GV *p  = _data(v+1);
    int         n  = _size(v+1);
    for (U32 i=0; i<n; i++, p+=2) {
        _set(&ret, p, p+1);
        ref_inc(p);						// extra ref on incoming kv
        ref_inc(p+1);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) merge!
 */
__GURU__ void
c_hash_merge_self(GV v[], U32 argc)
{
    GV *p  = _data(v+1);
    int         n  = _size(v+1);
    for (U32 i=0; i<n; i++, p+=2) {
        _set(v, p, p+1);
        ref_inc(p);						// extra ref on incoming kv
        ref_inc(p+1);
    }
}

//================================================================
/*! (method) values
 */
__GURU__ void
c_hash_values(GV v[], U32 argc)
{
    GV *p  = _data(v);
    int         n  = _size(v);
    GV ret = guru_array_new(n);

    for (U32 i=0; i<n; i++, p+=2) {
        guru_array_push(&ret, p+1);
    }
    RETURN_VAL(ret);
}

#if GURU_USE_STRING
//================================================================
__GURU__ void
c_hash_inspect(GV v[], U32 argc)
{
    GV blank = guru_str_new("");
    GV comma = guru_str_new(", ");
    GV ret   = guru_str_new("{");
    if (!ret.str) {
    	RETURN_NIL();
    }

    GV s[3];
    GV *p = _data(v);
    int         n = _size(v);
    for (U32 i=0; i<n; i++, p+=2) {
    	s[0] = (i==0) ? blank : comma;
        s[1] = guru_inspect(v+argc, p);			// key
        s[2] = guru_inspect(v+argc, p+1);		// value

        guru_str_append(&ret, &s[0]);
        guru_str_append(&ret, &s[1]);
        guru_str_append_cstr(&ret, "=>");
        guru_str_append(&ret, &s[2]);

        ref_clr(&s[1]);							// free locally allocated memory
        ref_clr(&s[2]);
    }
    guru_str_append_cstr(&ret, "}");

    RETURN_VAL(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void
guru_init_class_hash()
{
    guru_class *c = guru_class_hash = guru_add_class("Hash", guru_class_object);

    guru_add_proc(c, "new",	    	c_hash_new);
    guru_add_proc(c, "[]",			c_hash_get);
    guru_add_proc(c, "[]=",	    	c_hash_set);
    guru_add_proc(c, "clear",		c_hash_clear);
    guru_add_proc(c, "dup",	    	c_hash_dup);
    guru_add_proc(c, "delete",	    c_hash_delete);
    guru_add_proc(c, "empty?",	    c_hash_empty);
    guru_add_proc(c, "has_key?",	c_hash_has_key);
    guru_add_proc(c, "has_value?",	c_hash_has_value);
    guru_add_proc(c, "key",	    	c_hash_key);
    guru_add_proc(c, "keys",	    c_hash_keys);
    guru_add_proc(c, "size",	    c_hash_size);
    guru_add_proc(c, "length",	    c_hash_size);
    guru_add_proc(c, "count",	    c_hash_size);
    guru_add_proc(c, "merge",	    c_hash_merge);
    guru_add_proc(c, "merge!",	    c_hash_merge_self);
    guru_add_proc(c, "values",	    c_hash_values);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect",		c_hash_inspect);
    guru_add_proc(c, "to_s",	    c_hash_inspect);
#endif
}
