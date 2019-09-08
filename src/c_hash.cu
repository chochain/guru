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
_size(const GV *kv) {
    return kv->hash->n >> 1;
}

__GURU__ __INLINE__ GV*
_data(const GV *kv) {
	return kv->hash->data;
}

__GURU__ __INLINE__ void
_resize(GV *kv, int size)
{
	guru_array_resize(kv->array, size<<1);
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
__GURU__ void
_set(GV *kv, GV *key, GV *val)
{
    GV *v = _search(kv, key);
    if (v==NULL) {					// key not found, create new kv pair
        guru_array_push(kv, key);	// push into array tail (ref counter += 1)
        guru_array_push(kv, val);
    }
    else {
    	ref_dec(v);					// release previous kv elements
    	ref_dec(v+1);
        *(v)   = *key;
        *(v+1) = *val;
    }
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
_clr(GV *kv)
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
__GURU__ GV
guru_hash_new(int size)
{
    GV ret = {.gt = GT_HASH};
    /*
      Allocate handle and data buffer.
    */
    guru_hash *h    = (guru_hash *)guru_alloc(sizeof(guru_hash));
    GV        *data = (GV *)guru_alloc(sizeof(GV) * (size<<1));

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
	int n   = _size(kv);
    GV  ret = guru_hash_new(n);

    GV  *d  = _data(&ret);
    GV  *s  = _data(kv);
    U32 n2  = ret.hash->n = n<<1;		// n pairs (k,v)
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
guru_hash_del(GV *kv)
{
    guru_array_del(kv);		// free content
}


//================================================================
/*! compare

  @param  v1	Pointer to GV
  @param  v2	Pointer to another GV
  @retval 0	v1==v2
  @retval 1	v1 != v2
*/
__GURU__ int
guru_hash_cmp(const GV *v0, const GV *v1)
{
	int n0 = _size(v0);
    if (n0 != _size(v1)) 			return 1;	// size different

    GV *p0 = _data(v0);
    for (U32 i=0; i < n0; i++, p0+=2) {			// walk the hash element by element
        GV *p1 = _search(v1, p0);				// check key
        if (p1==NULL) 				return 1;	// no key found
        if (guru_cmp(p0+1, p1+1)) 	return 1;	// compare data
    }
    return 0;									// matched
}

//================================================================
/*! (method) new
 */
__GURU__ void
hsh_new(GV v[], U32 argc)
{
	RETURN_VAL(guru_hash_new(0));
}

//================================================================
/*! (operator) []
 */
__GURU__ void
hsh_get(GV v[], U32 argc)
{
    if (argc != 1) {
    	assert(argc!=1);
        return;	// raise ArgumentError.
    }
    GV ret = _get(v, v+1);

    RETURN_VAL(ret);
}

//================================================================
/*! (operator) []=
 */
__GURU__ void
hsh_set(GV v[], U32 argc)
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
hsh_clr(GV v[], U32 argc)
{
    _clr(v);
}

//================================================================
/*! (method) dup
 */
__GURU__ void
hsh_dup(GV v[], U32 argc)
{
    RETURN_VAL(_hash_dup(v));
}

//================================================================
/*! (method) delete
 */
__GURU__ void
hsh_del(GV v[], U32 argc)
{
    // TODO : now, support only delete(key) -> object
    // TODO: re-index hash table if need.
	RETURN_VAL(_remove(v, v+1));
}

//================================================================
/*! (method) empty?
 */
__GURU__ void
hsh_empty(GV v[], U32 argc)
{
    RETURN_BOOL(_size(v)==0);
}

//================================================================
/*! (method) has_key?
 */
__GURU__ void
hsh_has_key(GV v[], U32 argc)
{
    RETURN_BOOL(_search(v, v+1)!=NULL);
}

//================================================================
/*! (method) has_value?
 */
__GURU__ void
hsh_has_value(GV v[], U32 argc)
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
hsh_key(GV v[], U32 argc)
{
    GV *p = _data(v);
    int         n = _size(v);
    for (U32 i=0; i<n; i++, p+=2) {
        if (guru_cmp(p+1, v+1)==0) {
            RETURN_VAL(*p);
        }
    }
    RETURN_NIL();
}

//================================================================
/*! (method) keys
 */
__GURU__ void
hsh_keys(GV v[], U32 argc)
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
hsh_size(GV v[], U32 argc)
{
    RETURN_INT(_size(v));
}

//================================================================
/*! (method) merge
 */
__GURU__ void
hsh_merge(GV v[], U32 argc)		// non-destructive merge
{
    GV ret = _hash_dup(v);
    GV *p  = _data(v+1);
    int         n  = _size(v+1);
    for (U32 i=0; i<n; i++, p+=2) {
        _set(&ret, p, p+1);
    }
    RETURN_VAL(ret);
}

//================================================================
/*! (method) merge!
 */
__GURU__ void
hsh_merge_self(GV v[], U32 argc)
{
    GV *p  = _data(v+1);
    int         n  = _size(v+1);
    for (U32 i=0; i<n; i++, p+=2) {
        _set(v, p, p+1);
    }
}

//================================================================
/*! (method) values
 */
__GURU__ void
hsh_values(GV v[], U32 argc)
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
hsh_inspect(GV v[], U32 argc)
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
    guru_class *c = guru_class_hash = NEW_CLASS("Hash", guru_class_object);

    NEW_PROC("new",	    	hsh_new);
    NEW_PROC("[]",			hsh_get);
    NEW_PROC("[]=",	    	hsh_set);
    NEW_PROC("clear",		hsh_clr);
    NEW_PROC("dup",	    	hsh_dup);
    NEW_PROC("delete",	    hsh_del);
    NEW_PROC("empty?",	    hsh_empty);
    NEW_PROC("has_key?",	hsh_has_key);
    NEW_PROC("has_value?",	hsh_has_value);
    NEW_PROC("key",	    	hsh_key);
    NEW_PROC("keys",	    hsh_keys);
    NEW_PROC("size",	    hsh_size);
    NEW_PROC("length",	    hsh_size);
    NEW_PROC("count",	    hsh_size);
    NEW_PROC("merge",	    hsh_merge);
    NEW_PROC("merge!",	    hsh_merge_self);
    NEW_PROC("values",	    hsh_values);
#if GURU_USE_STRING
    NEW_PROC("inspect",		hsh_inspect);
    NEW_PROC("to_s",	    hsh_inspect);
#endif
}
