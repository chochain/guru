/*! @file
  @brief
  mruby/c Hash class

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
#include "class.h"

#include "console.h"
#include "sprintf.h"

#include "vm.h"
#include "object.h"
#include "c_hash.h"
#include "c_array.h"
#include "c_string.h"

/*
  function summary

 (constructor)
    mrbc_hash_new

 (destructor)
    mrbc_hash_delete

 (setter)
  --[name]-------------[arg]---[ret]-------
    mrbc_hash_set		*K,*V	int

 (getter)
  --[name]-------------[arg]---[ret]---[note]------------------------
    mrbc_hash_get		*K		T		Data remains in the container
    mrbc_hash_remove	*K		T		Data does not remain in the container
    mrbc_hash_i_next		   *T		Data remains in the container
*/


//================================================================
/*! get size
 */
__GURU__ __INLINE__ int
_size(const mrbc_value *kv) {
    return kv->hash->n >> 1;
}

__GURU__ __INLINE__ mrbc_value*
_data(const mrbc_value *kv) {
	return kv->hash->data;
}

__GURU__ __INLINE__ int
_resize(mrbc_value *kv, int size)
{
	return mrbc_array_resize(kv->array, size<<1);
}

//================================================================
/*! search key

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	pointer to found key or NULL(not found).
*/
__GURU__ mrbc_value*
_search(const mrbc_value v[], const mrbc_value *key)
{
#ifndef MRBC_HASH_SEARCH_LINER
#define MRBC_HASH_SEARCH_LINER
#endif
    mrbc_value *p = v->hash->data;
    int         n = _size(v);

#ifdef MRBC_HASH_SEARCH_LINER
    for (int i=0; i < n; i+=2, p+=2) {
        if (mrbc_compare(p, key)==0) return p;
    }
    return NULL;
#endif

#ifdef MRBC_HASH_SEARCH_LINER_ITERATOR
    for (int i=0; i < n; i+=2, p+=2) {
        if (mrbc_compare(p, key)==0) return p;
    }
    return NULL;
#endif
}

//================================================================
/*! setter

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @param  val	pointer to value
  @return	mrbc_error_code
*/
__GURU__ int
_set(mrbc_value *kv, mrbc_value *key, mrbc_value *val)
{
    mrbc_value *v = _search(kv, key);
    int ret = 0;
    if (v==NULL) {
        // set a new value
        if ((ret = mrbc_array_push(kv, key)) != 0) return ret;
        ret = mrbc_array_push(kv, val);
    }
    else {
    	mrbc_release(v);		// CC: added 20181101
        mrbc_release(v+1);		// CC: added 20181101
        *(v)   = *key;
        *(v+1) = *val;
        mrbc_retain(key);		// CC: added 20181101
        mrbc_retain(val);		// CC: added 20181101
    }
    return ret;
}

//================================================================
/*! getter

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	mrbc_value data at key position or Nil.
*/
__GURU__ mrbc_value
_get(mrbc_value *kv, mrbc_value *key)
{
    mrbc_value *v = _search(kv, key);

    if (v) {
    	mrbc_retain(++v);		// CC: added 20181101, inc_ref the value
    	return *v;
    }
    else return mrbc_nil_value();
}

//================================================================
/*! remove a data

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	removed data or Nil
*/
__GURU__ mrbc_value
_remove(mrbc_value *kv, mrbc_value *key)
{
    mrbc_value *v = _search(kv, key);
    if (v==NULL) return mrbc_nil_value();

    mrbc_release(v);				// CC: was dec_refc 20181101
    mrbc_value ret = v[1];			// value
    mrbc_hash  *h  = kv->hash;
    h->n -= 2;

    MEMCPY((uint8_t *)v, (uint8_t *)(v+2), (uint8_t *)(h->data + h->n) - (uint8_t *)v);

    // TODO: re-index hash table if need.

    return ret;
}

//================================================================
/*! clear all

  @param  hash	pointer to target hash
*/
__GURU__ void
_clear(mrbc_value *kv)
{
    mrbc_array_clear(kv);

    // TODO: re-index hash table if need.
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size
  @return 	hash object
*/
__GURU__ mrbc_value
mrbc_hash_new(int size)
{
    mrbc_value ret = {.tt = MRBC_TT_HASH};
    /*
      Allocate handle and data buffer.
    */
    mrbc_hash *h = (mrbc_hash *)mrbc_alloc(sizeof(mrbc_hash));
    if (!h) return ret;	// ENOMEM

    mrbc_value *data = (mrbc_value *)mrbc_alloc(sizeof(mrbc_value) * (size<<1));
    if (!data) {			// ENOMEM
        mrbc_free(h);
        return ret;
    }
    h->refc = 1;
    h->tt  	= MRBC_TT_HASH;
    h->size	= size<<1;
    h->n  	= 0;
    h->data = data;

    ret.hash = h;

    return ret;
}


//================================================================
/*! destructor

  @param  hash	pointer to target value
*/
__GURU__ void
mrbc_hash_delete(mrbc_value *kv)
{
    mrbc_array_delete(kv);		// free content
    mrbc_release(kv);			// release handle
    mrbc_free(kv);				// free handle
}


//================================================================
/*! compare

  @param  v1	Pointer to mrbc_value
  @param  v2	Pointer to another mrbc_value
  @retval 0	v1==v2
  @retval 1	v1 != v2
*/
__GURU__ int
mrbc_hash_compare(const mrbc_value *v0, const mrbc_value *v1)
{
    if (_size(v0) != _size(v1)) return 1;

    mrbc_value *p0 = _data(v0);
    for (int i = 0; i < _size(v0); i++, p0+=2) {
        mrbc_value *p1 = _search(v1, p0);	// check key
        if (p1==NULL) return 1;
        if (mrbc_compare(p0+1, p1+1)) return 1;	// check data
    }
    return 0;
}

//================================================================
/*! duplicate

  @param  vm	pointer to VM.
  @param  src	pointer to target hash.
*/
__GURU__ mrbc_value
_hash_dup(mrbc_value *kv)
{
    mrbc_value ret = mrbc_hash_new(_size(kv));
    if (ret.hash==NULL) return ret;		// ENOMEM

    int n2 = ret.hash->n = _size(kv) << 1;
    MEMCPY((uint8_t *)_data(&ret), (uint8_t *)_data(kv), sizeof(mrbc_value) * n2);

    mrbc_value *p = _data(&ret);
    for (int i=0; i < n2; i++, p++) {
        mrbc_retain(p++);				// dup, add one extra reference
    }
    // TODO: dup other members.

    return ret;
}

//================================================================
/*! (method) new
 */
__GURU__ void
c_hash_new(mrbc_value v[], int argc)
{
    SET_RETURN(mrbc_hash_new(0));
}

//================================================================
/*! (operator) []
 */
__GURU__ void
c_hash_get(mrbc_value v[], int argc)
{
    if (argc != 1) {
    	assert(argc!=1);
        return;	// raise ArgumentError.
    }
    SET_RETURN(_get(v, v+1));
}

//================================================================
/*! (operator) []=
 */
__GURU__ void
c_hash_set(mrbc_value v[], int argc)
{
    if (argc != 2) {
    	assert(argc!=2);
        return;				// raise ArgumentError.
    }
    _set(v, v+1, v+2);	// k + v
}


//================================================================
/*! (method) clear
 */
__GURU__ void
c_hash_clear(mrbc_value v[], int argc)
{
    _clear(v);
}

//================================================================
/*! (method) dup
 */
__GURU__ void
c_hash_dup(mrbc_value v[], int argc)
{
    mrbc_value ret = _hash_dup(v);

    SET_RETURN(ret);
}

//================================================================
/*! (method) delete
 */
__GURU__ void
c_hash_delete(mrbc_value v[], int argc)
{
    // TODO : now, support only delete(key) -> object

    SET_RETURN(_remove(v, v+1));

    // TODO: re-index hash table if need.
}

//================================================================
/*! (method) empty?
 */
__GURU__ void
c_hash_empty(mrbc_value v[], int argc)
{
    SET_BOOL_RETURN(_size(v)==0);
}

//================================================================
/*! (method) has_key?
 */
__GURU__ void
c_hash_has_key(mrbc_value v[], int argc)
{
    SET_BOOL_RETURN(_search(v, v+1)!=NULL);
}

//================================================================
/*! (method) has_value?
 */
__GURU__ void
c_hash_has_value(mrbc_value v[], int argc)
{
    mrbc_value *p = _data(v);
    int         n = _size(v);
    for (int i=0; i<n; i++, p+=2) {
        if (mrbc_compare(p+1, v+1)==0) {	// value to value
            SET_BOOL_RETURN(1);
            return;
        }
    }
    SET_BOOL_RETURN(0);
}

//================================================================
/*! (method) key
 */
__GURU__ void
c_hash_key(mrbc_value v[], int argc)
{
    mrbc_value *p = _data(v);
    int         n = _size(v);
    for (int i=0; i<n; i++, p+=2) {
        if (mrbc_compare(p+1, v+1)==0) {
            SET_RETURN(*p);
            return;
        }
    }
    SET_NIL_RETURN();
}

//================================================================
/*! (method) keys
 */
__GURU__ void
c_hash_keys(mrbc_value v[], int argc)
{
    mrbc_value *p  = _data(v);
    int         n  = _size(v);
    mrbc_value ret = mrbc_array_new(n);

    for (int i=0; i<n; i++, p+=2) {
        mrbc_array_push(&ret, p);
    }
    SET_RETURN(ret);
}

//================================================================
/*! (method) size,length,count
 */
__GURU__ void
c_hash_size(mrbc_value v[], int argc)
{
    SET_INT_RETURN(_size(v));
}

//================================================================
/*! (method) merge
 */
__GURU__ void
c_hash_merge(mrbc_value v[], int argc)		// non-destructive merge
{
    mrbc_value ret = _hash_dup(v);
    mrbc_value *p  = _data(v+1);
    int         n  = _size(v+1);
    for (int i=0; i<n; i++, p+=2) {
        _set(&ret, p, p+1);
        mrbc_retain(p);
        mrbc_retain(p+1);
    }
    SET_RETURN(ret);
}

//================================================================
/*! (method) merge!
 */
__GURU__ void
c_hash_merge_self(mrbc_value v[], int argc)
{
    mrbc_value *p  = _data(v+1);
    int         n  = _size(v+1);
    for (int i=0; i<n; i++, p+=2) {
        _set(v, p, p+1);
    }
}

//================================================================
/*! (method) values
 */
__GURU__ void
c_hash_values(mrbc_value v[], int argc)
{
    mrbc_value *p  = _data(v);
    int         n  = _size(v);
    mrbc_value ret = mrbc_array_new(n);

    for (int i=0; i<n; i++, p+=2) {
        mrbc_array_push(&ret, p+1);
    }
    SET_RETURN(ret);
}

#if MRBC_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__ void
_hrfc(mrbc_value *str, mrbc_value *v)
{
	char buf[8];
	guru_sprintf(buf, "^%d_", v->self->refc);
	mrbc_string_append_cstr(str, buf);
}

__GURU__ void
c_hash_inspect(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_new("{");
    if (!ret.str) {
    	SET_NIL_RETURN();
    	return;
    }

    int rc;
    mrbc_value s1;
    mrbc_value *p = _data(v);
    int         n = _size(v);
    for (int i=0; i<n; i++, p+=2) {
        if (i!=0) mrbc_string_append_cstr(&ret, ", ");

        s1 = mrbc_send(v+argc, p, "inspect", 0);
        rc = s1.self->refc;
        mrbc_string_append(&ret, &s1);
        mrbc_string_delete(&s1);						// free locally allocated memory

        mrbc_string_append_cstr(&ret, "=>");

        s1 = mrbc_send(v+argc, p+1, "inspect", 0);
        rc = s1.self->refc;
        mrbc_string_append(&ret, &s1);
        mrbc_string_delete(&s1);
    }
    mrbc_string_append_cstr(&ret, "}");

    SET_RETURN(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void
mrbc_init_class_hash()
{
    mrbc_class *c = mrbc_class_hash = mrbc_define_class("Hash", mrbc_class_object);

    mrbc_define_method(c, "new",	    c_hash_new);
    mrbc_define_method(c, "[]",		    c_hash_get);
    mrbc_define_method(c, "[]=",	    c_hash_set);
    mrbc_define_method(c, "clear",	    c_hash_clear);
    mrbc_define_method(c, "dup",	    c_hash_dup);
    mrbc_define_method(c, "delete",	    c_hash_delete);
    mrbc_define_method(c, "empty?",	    c_hash_empty);
    mrbc_define_method(c, "has_key?",	c_hash_has_key);
    mrbc_define_method(c, "has_value?",	c_hash_has_value);
    mrbc_define_method(c, "key",	    c_hash_key);
    mrbc_define_method(c, "keys",	    c_hash_keys);
    mrbc_define_method(c, "size",	    c_hash_size);
    mrbc_define_method(c, "length",	    c_hash_size);
    mrbc_define_method(c, "count",	    c_hash_size);
    mrbc_define_method(c, "merge",	    c_hash_merge);
    mrbc_define_method(c, "merge!",	    c_hash_merge_self);
    mrbc_define_method(c, "to_h",	    c_nop);
    mrbc_define_method(c, "values",	    c_hash_values);
#if MRBC_USE_STRING
    mrbc_define_method(c, "inspect",	c_hash_inspect);
    mrbc_define_method(c, "to_s",	    c_hash_inspect);
#endif
}
