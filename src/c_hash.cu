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
#include "c_array.h"
#include "c_hash.h"
#include "c_string.h"

//================================================================
/*!@brief
  Define Hash iterator.
*/
extern "C" typedef struct RHashIterator {
    mrbc_hash  *h;
    mrbc_value *p;
    mrbc_value *p_end;
} _iterator;

//================================================================
/*! iterator constructor
 */
__GURU__
_iterator _iterator_new(mrbc_value *v)
{
    _iterator itr;

    itr.h     = v->hash;
    itr.p     = v->hash->data;
    itr.p_end = itr.p + v->hash->n;

    return itr;
}

//================================================================
/*! iterator getter
 */
__GURU__
mrbc_value *_next(_iterator *itr)
{
	if (itr->p >= itr->p_end) return NULL;

    mrbc_value *ret = itr->p;
    itr->p += 2;					// k + v

    return ret;
}

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
__GURU__
int mrbc_hash_size(const mrbc_value *kv) {
    return kv->hash->n / 2;
}

//================================================================
/*! resize buffer
 */
__GURU__
int mrbc_hash_resize(mrbc_value *kv, int size)
{
    return mrbc_array_resize(kv->array, size * 2);
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size
  @return 	hash object
*/
__GURU__
mrbc_value mrbc_hash_new(int size)
{
    mrbc_value ret = {.tt = MRBC_TT_HASH};
    /*
      Allocate handle and data buffer.
    */
    mrbc_hash *h = (mrbc_hash *)mrbc_alloc(sizeof(mrbc_hash));
    if (!h) return ret;	// ENOMEM

    mrbc_value *data = (mrbc_value *)mrbc_alloc(sizeof(mrbc_value) * size * 2);
    if (!data) {			// ENOMEM
        mrbc_free(h);
        return ret;
    }
    h->refc = 1;
    h->tt  	= MRBC_TT_HASH;
    h->size	= size * 2;
    h->n  	= 0;
    h->data = data;

    ret.hash = h;

    return ret;
}


//================================================================
/*! destructor

  @param  hash	pointer to target value
*/
__GURU__
void mrbc_hash_delete(mrbc_value *kv)
{
    // TODO: delete other members (for search).

    mrbc_array_delete(kv);
}


//================================================================
/*! search key

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	pointer to found key or NULL(not found).
*/
__GURU__
mrbc_value *_hash_search(const mrbc_value v[], const mrbc_value *key)
{
#ifndef MRBC_HASH_SEARCH_LINER
#define MRBC_HASH_SEARCH_LINER
#endif

#ifdef MRBC_HASH_SEARCH_LINER
    mrbc_value *p = v->hash->data;
    for (int i=0; i<v->hash->n; i+=2, p+=2) {
        if (mrbc_compare(p, key)==0) return p;
    }
    return NULL;
#endif

#ifdef MRBC_HASH_SEARCH_LINER_ITERATOR
    _iterator itr = _iterator_new(hash);
    mrbc_value *p;
    while ((p=_next(&itr))!=NULL) {
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
__GURU__
int _hash_set(mrbc_value *kv, mrbc_value *key, mrbc_value *val)
{
    mrbc_value *v = _hash_search(kv, key);
    int ret = 0;
    if (v==NULL) {
        // set a new value
        if ((ret = mrbc_array_push(kv, key)) != 0) return ret;
        ret = mrbc_array_push(kv, val);
    }
    else {
        *v     = *key;
        *(v+1) = *val;
    }
    return ret;
}

//================================================================
/*! getter

  @param  hash	pointer to target hash
  @param  key	pointer to key value
  @return	mrbc_value data at key position or Nil.
*/
__GURU__
mrbc_value _hash_get(mrbc_value *kv, mrbc_value *key)
{
    mrbc_value *v = _hash_search(kv, key);

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
__GURU__
mrbc_value _hash_remove(mrbc_value *kv, mrbc_value *key)
{
    mrbc_value *v = _hash_search(kv, key);
    if (v==NULL) return mrbc_nil_value();

    mrbc_release(v);			// CC: was dec_refc 20181101
    mrbc_value ret = v[1];		// value
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
__GURU__
void _hash_clear(mrbc_value *kv)
{
    mrbc_array_clear(kv);

    // TODO: re-index hash table if need.
}

//================================================================
/*! compare

  @param  v1	Pointer to mrbc_value
  @param  v2	Pointer to another mrbc_value
  @retval 0	v1==v2
  @retval 1	v1 != v2
*/
__GURU__
int mrbc_hash_compare(const mrbc_value *v0, const mrbc_value *v1)
{
    if (v0->hash->n != v1->hash->n) return 1;

    mrbc_value *p0 = v0->hash->data;
    for (int i = 0; i < mrbc_hash_size(v0); i++, p0++) {
        mrbc_value *p1 = _hash_search(v1, p0);	// check key
        if (p1==NULL) return 1;
        if (mrbc_compare(++p0, ++p1)) return 1;	// check data
    }
    return 0;
}

//================================================================
/*! duplicate

  @param  vm	pointer to VM.
  @param  src	pointer to target hash.
*/
__GURU__
mrbc_value _hash_dup(mrbc_value *kv)
{
    mrbc_value ret = mrbc_hash_new(mrbc_hash_size(kv));
    if (ret.hash==NULL) return ret;		// ENOMEM

    mrbc_hash *h = kv->hash;
    MEMCPY((uint8_t *)ret.hash->data, (uint8_t *)h->data, sizeof(mrbc_value) * h->n);
    ret.hash->n = h->n;

    mrbc_value *p = h->data;
    for (int i=0; i<h->n; i++, p++) {
        mrbc_retain(p++);					// dup, add one extra reference
    }
    // TODO: dup other members.

    return ret;
}

//================================================================
/*! (method) new
 */
__GURU__
void c_hash_new(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_hash_new(0);
    SET_RETURN(ret);
}

//================================================================
/*! (operator) []
 */
__GURU__
void c_hash_get(mrbc_value v[], int argc)
{
    if (argc != 1) {
    	assert(argc!=1);
        return;	// raise ArgumentError.
    }
    mrbc_value kv = _hash_get(v, v+1);

    SET_RETURN(kv);
}

//================================================================
/*! (operator) []=
 */
__GURU__
void c_hash_set(mrbc_value v[], int argc)
{
    if (argc != 2) {
    	assert(argc!=2);
        return;				// raise ArgumentError.
    }
    _hash_set(v, v+1, v+2);	// k + v

    mrbc_release(v+1);
    mrbc_release(v+2);
}


//================================================================
/*! (method) clear
 */
__GURU__
void c_hash_clear(mrbc_value v[], int argc)
{
    _hash_clear(v);
}

//================================================================
/*! (method) dup
 */
__GURU__
void c_hash_dup(mrbc_value v[], int argc)
{
    mrbc_value ret = _hash_dup(v);

    SET_RETURN(ret);
}

//================================================================
/*! (method) delete
 */
__GURU__
void c_hash_delete(mrbc_value v[], int argc)
{
    // TODO : now, support only delete(key) -> object

    mrbc_value ret = _hash_remove(v, v+1);

    // TODO: re-index hash table if need.

    SET_RETURN(ret);
}

//================================================================
/*! (method) empty?
 */
__GURU__
void c_hash_empty(mrbc_value v[], int argc)
{
    int n = mrbc_hash_size(v);

    SET_BOOL_RETURN(!n);
}

//================================================================
/*! (method) has_key?
 */
__GURU__
void c_hash_has_key(mrbc_value v[], int argc)
{
    mrbc_value *res = _hash_search(v, v+1);

    SET_BOOL_RETURN(res);
}

//================================================================
/*! (method) has_value?
 */
__GURU__
void c_hash_has_value(mrbc_value v[], int argc)
{
    _iterator itr = _iterator_new(&v[0]);

    mrbc_value *kv;
    while ((kv=_next(&itr))) {
        if (mrbc_compare(kv+1, v+1)==0) {	// value to value
            SET_BOOL_RETURN(1);
            return;
        }
    }
    SET_BOOL_RETURN(0);
}

//================================================================
/*! (method) key
 */
__GURU__
void c_hash_key(mrbc_value v[], int argc)
{
    mrbc_value *ret = NULL;
    _iterator  itr = _iterator_new(&v[0]);

    mrbc_value *kv;
    while ((kv=_next(&itr))) {
        if (mrbc_compare(kv+1, v+1)==0) {
            mrbc_retain(kv);
            ret = kv;
            break;
        }
    }
    if (ret) SET_RETURN(*ret);
    else 	 SET_NIL_RETURN();
}

//================================================================
/*! (method) keys
 */
__GURU__
void c_hash_keys(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_array_new(mrbc_hash_size(v));
    _iterator  itr = _iterator_new(v);

    mrbc_value *kv;
    while ((kv=_next(&itr))) {
        mrbc_array_push(&ret, kv);
        mrbc_retain(kv);
    }
    SET_RETURN(ret);
}

//================================================================
/*! (method) size,length,count
 */
__GURU__
void c_hash_size(mrbc_value v[], int argc)
{
    int n = mrbc_hash_size(v);

    SET_INT_RETURN(n);
}

//================================================================
/*! (method) merge
 */
__GURU__
void c_hash_merge(mrbc_value v[], int argc)		// non-destructive merge
{
    mrbc_value ret = _hash_dup(&v[0]);
    _iterator  itr = _iterator_new(&v[1]);

    mrbc_value *kv;
    while ((kv=_next(&itr))) {
        // mrbc_retain(&kv[0]);                 // CC: removed 20181029
        // mrbc_retain(&kv[1]);
        _hash_set(&ret, kv, kv+1);
    }
    SET_RETURN(ret);
}

//================================================================
/*! (method) merge!
 */
__GURU__
void c_hash_merge_self(mrbc_value v[], int argc)
{
    _iterator itr = _iterator_new(&v[1]);

    mrbc_value *kv;
    while ((kv=_next(&itr))) {
        // mrbc_retain(&kv[0]);                   // CC: removed 20181029
        // mrbc_retain(&kv[1]); 
        _hash_set(v, kv, kv+1);
    }
}

//================================================================
/*! (method) values
 */
__GURU__
void c_hash_values(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_array_new(mrbc_hash_size(v));
    _iterator itr = _iterator_new(v);


    mrbc_value *kv;
    while ((kv=_next(&itr))) {
        mrbc_array_push(&ret, kv+1);
        mrbc_retain(kv+1);
    }
    SET_RETURN(ret);
}

#if MRBC_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__
void _hrfc(mrbc_value *str, mrbc_value *v)
{
	char buf[8];
	guru_sprintf(buf, "^%d_", v->self->refc);
	mrbc_string_append_cstr(str, buf);
}

__GURU__
void c_hash_inspect(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_new("{");
    _hrfc(&ret, v+argc);
    if (!ret.str) {
    	SET_NIL_RETURN();
    	return;
    }

    _iterator itr = _iterator_new(v);
    int first = 1;

    mrbc_value *kv;
    while ((kv=_next(&itr))) {
        if (!first) mrbc_string_append_cstr(&ret, ", ");
        first = 0;

        mrbc_value s1  = mrbc_send(v+argc, kv, "inspect", 0);
        mrbc_string_append(&ret, &s1);
        mrbc_release(&s1);

        mrbc_string_append_cstr(&ret, "=>");

        s1 = mrbc_send(v+argc, kv+1, "inspect", 0);
        mrbc_string_append(&ret, &s1);
        mrbc_release(&s1);
    }
    mrbc_string_append_cstr(&ret, "}");
    _hrfc(&ret, v+argc);

    SET_RETURN(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__
void mrbc_init_class_hash()
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
