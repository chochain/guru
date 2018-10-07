/*! @file
  @brief
  mrubyc memory management.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  Memory management for objects in Guru.

  </pre>
*/
#include "alloc.h"
#include "value.h"
#include "keyvalue.h"
#include "symbol.h"
#include "vmalloc.h"

#include <assert.h>

// memory pool
extern __GURU__ uint8_t *memory_pool;

// << from value.cu
__GURU__
mrbc_object *mrbc_obj_alloc(mrbc_vtype tt)
{
    mrbc_object *ptr = (mrbc_object *)mrbc_alloc(sizeof(mrbc_object));
    if (ptr){
        ptr->tt = tt;
    }
    return ptr;
}

__GURU__
mrbc_proc *mrbc_rproc_alloc(const char *name)
{
    mrbc_proc *ptr = (mrbc_proc *)mrbc_alloc(sizeof(mrbc_proc));
    if (ptr) {
        ptr->ref_count = 1;
        ptr->sym_id = str_to_symid(name);
#ifdef MRBC_DEBUG
        ptr->names = name;	// for debug; delete soon.
#endif
        ptr->next = 0;
    }
    return ptr;
}

//================================================================
/*! mrbc_instance constructor

  @param  vm    Pointer to VM.
  @param  cls	Pointer to Class (mrbc_class).
  @param  size	size of additional data.
  @return       mrbc_instance object.
*/
__GURU__
mrbc_value mrbc_instance_new(mrbc_class *cls, int size)
{
    mrbc_value v = {.tt = MRBC_TT_OBJECT};
    v.instance = (mrbc_instance *)mrbc_alloc(sizeof(mrbc_instance) + size);
    if (v.instance == NULL) return v;	// ENOMEM

    v.instance->ivar = mrbc_kv_new(0);
    if (v.instance->ivar == NULL) {	// ENOMEM
        mrbc_raw_free(v.instance);
        v.instance = NULL;
        return v;
    }

    v.instance->ref_count = 1;
    v.instance->tt = MRBC_TT_OBJECT;	// for debug only.
    v.instance->cls = cls;

    return v;
}

//================================================================
/*! mrbc_instance destructor

  @param  v	pointer to target value
*/
__GURU__
void mrbc_instance_delete(mrbc_value *v)
{
    mrbc_kv_delete(v->instance->ivar);
    mrbc_raw_free(v->instance);
}


//================================================================
/*! instance variable setter

  @param  obj		pointer to target.
  @param  sym_id	key symbol ID.
  @param  v		pointer to value.
*/
__GURU__
void mrbc_instance_setiv(mrbc_object *obj, mrbc_sym sym_id, mrbc_value *v)
{
    mrbc_dup(v);
    mrbc_kv_set(obj->instance->ivar, sym_id, v);
}


//================================================================
/*! instance variable getter

  @param  obj		pointer to target.
  @param  sym_id	key symbol ID.
  @return		value.
*/
__GURU__
mrbc_value mrbc_instance_getiv(mrbc_object *obj, mrbc_sym sym_id)
{
    mrbc_value *v = mrbc_kv_get(obj->instance->ivar, sym_id);
    if (!v) return mrbc_nil_value();

    mrbc_dup(v);
    return *v;
}

//================================================================
/*!@brief
  Decrement reference counter

  @param   v     Pointer to target mrbc_value
*/
__GURU__
void mrbc_dec_ref_counter(mrbc_value *v)
{
    switch(v->tt){
    case MRBC_TT_OBJECT:
    case MRBC_TT_PROC:
    case MRBC_TT_ARRAY:
    case MRBC_TT_STRING:
    case MRBC_TT_RANGE:
    case MRBC_TT_HASH:
        assert(v->instance->ref_count != 0);
        v->instance->ref_count--;
        break;

    default:
        // Nothing
        return;
    }

    // release memory?
    if (v->instance->ref_count != 0) return;

    switch(v->tt) {
    case MRBC_TT_OBJECT:	mrbc_instance_delete(v);	break;
    case MRBC_TT_PROC:	    mrbc_raw_free(v->handle);	break;
#if MRBC_USE_STRING
    case MRBC_TT_STRING:	mrbc_string_delete(v);		break;
#endif
#if MRBC_USE_ARRAY
    case MRBC_TT_ARRAY:	    mrbc_array_delete(v);		break;
    case MRBC_TT_RANGE:	    mrbc_range_delete(v);		break;
    case MRBC_TT_HASH:	    mrbc_hash_delete(v);		break;
#endif
    default:
        // Nothing
        break;
    }
}

//================================================================
/*!@brief
  Release object related memory

  @param   v     Pointer to target mrbc_value
*/
__GURU__
void mrbc_release(mrbc_value *v)
{
    mrbc_dec_ref_counter(v);
    v->tt = MRBC_TT_EMPTY;
}










