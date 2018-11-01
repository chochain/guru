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
#include "instance.h"

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
    v.self = (mrbc_instance *)mrbc_alloc(sizeof(mrbc_instance) + size);
    if (v.self == NULL) return v;	// ENOMEM

    v.self->ivar = mrbc_kv_new(0);
    if (v.self->ivar == NULL) {		// ENOMEM
        mrbc_free(v.self);
        v.self = NULL;
        return v;
    }

    v.self->refc = 1;
    v.self->tt   = MRBC_TT_OBJECT;	// for debug only.
    v.self->cls  = cls;

    return v;
}

//================================================================
/*! mrbc_instance destructor

  @param  v	pointer to target value
*/
__GURU__
void mrbc_instance_delete(mrbc_value *v)
{
    mrbc_kv_delete(v->self->ivar);
    mrbc_free(v->self);
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
    mrbc_kv_set(obj->self->ivar, sym_id, v);
    mrbc_retain(v);
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
    mrbc_value *v = mrbc_kv_get(obj->self->ivar, sym_id);
    if (!v) return mrbc_nil_value();

    mrbc_retain(v);
    return *v;
}










