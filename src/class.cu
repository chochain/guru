/*! @file
  @brief
  Guru Object, Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <assert.h>
#include <stdarg.h>

#include "guru.h"
#include "value.h"
#include "alloc.h"
#include "symbol.h"
#include "global.h"
#include "static.h"

#include "class.h"
#include "puts.h"

__GURU__ U32 _mutex_cls;

//================================================================
/* methods to add core class/proc for GURU
 * it uses (const U8 *) for static string
 */
__GURU__ guru_class*
guru_add_class(const char *name, guru_class *super)
{
	return mrbc_define_class((U8P)name, super);
}

__GURU__ guru_proc*
guru_add_proc(guru_class *cls, const char *name, guru_fptr cfunc)
{
	return mrbc_define_method(cls, (U8P)name, cfunc);
}

//================================================================
/*!@brief
  find class by object

  @param  vm
  @param  obj
  @return pointer to guru_class
*/
__GURU__ guru_class*
mrbc_get_class_by_object(guru_obj *obj)
{
    guru_class *cls;

    switch (obj->tt) {
    case GURU_TT_TRUE:	  cls = guru_class_true;		break;
    case GURU_TT_FALSE:	  cls = guru_class_false; 	    break;
    case GURU_TT_NIL:	  cls = guru_class_nil;		    break;
    case GURU_TT_FIXNUM:  cls = guru_class_fixnum;	    break;
#if GURU_USE_FLOAT
    case GURU_TT_FLOAT:	  cls = guru_class_float; 	    break;
#endif
    case GURU_TT_SYMBOL:  cls = guru_class_symbol;	    break;

    case GURU_TT_OBJECT:  cls = obj->self->cls; 		break;
    case GURU_TT_CLASS:   cls = obj->cls;               break;
    case GURU_TT_PROC:	  cls = guru_class_proc;		break;
#if GURU_USE_STRING
    case GURU_TT_STRING:  cls = guru_class_string;	    break;
#endif
#if GURU_USE_ARRAY
    case GURU_TT_ARRAY:   cls = guru_class_array; 	    break;
    case GURU_TT_RANGE:	  cls = guru_class_range; 	    break;
    case GURU_TT_HASH:	  cls = guru_class_hash;		break;
#endif
    default:		      cls = guru_class_object;	    break;
    }
    return cls;
}

//================================================================
/*! get class by name

  @param  name		class name.
  @return		pointer to class object.
*/
__GURU__ guru_class*
mrbc_get_class_by_name(const U8P name)
{
    guru_obj obj = const_object_get(name2symid(name));

    return (obj.tt==GURU_TT_CLASS) ? obj.cls : NULL;
}

//================================================================
/*!@brief
  find method from

  @param  vm
  @param  recv
  @param  sym_id
  @return
*/
__GURU__ guru_proc*
mrbc_get_proc_by_symid(GV rcv, guru_sym sid)
{
    guru_class *cls = mrbc_get_class_by_object(&rcv);

    while (cls != 0) {
        guru_proc *proc = cls->vtbl;
        while (proc != 0) {					// walk the linked list
            if (proc->sym_id == sid) {
                return proc;
            }
            proc = proc->next;
        }
        cls = cls->super;					// search the super class
    }
    return NULL;
}

//================================================================
/*!@brief
  define class

  @param  vm		pointer to vm.
  @param  name		class name.
  @param  super		super class.
*/
__GURU__ guru_class*
mrbc_define_class(const U8P name, guru_class *super)
{
    if (super == NULL) super = guru_class_object;  // set default to Object.

    guru_class *cls = mrbc_get_class_by_name(name);
    if (cls) return cls;

    // create a new class?
    cls = (guru_class *)mrbc_alloc(sizeof(guru_class));
    if (!cls) return NULL;			// ENOMEM

    guru_sym sid = name2symid(name);

    cls->sym_id = sid;
    cls->super 	= super;
    cls->vtbl 	= NULL;
    cls->name   = name;

    // register to global constant.
    GV v = { .tt = GURU_TT_CLASS };
    v.cls = cls;
    const_object_add(sid, &v);

    return cls;
}

__GURU__ guru_proc *
mrbc_alloc_proc(const U8P name)
{
    guru_proc *proc = (guru_proc *)mrbc_alloc(sizeof(guru_proc));

    if (!proc) return NULL;

    proc->tt     = GURU_TT_PROC;
    proc->flag   = GURU_CFUNC;
    proc->refc   = 1;
    proc->sym_id = name2symid(name);
    proc->next   = NULL;
    proc->name   = name;

    return proc;
}

//================================================================
/*!@brief
  define class method or instance method.

  @param  vm		pointer to vm.
  @param  cls		pointer to class.
  @param  name		method name.
  @param  cfunc		pointer to function.
*/
__GURU__ guru_proc*
mrbc_define_method(guru_class *cls, const U8P name, guru_fptr cfunc)
{
    if (cls==NULL) cls = guru_class_object;		// set default to Object.

    guru_proc *proc = mrbc_alloc_proc(name);

    if (!proc) return NULL;

    MUTEX_LOCK(_mutex_cls);

    proc->func 	= cfunc;						// set function pointer
    proc->next 	= cls->vtbl;					// add as the new list head
    cls->vtbl 	= proc;

    MUTEX_FREE(_mutex_cls);

    return proc;
}
