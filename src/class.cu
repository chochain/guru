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

#include "console.h"
#include "class.h"

//================================================================
/*!@brief
  find class by object

  @param  vm
  @param  obj
  @return pointer to mrbc_class
*/
__GURU__ mrbc_class*
mrbc_get_class_by_object(mrbc_object *obj)
{
    mrbc_class *cls;

    switch (obj->tt) {
    case GURU_TT_TRUE:	  cls = mrbc_class_true;		break;
    case GURU_TT_FALSE:	  cls = mrbc_class_false; 	    break;
    case GURU_TT_NIL:	  cls = mrbc_class_nil;		    break;
    case GURU_TT_FIXNUM:  cls = mrbc_class_fixnum;	    break;
#if GURU_USE_FLOAT
    case GURU_TT_FLOAT:	  cls = mrbc_class_float; 	    break;
#endif
    case GURU_TT_SYMBOL:  cls = mrbc_class_symbol;	    break;

    case GURU_TT_OBJECT:  cls = obj->self->cls;     	break;
    case GURU_TT_CLASS:   cls = obj->cls;               break;
    case GURU_TT_PROC:	  cls = mrbc_class_proc;		break;
#if GURU_USE_STRING
    case GURU_TT_STRING:  cls = mrbc_class_string;	    break;
#endif
#if GURU_USE_ARRAY
    case GURU_TT_ARRAY:   cls = mrbc_class_array; 	    break;
    case GURU_TT_RANGE:	  cls = mrbc_class_range; 	    break;
    case GURU_TT_HASH:	  cls = mrbc_class_hash;		break;
#endif
    default:		      cls = mrbc_class_object;	    break;
    }
    return cls;
}

//================================================================
/*! get class by name

  @param  name		class name.
  @return		pointer to class object.
*/
__GURU__ mrbc_class*
mrbc_get_class_by_name(const char *name)
{
    mrbc_value v = const_object_get(name2symid(name));

    return (v.tt == GURU_TT_CLASS) ? v.cls : NULL;
}

//================================================================
/*!@brief
  find method from

  @param  vm
  @param  recv
  @param  sym_id
  @return
*/
__GURU__ mrbc_proc*
mrbc_get_class_method(mrbc_value rcv, mrbc_sym sid)
{
    mrbc_class *cls = mrbc_get_class_by_object(&rcv);

    while (cls != 0) {
        mrbc_proc *proc = cls->vtbl;
        while (proc != 0) {
            if (proc->sym_id == sid) {
                return proc;
            }
            proc = proc->next;
        }
        cls = cls->super;
    }
    return 0;
}

//================================================================
/*!@brief
  define class

  @param  vm		pointer to vm.
  @param  name		class name.
  @param  super		super class.
*/
__GURU__ mrbc_class*
mrbc_define_class(const char *name, mrbc_class *super)
{
    if (super == NULL) super = mrbc_class_object;  // set default to Object.

    mrbc_class *cls = mrbc_get_class_by_name(name);
    if (cls) return cls;

    // create a new class?
    cls = (mrbc_class *)mrbc_alloc(sizeof(mrbc_class));
    if (!cls) return NULL;			// ENOMEM

    mrbc_sym sid = name2symid(name);

    cls->sym_id = sid;
    cls->super 	= super;
    cls->vtbl 	= NULL;
#ifdef GURU_DEBUG
    cls->name 	= name;				// for debug; delete soon.
#endif

    // register to global constant.
    mrbc_value v = { .tt = GURU_TT_CLASS };
    v.cls    = cls;
    const_object_add(sid, &v);

    return cls;
}

__GURU__ mrbc_proc*
mrbc_proc_alloc(const char *name)
{
    mrbc_proc *proc = (mrbc_proc *)mrbc_alloc(sizeof(mrbc_proc));
    if (proc) {
        proc->refc   = 1;
        proc->sym_id = name2symid(name);
        proc->next   = 0;
#ifdef GURU_DEBUG
        proc->name   = name;	// for debug; delete soon.
#endif
    }
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
__GURU__ void
mrbc_define_method(mrbc_class *cls, const char *name, mrbc_func_t cfunc)
{
    if (cls==NULL) cls = mrbc_class_object;	// set default to Object.

    mrbc_proc *proc = mrbc_proc_alloc(name);

    proc->flag  |= GURU_PROC_C_FUNC;  			// c-func
    proc->func 	= cfunc;
    proc->next 	= cls->vtbl;

    cls->vtbl 	= proc;
}

// =============== ProcClass
__GURU__ void
c_proc_call(mrbc_value v[], int argc)
{
	// not suppose to come here
	assert(1==0);		// taken care by vm#op_send
}

//================================================================
// Object class
//================================================================
/*! Nop operator / method
 */
__GURU__ void
c_nop(mrbc_value v[], int argc)
{
    // nothing to do.
}



