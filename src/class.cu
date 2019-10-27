/*! @file
  @brief
  GURU class building functions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <stdarg.h>
#include "guru.h"
#include "class.h"
#include "mmu.h"
#include "static.h"
#include "value.h"
#include "global.h"
#include "symbol.h"

#define _LOCK		{ MUTEX_LOCK(_mutex_cls); }
#define _UNLOCK 	{ MUTEX_FREE(_mutex_cls); }

__GURU__ U32 _mutex_cls;

//================================================================
/*!@brief
  find class by object

  @param  vm
  @param  obj
  @return pointer to guru_class
*/
__GURU__ guru_class*
class_by_obj(GV *v)
{
    switch (v->gt) {
    case GT_TRUE:	 return guru_class_true;
    case GT_FALSE:	 return guru_class_false;
    case GT_NIL:	 return guru_class_nil;
    case GT_INT:     return guru_class_int;
#if GURU_USE_FLOAT
    case GT_FLOAT:	 return	guru_class_float;
#endif // GURU_USE_FLOAT
    case GT_SYM:  	 return guru_class_symbol;
    case GT_OBJ:  	 return v->self->cls;
    case GT_CLASS:
    	return IS_BUILTIN(v) || IS_TCLASS(v)
    		? v->cls
    		: (v->cls->cls ? v->cls->cls : guru_class_object);
    case GT_PROC:	 return guru_class_proc;
#if GURU_USE_STRING
    case GT_STR:     return guru_class_string;
#endif // GURU_USE_STRING
#if GURU_USE_ARRAY
    case GT_ARRAY:   return guru_class_array;
    case GT_RANGE:	 return guru_class_range;
    case GT_HASH:	 return guru_class_hash;
#endif  // GURU_USE_ARRAY
    default:		 return guru_class_object;
    }
}

//================================================================
/*! get class by name

  @param  name		class name.
  @return		pointer to class object.
*/
__GURU__ guru_class*
_name2class(const U8 *name)
{
	GS sid = name2id(name);
    GV *v  = const_get(sid);

    return (v->gt==GT_CLASS) ? v->cls : NULL;
}

//================================================================
/*!@brief
  walk linked list to find method from vtbl of class (and super class if needs to)

  @param  vm
  @param  recv
  @param  sid
  @return proc pointer
*/
__GURU__ guru_proc*
proc_by_sid(GV *v, GS sid)
{
	// TODO: heavy-weight method, use Dynamic Parallelism or a cache to speed up lookup
    guru_proc  *p;
    for (guru_class *cls=class_by_obj(v); cls; cls=cls->super) {// search up class hierarchy
        for (p=cls->vtbl; p && (p->sid != sid); p=p->next);		// linear search thru class or meta vtbl
        if (p) return p;										// break if found
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
guru_define_class(const U8 *name, guru_class *super)
{
    if (super == NULL) super = guru_class_object;  // set default to Object.

    guru_class *cls = _name2class(name);
    if (cls) return cls;

    // class does not exist, create a new one
    GS sid = name2id(name);

    cls = (guru_class *)guru_alloc(sizeof(guru_class));
    cls->rc     = 0;
    cls->meta   = 0;					// ~META_FLAG
    cls->sid    = sid;
    cls->super  = super;
    cls->vtbl 	= NULL;					// head of list
    cls->ivar   = NULL;					// lazily allocated when needed
    cls->cls 	= NULL;					// meta-class, lazily allocated when needed
#ifdef GURU_DEBUG
    // change to sid later
    cls->name   = (char *)id2name(sid);	// retrive from stored symbol table (the one caller passed might be destroyed)
#endif
    GV v; { v.gt=GT_CLASS; v.acl=0; v.self=(guru_obj*)cls; }
    const_set(sid, &v);					// register new class in constant cache

    return cls;
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
guru_define_method(guru_class *cls, const U8 *name, guru_fptr cfunc)
{
    if (cls==NULL) cls = guru_class_object;		// set default to Object.

    guru_proc *prc = (guru_proc *)guru_alloc(sizeof(guru_proc));

    prc->meta = 0;								// C-function
    prc->n    = 0;								// No LAMBDA register file
    prc->sid  = name2id(name);
    prc->func = cfunc;							// set function pointer

    _LOCK;
    prc->next = cls->vtbl;						// add as the new list head
    cls->vtbl = prc;
    _UNLOCK;

#ifdef GURU_DEBUG
    prc->cname  = (char *)id2name(cls->sid);
    prc->name   = (char *)id2name(prc->sid);
#endif

    return prc;
}

//================================================================
/* methods to add core class/proc for GURU
 * it uses (const U8 *) for static string
 */
__GURU__ guru_class*
guru_add_class(const char *name, guru_class *super, Vfunc vtbl[], int n)
{
	guru_class *c = guru_define_class((U8*)name, super);

	Vfunc *p = vtbl;
	for (U32 i=0; i<n && p && p->func; i++, p++) {
		guru_define_method(c, (U8*)p->name, p->func);
	}
	return c;
}


