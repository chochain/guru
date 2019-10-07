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
    case GT_CLASS:   return v->cls;
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
	// TODO: heavy-weight method, add a cache here to speed up lookup
    guru_proc  *p;
    for (guru_class *cls=class_by_obj(v); cls!=NULL; cls=cls->super) {	// search up hierarchy tree
//    	p = (obj->gt==GT_CLASS) ? cls->meta->vtbl : cls->vtbl;
        for (p=cls->vtbl; p && (p->sid != sid); p=p->next);				// linear search thru class or meta vtbl
        if (p) return p;												// break if found
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

    guru_class *cls1, *cls0 = _name2class(name);
    if (cls0) return cls0;

    // class does not exist, create a new one
    GS sid0 = name2id(name);
    cls0 = (guru_class *)guru_alloc(sizeof(guru_class)*2);
    cls1 = cls0+1;
    cls0->sid    = sid0;
    cls0->super  = super;
    cls0->vtbl 	 = NULL;
    cls0->cvar   = NULL;
    cls0->meta   = (cls0+1);
#ifdef GURU_DEBUG
    // change to sid later
    cls0->name   = (char *)id2name(sid0);		// retrive from stored symbol table
#endif

    // meta class
    U32 sz  = STRLEN(name)+1;
    U8  buf[256];
    MEMCPY(buf,   "Class:", 6);
    MEMCPY(buf+6, name, sz);
    GS sid1 = name2id(buf);
    cls1->sid   = sid1;
    cls1->super = guru_class_object;
    cls1->vtbl  = NULL;							// stores class (static) methods
    cls1->cvar  = NULL;
    cls1->meta  = NULL;

    // register to global constant.
    GV v0; { v0.gt = GT_CLASS; v0.acl = 0; v0.fil=0xcccccccc; v0.cls = cls0; }
    GV v1; { v1.gt = GT_CLASS; v1.acl = 0; v1.fil=0xcccccccc; v1.cls = cls1; }
    const_set(sid0, &v0);
    const_set(sid1, &v1);

    return cls0;
}

__GURU__ guru_proc *
_alloc_proc(guru_class *cls, const U8 *name)
{
    guru_proc *proc = (guru_proc *)guru_alloc(sizeof(guru_proc));

    proc->sid    = name2id(name);
    proc->next   = NULL;
#ifdef GURU_DEBUG
    proc->cname  = (char *)id2name(cls->sid);
    proc->name   = (char *)id2name(proc->sid);
#endif
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
guru_define_method(guru_class *cls, const U8 *name, guru_fptr cfunc)
{
    if (cls==NULL) cls = guru_class_object;		// set default to Object.

    guru_proc *prc = _alloc_proc(cls, name);	// with sid assigned

    _LOCK;

    prc->func 	= cfunc;						// set function pointer
    prc->irep   = NULL;
    prc->next 	= cls->vtbl;					// add as the new list head
    cls->vtbl 	= prc;

    _UNLOCK;

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

