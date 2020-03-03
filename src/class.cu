/*! @file
  @brief
  GURU class building functions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <stdio.h>
#include <stdarg.h>
#include "guru.h"
#include "class.h"
#include "mmu.h"
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
	guru_class *scls;

	switch (v->gt) {
    case GT_OBJ: return v->self->cls;
    case GT_CLASS:
    	scls = v->cls->cls ? v->cls->cls : guru_rom_get_class(GT_OBJ);
    	return IS_BUILTIN(v->cls)
    		? v->cls
    		: (IS_SCLASS(v) ? scls : (IS_SELF(v) ? v->cls : scls));
    default: return guru_rom_get_class(v->gt);
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
__GPU__ void
__find_proc(S32 *idx, guru_class *cls, GS sid)
{
	U32 x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x<cls->rc && cls->vtbl[x].sid==sid) {
		*idx = x;
	}
}

__GURU__ S32 _proc_idx[32];
__GURU__ guru_proc*
proc_by_sid(GV *v, GS sid)
{
#if CC_DEBUG
	U8* fname = id2name(sid);
    printf("proc_by_sid:%s=>%d(0x%02x)\n", fname, sid, sid);
    for (guru_class *cls=class_by_obj(v); cls; cls=cls->super) {	// search up class hierarchy
        printf("\t%p:sc=%d,self=%d:%s\n", cls, IS_SCLASS(v), IS_SELF(v), cls->name);
#else
    for (guru_class *cls=class_by_obj(v); cls; cls=cls->super) {	// search up class hierarchy
#endif // CC_DEBUG
        if (IS_BUILTIN(cls)) {
        	S32 *idx = &_proc_idx[threadIdx.x];
        	*idx = -1;
        	__find_proc<<<(cls->rc>>5)+1, 32>>>(idx, cls, sid);
        	SYNC_CHK();
            if (*idx>=0) return &cls->vtbl[*idx];
        }
    	guru_proc *p;
        for (p=cls->plist; p && (p->sid != sid); p=p->next);		// linear search thru class or meta vtbl
        if (p) return p;											// break if found
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
__GURU__ void
_define_class(const U8 *name, guru_class *cls, guru_class *super)
{
	GS sid = new_sym(name);

    cls->rc     = 0;
    cls->kt     = 0;					// BUILT-IN classes
    cls->sid    = sid;
    cls->super  = super;
    cls->vtbl   = NULL;
    cls->plist  = NULL;					// head of list
    cls->ivar   = NULL;					// lazily allocated when needed
    cls->cls 	= NULL;					// meta-class, lazily allocated when needed
#ifdef GURU_DEBUG
    cls->name   = (char *)id2name(sid);	// retrieve from stored symbol table (the one caller passed might be destroyed)
#endif

    GV v; { v.gt=GT_CLASS; v.acl=0; v.self=(guru_obj*)cls; }
    const_set(sid, &v);					// register new class in constant cache
}

__GURU__ guru_class*
guru_define_class(const U8 *name, guru_class *super)
{
    if (super == NULL) super = guru_rom_get_class(GT_OBJ);  // set default to Object.

    guru_class *cls = _name2class(name);
    if (cls) return cls;

    // class does not exist, create a new one
    cls = (guru_class *)guru_alloc(sizeof(guru_class));
    _define_class(name, cls, super);

#if CC_DEBUG
    printf("%p:%s defined\n", cls, name);
#endif // CC_DEBUG
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
    if (cls==NULL) cls = guru_rom_get_class(GT_OBJ);		// set default to Object.

    guru_proc *prc = (guru_proc *)guru_alloc(sizeof(guru_proc));

    prc->n     = 0;								// No LAMBDA register file
    prc->sid   = new_sym(name);
    prc->kt    = 0;								// C-function (from BUILT-IN class)
    prc->func  = cfunc;							// set function pointer

    _LOCK;
    prc->next  = cls->plist;					// add as the new list head
    cls->plist = prc;
    _UNLOCK;

#ifdef GURU_DEBUG
    prc->cname = (char *)id2name(cls->sid);
    prc->name  = (char *)id2name(prc->sid);
#endif

    return prc;
}

//================================================================
/* methods to add builtin (ROM) class/proc for GURU
 * it uses (const U8 *) for static string
 */
__GURU__ guru_class _class_list[GT_MAX];

__GURU__ guru_class*
guru_rom_get_class(GT idx) {
	return idx==GT_EMPTY ? NULL : &_class_list[idx];
}

__GURU__ guru_class*
guru_rom_set_class(GT cidx, const char *name, GT super_cidx, const Vfunc vtbl[], int n)
{
	guru_class *cls   = guru_rom_get_class(cidx);
	guru_class *super = guru_rom_get_class(super_cidx);
	_define_class((U8*)name, cls, super);

    guru_proc  *prc = (guru_proc *)guru_alloc(sizeof(guru_proc) * n);
    cls->rc   = n;								// number of built-in functions
    cls->vtbl = prc;							// built-in proc list

    for (U32 i=0; i<n; i++, prc++) {
    	prc->n    = 0;							// NOT USER_DEF_CLASS
    	prc->sid  = new_sym((U8*)vtbl[i].name);
    	prc->kt   = 0;							// built-in class type
    	prc->func = vtbl[i].func;
#ifdef GURU_DEBUG
    	prc->cname= (char *)id2name(cls->sid);
    	prc->name = (char *)id2name(prc->sid);
#endif
    }
	return cls;
}

