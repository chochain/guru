/*! @file
  @brief
  GURU class factory and building functions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <stdarg.h>
#include "guru.h"
#include "global.h"
#include "symbol.h"
#include "mmu.h"

#include "base.h"
#include "class.h"

#define _LOCK		{ MUTEX_LOCK(_mutex_cls); }
#define _UNLOCK 	{ MUTEX_FREE(_mutex_cls); }

__GURU__ U32 _mutex_cls;

//================================================================
/*! (BETA) Call any method of the object, but written by C.

  @param  vm		pointer to vm.
  @param  v		see bellow example.
  @param  reg_ofs	see bellow example.
  @param  recv		pointer to receiver.
  @param  name		method name.
  @param  argc		num of params.

  @example
  void int_to_s(GR r[], U32 ri)
  {
  	  GR *rcv = &r[1];
  	  GR ret  = _send(v, rcv, "to_s", argc);
  	  RETURN_VAL(ret);
  }
*/
__GURU__ GR
_send(GR r[], GR *rcv, const U8 *method, U32 argc, ...)
{
    GR *regs = r + 2;	     		// allocate 2 for stack
    GS sid   = name2id(method);		// symbol lookup

    guru_proc  *m = proc_by_sid(r, sid);	// find method for receiver object
    ASSERT(m);

    // create call stack.
    regs[0] = *ref_inc(rcv);		// create call stack, start with receiver object

    va_list ap;						// setup calling registers
    va_start(ap, argc);
    for (U32 i = 1; i <= argc+1; i++) {
        regs[i] = (i>argc) ? NIL : *va_arg(ap, GR *);
    }
    va_end(ap);

    m->func(regs, argc);			// call method, put return value in regs[0]

#if GURU_DEBUG
    GR *x = r;						// _wipe_stack
    for (U32 i=1; i<=argc+1; i++) {
    	*x++ = EMPTY;				// clean up the stack
    }
#endif
    return regs[0];
}

__GURU__ GR
inspect(GR *r, GR *obj)
{
	return _send(r, obj, (U8*)"inspect", 0);
}

__GURU__ GR
kind_of(GR *r)		// whether v1 is a kind of v0
{
	return _send(r, r+1, (U8*)"kind_of?", 1, r);
}

//================================================================
/*!@brief
  find class by object

  @param  vm
  @param  obj
  @return pointer to guru_class
*/
__GURU__ guru_class*
class_by_obj(GR *r)
{
	guru_class *scls;
	guru_class *cls = GR_CLS(r);

	switch (r->gt) {
    case GT_OBJ: return GR_OBJ(r)->cls;
    case GT_CLASS:
    	scls = cls->meta ? cls->meta : guru_rom_get_class(GT_OBJ);
    	return IS_BUILTIN(cls)
    		? cls
    		: (IS_SCLASS(r) ? scls : (IS_SELF(r) ? cls : scls));
    default: return guru_rom_get_class(r->gt);
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
    GR *r  = const_get(sid);

    return (r->gt==GT_CLASS) ? GR_CLS(r) : NULL;
}

//================================================================
/*!@brief
  walk linked list to find method from vtbl of class (and super class if needs to)

  @param  vm
  @param  recv
  @param  sid
  @return proc pointer
*/
#if CUDA_PROFILE_CDP
__GPU__ void
__find_proc(S32 *idx, guru_class *cls, GS sid)
{
	U32 x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x<cls->rc && cls->vtbl[x].sid==sid) {
		*idx = x;
	}
}
#else
__GURU__ S32
__find_proc(guru_class *cls, GS sid)
{
	for (int i=0; i<cls->rc; i++) {
		if (cls->vtbl[i].sid==sid) return i;
	}
	return -1;
}
#endif // CUDA_PROFILE_CDP

__GURU__ S32 _proc_idx[32];
__GURU__ guru_proc*
proc_by_sid(GR *r, GS sid)
{
#if CC_DEBUG
	U8* fname = id2name(sid);
    printf("proc_by_sid:%s=>%d(0x%02x)\n", fname, sid, sid);
    for (guru_class *cls=class_by_obj(r); cls; cls=cls->super) {	// search up class hierarchy
        printf("\t%p:sc=%d,self=%d:%s\n", cls, IS_SCLASS(r), IS_SELF(v), cls->name);
#else
    for (guru_class *cls=class_by_obj(r); cls; cls=cls->super) {	// search up class hierarchy
#endif // CC_DEBUG
#if CUDA_PROFILE_CDP
    	/* CC: hold! CUDA 10.2 profiler does not support CDP yet,
        if (IS_BUILTIN(cls)) {
        	S32 *idx = &_proc_idx[threadIdx.x];
        	*idx = -1;
        	__find_proc<<<(cls->rc>>5)+1, 32>>>(idx, cls, sid);
        	GPU_CHK();
            if (*idx>=0) return &cls->vtbl[*idx];
        }
        */
#else
    	S32 idx = __find_proc(cls, sid);
    	if (idx>=0) return &cls->vtbl[idx];
#endif // CUDA_PROFILE_CDP
    	guru_proc *p;
        for (p=cls->flist; p && (p->sid != sid); p=p->next);		// linear search thru class or meta vtbl
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
	GS sid = create_sym(name);

    cls->rc     = cls->n = cls->kt = 0;	// BUILT-IN class
    cls->sid    = sid;
    cls->var    = NULL;					// class variables, lazily allocated when needed
    cls->meta 	= NULL;					// meta-class, lazily allocated when needed
    cls->super  = super;
    cls->vtbl   = NULL;
    cls->flist  = NULL;					// head of list
#ifdef GURU_DEBUG
    cls->name   = id2name(sid);			// retrieve from stored symbol table (the one caller passed might be destroyed)
#endif

    GR r; { r.gt=GT_CLASS; r.acl=0; r.cls=MEMOFF(cls); }
    const_set(sid, &r);					// register new class in constant cache
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
    prc->sid   = create_sym(name);
    prc->kt    = 0;								// C-function (from BUILT-IN class)
    prc->func  = cfunc;							// set function pointer

    _LOCK;
    prc->next  = cls->flist;					// add as the new list head
    cls->flist = prc;
    _UNLOCK;

#ifdef GURU_DEBUG
    prc->cname = id2name(cls->sid);
    prc->name  = id2name(prc->sid);
#endif

    return prc;
}

//================================================================
/*!@brief
  add metaclass to a class

  @param  vm		pointer to vm.
  @param  cls		pointer to class.
  @param  name		method name.
  @param  cfunc		pointer to function.
*/
__GURU__ guru_class*
guru_class_add_meta(GR *r)						// lazy add metaclass to a class
{
	ASSERT(r->gt==GT_CLASS);

	guru_class *cls = GR_CLS(r);
	if (cls->meta!=NULL) return cls->meta;

	// lazily create the metaclass
	const U8	*name = (U8*)"_meta";
	guru_class 	*mcls = guru_define_class(name, guru_rom_get_class(GT_OBJ));

	return cls->meta = mcls;					// self pointing =~ metaclass
}

//================================================================
/* methods to add builtin (ROM) class/proc for GURU
 * it uses (const U8 *) for static string
 */
__GURU__ guru_class *_class_list = NULL;

__GURU__ guru_class*
guru_rom_get_class(GT idx) {
	if (_class_list==NULL) {					// lazy allocation
		_class_list = (guru_class*)guru_alloc(sizeof(guru_class)*GT_MAX);
	}
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

    Vfunc *fp = (Vfunc*)&vtbl[0];				// TODO: nvcc allocates very sparsely for String literals
    for (U32 i=0; i<n; i++, prc++, fp++) {
    	prc->n    = 0;
    	prc->sid  = create_sym((U8*)fp->name);
    	prc->kt   = 0;							// built-in class type (not USER_DEF_CLASS)
    	prc->func = fp->func;
#ifdef GURU_DEBUG
    	prc->cname= id2name(cls->sid);
    	prc->name = id2name(prc->sid);
#endif
    }
	return cls;
}
