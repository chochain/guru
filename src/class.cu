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
#define CHK_ERR		ASSERT(cudaGetLastError()==cudaSuccess)

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
    GP prc   = proc_by_sid(r, sid);	// find method for receiver object

    ASSERT(prc);

    // create call stack.
    regs[0] = *ref_inc(rcv);		// create call stack, start with receiver object

    va_list ap;						// setup calling registers
    va_start(ap, argc);
    for (U32 i = 1; i <= argc+1; i++) {
        regs[i] = (i>argc) ? NIL : *va_arg(ap, GR *);
    }
    va_end(ap);

    _CALL(prc, regs, argc);	// call method, put return value in regs[0]

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
__GURU__ GP
class_by_obj(GR *r)
{
#if CC_DEBUG
	PRINTF("!!!class_by_obj(%p) r->gt=%d, r->off=x%x: ", r, r->gt, r->off);
#endif // CC_DEBUG
	GP ret;
	switch (r->gt) {
	case GT_OBJ: {
    	ret = GR_OBJ(r)->cls;
#if CC_DEBUG
    	PRINTF("OBJ");
#endif // CC_DEBUG
    } break;
    case GT_CLASS: {
    	guru_class *cx = GR_CLS(r);
    	GP scls = cx->meta ? cx->meta : guru_rom_get_class(GT_OBJ);
    	GP cls  = r->off;
#if CC_DEBUG
    	PRINTF(" CLS[x%04x]=%s:%p", cls, MEMPTR(cx->name), cx);
#endif // CC_DEBUG
    	ret  = IS_BUILTIN(cx)
    		? cls
    		: (IS_SCLASS(r) ? scls : (IS_SELF(r) ? cls : scls));
    } break;
    default:
#if CC_DEBUG
    	PRINTF("???");
#endif // CC_DEBUG
    	ret = guru_rom_get_class(r->gt);
    }
#if CC_DEBUG
	PRINTF("=> x%04x\n", ret);
#endif // CC_DEBUG
	return ret;
}

//================================================================
/*! get class by name

  @param  name		class name.
  @return		pointer to class object.
*/
__GURU__ GP
_name2class(const U8 *name)
{
	GS sid = name2id(name);
    GR *r  = const_get(sid);

    return (r->gt==GT_CLASS) ? r->off : 0;
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
__scan_vtbl(S32 *idx, U32 cls, GS sid)
{
	U32 x = threadIdx.x + blockIdx.x * blockDim.x;
	guru_class *cx = _CLS(cls);
	if (x < cx->rc && (_PRC(cx->vtbl)+x)->sid==sid) {
		*idx = x;
	}
}
#else
__GURU__ GP
__scan_vtbl(guru_class *cx, GS sid)
{
	U8 *cname = MEMPTR(cx->name);
	guru_proc *px = _PRC(cx->vtbl);				// sequential search thru the array
	for (int i=0; i < cx->rc; i++, px++) {		// TODO: parallel search (i.e. CDP, see above)
#if CC_DEBUG
		PRINTF("!!!vtbl scaning %p:%s[%2d] %p:%s->%d == %d\n", cx, cname, i, px, MEMPTR(px->name), px->sid, sid);
#endif // CC_DEBUG
		if (px->sid==sid) return MEMOFF(px);
	}
	return 0;
}

__GURU__ GP
__scan_flist(guru_class *cx, GS sid)
{
	U8 *cname = MEMPTR(cx->name);
	GP prc = cx->flist;							// walk IREP linked-list
	while (prc) {								// TODO: IREP should be added into guru_class->vtbl[]
		guru_proc *px = _PRC(prc);
#if CC_DEBUG
		PRINTF("!!!flst scaning %p:%s %p:%s->%d == %d\n", cx, cname, px, MEMPTR(px->name), px->sid, sid);
#endif // CC_DEBUG
		if (px->sid==sid) {
			return prc;
		}
		prc = px->next;
	}
	return 0;
}
#endif // CUDA_PROFILE_CDP

__GURU__ S32 _proc_idx[32];
__GURU__ GP
proc_by_sid(GR *r, GS sid)
{
    GP cls = class_by_obj(r);
    GP prc = 0;

    while (cls) {
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
    	guru_class *cx = _CLS(cls);
    	prc = __scan_vtbl(cx, sid);		// search for C-functions
    	if (prc) break;
#endif // CUDA_PROFILE_CDP

    	prc = __scan_flist(cx, sid);	// TODO: combine flist into vtbl[]
    	if (prc) break;

    	cls = cx->super;
    }
#if CC_DEBUG
	U8* fname = id2name(sid);
    PRINTF("!!!proc_by_sid(%p, %d)=>%s %d[x%04x]\n", r, sid, fname, prc, prc);
#endif // CC_DEBUG
    return prc;
}

//================================================================
/*!@brief
  define class

  @param  vm		pointer to vm.
  @param  name		class name.
  @param  super		super class.
*/
__GURU__ guru_class*
_define_class(const U8 *name, GP cls, GP super)
{
	guru_class *cx = _CLS(cls);
	GS         sid = create_sym(name);

    cx->rc     = cx->n = cx->kt = 0;	// BUILT-IN class
    cx->sid    = sid;
    cx->var    = 0;						// class variables, lazily allocated when needed
    cx->meta   = 0;						// meta-class, lazily allocated when needed
    cx->super  = super;
    cx->vtbl   = 0;
    cx->flist  = 0;						// head of list
#ifdef GURU_DEBUG
    cx->name   = MEMOFF(id2name(sid));	// retrieve from stored symbol table (the one caller passed might be destroyed)
#endif

    GR  r { .gt=GT_CLASS, .acl=0, .oid=0, { .off=cls }};
    const_set(sid, &r);					// register new class in constant cache

    return cx;
}

__GURU__ GP
guru_define_class(const U8 *name, GP super)
{
    GP cls = _name2class(name);
    if (cls) return cls;

    // class does not exist, create a new one
    guru_class *cx = (guru_class *)guru_alloc(sizeof(guru_class));
    _define_class(name, (cls=MEMOFF(cx)), super);

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
__GURU__ GP
guru_define_method(GP cls, const U8 *name, GP cfunc)
{
    if (!cls) cls = guru_rom_get_class(GT_OBJ);	// set default to Object.

    guru_proc  *px = (guru_proc*)guru_alloc(sizeof(guru_proc));
    guru_class *cx = _CLS(cls);

    px->rc = px->kt = px->n = 0;				// No LAMBDA register file, C-function (from BUILT-IN class)
    px->sid   = create_sym(name);
    px->func  = cfunc;							// set function pointer

    _LOCK;
    px->next  = cx->flist;						// add as the new list head
    cx->flist = MEMOFF(px);						// TODO: change to array implementation
    _UNLOCK;

#ifdef GURU_DEBUG
    px->cname = MEMOFF(id2name(cx->sid));
    px->name  = MEMOFF(id2name(px->sid));
#endif

    return MEMOFF(px);
}

//================================================================
/*!@brief
  add metaclass to a class

  @param  vm		pointer to vm.
  @param  cls		pointer to class.
  @param  name		method name.
  @param  cfunc		pointer to function.
*/
__GURU__ GP
guru_class_add_meta(GR *r)						// lazy add metaclass to a class
{
	ASSERT(r->gt==GT_CLASS);

	guru_class *cx = GR_CLS(r);
	if (cx->meta) return cx->meta;

	// lazily create the metaclass
	U8 *name = (U8*)"_meta";
	GP mcls  = guru_define_class(name, guru_rom_get_class(GT_OBJ));

	return cx->meta = mcls;					// self pointing =~ metaclass
}

//================================================================
/* methods to add builtin (ROM) class/proc for GURU
 * it uses (const U8 *) for static string
 */
__GURU__ guru_class *_class_list = NULL;

__GURU__ GP
guru_rom_get_class(GT idx) {
	if (_class_list==NULL) {					// lazy allocation
		_class_list = (guru_class*)guru_alloc(sizeof(guru_class)*GT_MAX);
	}
	return idx==GT_EMPTY ? 0 : MEMOFF(&_class_list[idx]);
}

__GURU__ GP
guru_rom_set_class(GT cidx, const char *name, GT super_cidx, const Vfunc vtbl[], int n)
{
	GP cls   = guru_rom_get_class(cidx);
	GP super = guru_rom_get_class(super_cidx);

	guru_class *cx = _define_class((U8*)name, cls, super);
    guru_proc  *px = (guru_proc *)guru_alloc(sizeof(guru_proc) * n);
    cx->rc   = n;								// number of built-in functions
    cx->vtbl = MEMOFF(px);						// built-in proc list

    Vfunc *fp = (Vfunc*)vtbl;					// TODO: nvcc allocates very sparsely for String literals
    for (U32 i=0; i<n; i++, px++, fp++) {
    	px->rc   = px->kt = px->n = 0;			// built-in class type (not USER_DEF_CLASS)
    	px->sid  = create_sym((U8*)fp->name);	// raw string function table defined in code
    	px->func = MEMOFF(fp->func);
    	px->next = 0;
#ifdef GURU_DEBUG
    	px->cname= MEMOFF(id2name(cx->sid));
    	px->name = MEMOFF(id2name(px->sid));
#endif
    }
	return cls;
}
