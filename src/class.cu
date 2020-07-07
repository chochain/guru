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
#include "util.h"
#include "global.h"
#include "symbol.h"
#include "mmu.h"

#include "base.h"
#include "static.h"
#include "class.h"
#include "c_string.h"		// buffer

//================================================================
/*! (BETA) Call any method of the object, but written by C.

  @param  vm		pointer to vm.
  @param  v		see bellow example.
  @param  reg_ofs	see bellow example.
  @param  recv		pointer to receiver.
  @param  name		method name.
  @param  argc		num of params.

  @example
  void int_to_s(GR r[], S32 ri)
  {
  	  GR *rcv = &r[1];
  	  GR ret  = _send(v, rcv, "to_s", argc);
  	  RETURN_VAL(ret);
  }
*/
__GURU__ GR
_send(GR r[], GR *rcv, const char *method, U32 argc, ...)
{
    GR *regs = r + 2;	     			// allocate 2 for stack
    GS pid   = name2id((U8*)method);	// symbol lookup
    GP prc   = find_proc(r, pid);		// find method for receiver object

    ASSERT(prc);

    // create call stack.
    regs[0] = *ref_inc(rcv);			// create call stack, start with receiver object

    va_list ap;							// setup calling registers
    va_start(ap, argc);
    for (int i = 1; i <= argc+1; i++) {
        regs[i] = (i>argc) ? NIL : *va_arg(ap, GR *);
    }
    va_end(ap);

    _CALL(prc, regs, argc);				// call method, put return value in regs[0]

#if GURU_DEBUG
    GR *x = r;							// _wipe_stack
    for (int i=1; i<=argc+1; i++) {
    	*x++ = EMPTY;					// clean up the stack
    }
#endif
    return regs[0];
}

__GURU__ GR
inspect(GR *r, GR *obj)
{
	return _send(r, obj, "inspect", 0);
}

__GURU__ GR
kind_of(GR *r)		// whether v1 is a kind of v0
{
	return _send(r, r+1, "kind_of?", 1, r);
}

__GURU__ GP
find_class_by_id(GS cid)
{
	guru_class *cx = _CLS(guru_device_rom.cls);
	for (int i=0; i<guru_device_rom.ncls; i++, cx++) {
		if (cx->cid == cid) {
			return MEMOFF(cx);
		}
	}
	return 0;
}

//================================================================
/*!@brief
  find class by object

  @param  obj
  @return pointer to guru_class
*/
__GURU__ GP
find_class_by_obj(GR *r)
{
	switch (r->gt) {
	case GT_OBJ: return _CLS(GR_OBJ(r)->klass)->csrc;
	case GT_CLASS: {
		guru_class *cx = GR_CLS(r);
		return IS_SCLASS(r)
					? cx->klass ? cx->klass : guru_rom_get_class(GT_OBJ)
					: cx->csrc;
	}
	default: return guru_rom_get_class(r->gt);					// singleton class pointing back to core class
	}
}

//================================================================
/*!@brief
  walk linked list to find method from mtbl of class (and super class if needs to)

  @param  vm
  @param  recv
  @param  sid
  @return proc pointer
*/
#if CUDA_ENABLE_CDP
__GPU__ void
__scan_mtbl(S32 *idx, U32 cls, GS pid)
{
	U32 x = threadIdx.x + blockIdx.x * blockDim.x;
	guru_class *cx = _CLS(cls);
	if (x < cx->rc && (_PRC(cx->mtbl)+x)->pid==pid) {
		*idx = x;
	}
}
#else
__GURU__ GP
__scan_mtbl(guru_class *cx, GS pid)
{
	guru_proc *px = _PRC(cx->mtbl);				// sequential search thru the array
	for (int i=0; i<cx->rc; i++, px++) {		// TODO: parallel search (i.e. CDP, see above)
		if (px->pid==pid) {
#if CC_DEBUG
			U8 *cname = _RAW(cx->cid);
			U8 *pname = _RAW(px->pid);
			PRINTF("!!!mtbl[%d] hit %p:%p %s#%s -> %d\n", i, cx, px, cname, pname, pid);
#endif // CC_DEBUG
			return MEMOFF(px);
		}
	}
	return 0;
}

__GURU__ GP
__scan_flist(guru_class *cx, GS pid)
{
	GP prc = cx->flist;							// walk IR	guru_class *cx  = GR_CLS(r);
	int i=0;
	while (prc) {								// TODO: IREP should be added into guru_class->mtbl[]
		guru_proc *px = _PRC(prc);
		if (px->pid==pid) {
#if CC_DEBUG
			U8 *cname = _RAW(cx->cid);
			U8 *pname = _RAW(px->pid);
			PRINTF("!!!flst[%d] hit %p:%p %s#%s -> %d\n", i, cx, px, cname, pname, pid);
#endif // CC_DEBUG
			return prc;
		}
		i++;
		prc = px->next;
	}
	return 0;
}
#endif // CUDA_ENABLE_CDP

//================================================================
/*!@brief
  determine lexical scope

  @param  obj
  @return pointer to guru_class
*/
__GURU__ GP
find_proc(GR *r, GS pid)
{
	U32 mta = 0;									// metaclass flag
	GP  lex;
	switch (r->gt) {								// determine lexical scope (== vm_state->klass)
	case GT_OBJ: 	lex = GR_OBJ(r)->klass; 				break;
	case GT_CLASS:	lex = r->off; 	mta = !IS_TCLASS(r);	break;
	default:		lex = guru_rom_get_class(r->gt);
	}
	GP 	prc = 0;
	while (lex) {
    	guru_class *cx = _CLS(lex);
    	guru_class *mx = mta 						// meta-class or class itself
    			? (cx->klass ? _CLS(cx->klass) : NULL)
    			:  _CLS(cx->csrc);
#if CUDA_ENABLE_CDP
    		static __GURU__ S32 _proc_idx[32];
    		/* CC: hold! CUDA 10.2 profiler does not support CDP yet,
        	if (IS_BUILTIN(cls)) {
        		S32 *idx = &_proc_idx[threadIdx.x];
    		    *idx = -1;
        	    __find_proc<<<(cls->rc>>5)+1, 32>>>(idx, cls, sid);
        		GPU_CHK();
            	if (*idx>=0) return &cls->mtbl[*idx];
        	}
    		*/
#else
    	if (mx) {
    		prc = __scan_flist(mx, pid);			// TODO: combine flist into mtbl[]
    		if (prc) break;

    		prc = __scan_mtbl(mx, pid);				// search for C-functions
    		if (prc) break;
    	}
#endif // CUDA_ENABLE_CDP
    	lex = cx->super;
    }
#if CC_DEBUG
	U8* pname = _RAW(pid);
    PRINTF("!!!find_proc(%x, %d)=>%s %d[x%04x]\n", cls, pid, pname, prc, prc);
#endif // CC_DEBUG
    return prc;
}

//================================================================
/*!@brief
  define class

  @param  name		class name.
  @param  super		super class.
*/
__GURU__ GP
guru_define_class(guru_class *cx, GS cid, GP super)		// fill the ROM class storage
{
	if (!cx) {
		ASSERT(guru_device_rom.ncls < MAX_ROM_CLASS);
		cx = _CLS(guru_device_rom.cls) + guru_device_rom.ncls++;
	}
    cx->rc     = cx->n = 0;						// zero function defined yet
	cx->kt     = 0;								// default to user defined (i.e. non-builtin) class
    cx->cid    = cid;							// class name symbol id
    cx->ivar   = 0;								// class variables, lazily allocated when needed
    cx->klass  = 0;								// meta-class, lazily allocated when needed
    cx->csrc   = MEMOFF(cx);					// keep class id for constant lookup
    cx->super  = super;
    cx->mtbl   = 0;
    cx->flist  = 0;								// head of list

    return MEMOFF(cx);
}

//================================================================
/*!@brief
  include module into a class (by walking up module hierarchy

  @param super 	pointer to super class
  @param mod    pointer to module
*/
__GURU__ GP
guru_class_include(GP cls, GP mod)
{
	GP OBJ = guru_rom_get_class(GT_OBJ);

	guru_class *cx = _CLS(cls);
	guru_class *mx = _CLS(mod);
	if (mx->super != OBJ) {									// at the top of the class hierarchy?
		guru_class_include(cls, mx->super);					// climb up recursively
	}
	// inherit from root and cascade down the tree
	guru_class *dup = (guru_class*)guru_alloc(sizeof(guru_class));
	MEMCPY(dup, mx, sizeof(guru_class));					// deep copy so mtbl can be modified later

	dup->super = cx->super;									// insert module between parent and current class

	return cx->super = MEMOFF(dup);
}
//================================================================
/*!@brief
  add metaclass to a class

  @param  r			pointer to class variable
*/
__GURU__ GP
_cls_meta(GR *r)									// add metaclass to a class
{
	guru_class *cx  = GR_CLS(r);
	if (cx->klass) return cx->klass;				// return if already exists

	guru_class *scx = _CLS(cx->super);
	guru_class *mcx = (guru_class*)guru_alloc(sizeof(guru_class));

	GP skls = scx->klass ? scx->klass : guru_rom_get_class(GT_OBJ);
	GP mkls = guru_define_class(mcx, cx->cid, skls);

	if ((r+1)->gt == GT_CLASS) {					// extend module
		mcx->csrc = (r+1)->off;						// point to the original module, instead of the dup
	}
	return cx->klass = mkls;						// self pointing =~ metaclass
}

//================================================================
/*!@brief
  add metaclass to an object

  @param  r			pointer to object variable
*/
__GURU__ GP
_obj_meta(GR *r)
{
	guru_obj   *obj = GR_OBJ(r);
	GP         kls  = obj->klass;
	guru_class *cx  = _CLS(kls);
	if (IS_SINGLETON(cx)) return kls;				// return if exists already

	GP skls = guru_define_class(NULL, cx->cid, kls);
	_CLS(skls)->kt |= CLASS_SINGLETON;

	return obj->klass = skls;						// set singleton class
}

__GURU__ GP
guru_create_metaclass(GR *r)
{
	GP kls = r->gt==GT_OBJ
		? _obj_meta(r)								// singleton class of an object
		: (r->gt==GT_CLASS ? _cls_meta(r) : 0);		// extending a class

	return kls;
}

