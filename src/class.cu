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
    GP prc   = proc_by_id(r, pid);		// find method for receiver object

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
class_by_id(GS cid)
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

  @param  vm
  @param  obj
  @return pointer to guru_class
*/
__GURU__ GP
class_by_obj(GR *r)
{
#if CC_DEBUG
	PRINTF("!!!class_by_obj(%p) r->gt=%d, r->off=x%x: ", r, r->gt, r->off);
	const char *tname[] = {
			"Empty", "Nil", "False", "True", "Integer", "Float", "Symbol", "Sys",
            "", "Proc", "", "Array", "String", "Range", "Hash", "???"
	};
#endif // CC_DEBUG
	GP ret;
	switch (r->gt) {
	case GT_OBJ: {						// object class
    	ret = GR_OBJ(r)->cls;
#if CC_DEBUG
    	PRINTF(" OBJ");
#endif // CC_DEBUG
    } break;
    case GT_CLASS: {					// singleton class
    	guru_class *cx = GR_CLS(r);
    	GP meta = cx->meta ? cx->meta : guru_rom_get_class(GT_OBJ);
    	GP cls  = r->off;
#if CC_DEBUG
    	PRINTF(" CLS[x%04x]=%s:%p", cls, _RAW(cx->cid), cx);
#endif // CC_DEBUG
    	ret  = IS_BUILTIN(cx)
    		? cls
    		: (IS_SCLASS(r) ? meta : (IS_TCLASS(r) ? cls : meta));
    } break;
    default:
    	ret = guru_rom_get_class(r->gt);
#if CC_DEBUG
        PRINTF(" CLS[x%04x]=%s:%p", ret, tname[r->gt], GR_CLS(r));
#endif // CC_DEBUG
    }
#if CC_DEBUG
	PRINTF("=> x%04x\n", ret);
#endif // CC_DEBUG
	return ret;
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
	for (int i=0; i<cx->rc; i++, px++) {	// TODO: parallel search (i.e. CDP, see above)
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
	GP prc = cx->flist;							// walk IREP linked-list
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

__GURU__ GP
proc_by_id(GR *r, GS pid)
{
	GP cls = class_by_obj(r);
	GP prc = 0;
	while (cls) {
    	guru_class *cx = _CLS(cls);
    	if (IS_META(cx)) {
    		cx = _CLS(cx->ctbl);		// use original singleton class (template)
    	}
    	prc = __scan_flist(cx, pid);	// TODO: combine flist into mtbl[]
    	if (prc) break;

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
    	prc = __scan_mtbl(cx, pid);		// search for C-functions
    	if (prc) break;
#endif // CUDA_ENABLE_CDP

    	cls = cx->super;
    }
#if CC_DEBUG
	U8* pname = _RAW(pid);
    PRINTF("!!!proc_by_id(%p, %d)=>%s %d[x%04x]\n", r, pid, pname, prc, prc);
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
    cx->ctbl   = MEMOFF(cx);					// keep class id for constant lookup
    cx->meta   = 0;								// meta-class, lazily allocated when needed
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
	GP top = guru_rom_get_class(GT_OBJ);

	guru_class *cx  = _CLS(cls);
	guru_class *mcx = _CLS(mod);
	if (mcx->super != top) {								// at the top of the class hierarchy?
		guru_class_include(cls, mcx->super);				// climb up recursively
	}
	guru_class *dup = (guru_class*)guru_alloc(sizeof(guru_class));
	MEMCPY(dup, mcx, sizeof(guru_class));					// deep copy so mtbl can be modified later

	dup->kt    |= USER_META_CLASS;
	dup->meta   = mod;										// pointing backward, for get_const
	dup->super  = cx->super;								// insert module between parent and current class

	return cx->super = MEMOFF(dup);
}
//================================================================
/*!@brief
  add metaclass to a class

  @param  r			pointer to class variable
*/
__GURU__ GP
guru_class_add_meta(GR *r)												// lazy add metaclass to a class
{
	ASSERT(r->gt==GT_CLASS);

	guru_class *cx  = GR_CLS(r);
	if (cx->meta) return cx->meta;

	// lazily create the metaclass
	guru_class *mcx = (guru_class*)guru_alloc(sizeof(guru_class));
	guru_class *scx = _CLS(cx->super);
	GP scls = scx->meta ? scx->meta : guru_rom_get_class(GT_OBJ);
	GP mcls = guru_define_class(mcx, cx->cid, scls);

	_CLS(mcls)->kt |= USER_META_CLASS;

	return cx->meta = mcls;												// self pointing =~ metaclass
}


