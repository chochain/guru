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
#include "static.h"
#include "mmu.h"

#include "base.h"
#include "class.h"

class ClassMgr::Impl
{
public:
	//================================================================
	/*! get class by name

	  @param  name		class name.
	  @return		pointer to class object.
	*/
	__GURU__ GP
	name2class(const U8 *name)
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
	#if CUDA_ENABLE_CDP
	static __GPU__ void
	scan_vtbl(S32 *idx, U32 cls, GS pid)
	{
		U32 x = threadIdx.x + blockIdx.x * blockDim.x;
		guru_class *cx = _CLS(cls);
		if (x < cx->rc && (_PRC(cx->vtbl)+x)->pid==pid) {
			*idx = x;
		}
	}
	#else
	__GURU__ GP
	scan_vtbl(guru_class *cx, GS pid)
	{
		guru_proc *px = _PRC(cx->vtbl);				// sequential search thru the array
		for (int i=0; i < cx->rc; i++, px++) {		// TODO: parallel search (i.e. CDP, see above)
            if (px->pid==pid) {
	#if CC_DEBUG
                U8 *cname = _RAW(cx->cid);
                U8 *pname = _RAW(px->pid);
                PRINTF("!!!vtbl[%d] hit %p:%p %s#%s -> %d\n", i, cx, px, cname, pname, pid);
	#endif // CC_DEBUG
                return MEMOFF(px);
            }
		}
		return 0;
	}

	__GURU__ GP
	scan_flist(guru_class *cx, GS pid)
	{
		GP prc = cx->flist;							// walk IREP linked-list
        int i = 0;
		while (prc) {
			// TODO: IREP should be added into guru_class->vtbl[]
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
};

__GURU__	ClassMgr::ClassMgr() : _impl(new Impl)	{}
__GURU__	ClassMgr::~ClassMgr() = default;

__GURU__ ClassMgr *ClassMgr::getInstance()
{
	static ClassMgr *_self = NULL;

	if (!_self) {
		_self = new ClassMgr();
	}
	return _self;
}
//================================================================
/* methods to add builtin (ROM) class/proc for GURU
 * it uses (const U8 *) for static string
 */
__GURU__ GP
ClassMgr::define_class(const U8 *name, GP super)
{
    GT super_cid = (GT)((super - guru_rom_get_class(GT_OBJ))/sizeof(guru_class) + GT_OBJ);
    
    return guru_rom_add_class(GT_EMPTY, (char*)name, super_cid, NULL, 0);
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
ClassMgr::class_add_meta(GR *r)			// lazy add metaclass to a class
{
	ASSERT(r->gt==GT_CLASS);

	guru_class *cx = GR_CLS(r);
	if (cx->meta) return cx->meta;

	// lazily create the metaclass
	U8 *name = (U8*)"_meta";
	GP mcls  = CLS_MGR->define_class(name, guru_rom_get_class(GT_OBJ));

	return cx->meta = mcls;					// self pointing =~ metaclass
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
ClassMgr::define_method(GP cls, const U8 *name, GP cfunc)
{
	return guru_define_method(cls, name, cfunc);
}

__GURU__ GP
ClassMgr::proc_by_id(GR *r, GS pid)
{
	GP cls = class_by_obj(r);
    GP prc = 0;

    while (cls) {
    	guru_class *cx = _CLS(cls);
    	prc = _impl->scan_vtbl(cx, pid);		// search for C-functions
    	if (prc) break;
        
#if CUDA_ENABLE_CDP
    	static S32 _proc_idx[32];

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

    	prc = _impl->scan_flist(cx, pid);		// TODO: combine flist into vtbl[]
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
  find class by object

  @param  vm
  @param  obj
  @return pointer to guru_class
*/
__GURU__ GP
ClassMgr::class_by_obj(GR *r)
{
#if CC_DEBUG
	PRINTF("!!!class_by_obj(%p) r->gt=%d, r->off=x%x: ", r, r->gt, r->off);
	const char *tname[] = {
			"???", "Nil", "False", "True", "Integer", "Float", "Symbol", "Sys",
            "", "Proc", "", "Array", "String", "Range", "Hash", "???"
	};
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
    	PRINTF(" CLS[x%04x]=%s:%p", cls, _RAW(cx->cid), cx);
#endif // CC_DEBUG
    	ret  = IS_BUILTIN(cx)
    		? cls
    		: (IS_SCLASS(r) ? scls : (IS_SELF(r) ? cls : scls));
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

__GURU__ GR
ClassMgr::inspect(GR *r, GR *obj)
{
	return send(r, obj, (U8*)"inspect", 0);
}

__GURU__ GR
ClassMgr::kind_of(GR *r)		// whether v1 is a kind of v0
{
	return send(r, r+1, (U8*)"kind_of?", 1, r);
}

__GURU__ GP
ClassMgr::class_by_id(GS cid)
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
ClassMgr::send(GR r[], GR *rcv, const U8 *method, U32 argc, ...)
{
    GR *regs = r + 2;	     		// allocate 2 for stack
    GS pid   = name2id(method);		// symbol lookup
    GP prc   = proc_by_id(r, pid);	// find method for receiver object

    ASSERT(prc);

    // create call stack.
    regs[0] = *ref_inc(rcv);		// create call stack, start with receiver object

    va_list ap;						// setup calling registers
    va_start(ap, argc);
    for (int i = 1; i <= argc+1; i++) {
        regs[i] = (i>argc) ? NIL : *va_arg(ap, GR *);
    }
    va_end(ap);

    _CALL(prc, regs, argc);	// call method, put return value in regs[0]

#if GURU_DEBUG
    GR *x = r;						// _wipe_stack
    for (int i=1; i<=argc+1; i++) {
    	*x++ = EMPTY;				// clean up the stack
    }
#endif
    return regs[0];
}

