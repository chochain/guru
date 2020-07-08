/*! @file
  @brief
  GURU Object classes i.e. Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <stdarg.h>

#include "guru.h"
#include "util.h"
#include "mmu.h"
#include "symbol.h"
#include "ostore.h"

#include "c_fixnum.h"
#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

#include "base.h"
#include "global.h"
#include "static.h"
#include "object.h"

#include "puts.h"
#include "inspect.h"

__GURU__ void
guru_obj_del(GR *r)
{
	ASSERT(r->gt==GT_OBJ);

	ostore_del(r);
}

//================================================================
__CFUNC__
obj_nop(GR r[], S32 ri)
{
	// do nothing
}

//================================================================
/*! (method) p
 */
__CFUNC__
obj_p(GR r[], S32 ri)
{
	guru_p(r, ri);
}

//================================================================
/*! (method) puts
 */
__CFUNC__
obj_puts(GR r[], S32 ri)
{
	guru_puts(r+1, ri);
	ref_dec(r);
	*r = EMPTY;
}

//================================================================
/*! (operator) !
 */
__CFUNC__
obj_not(GR r[], S32 ri)
{
    RETURN_FALSE();
}

//================================================================
/*! (operator) !=
 */
__CFUNC__
obj_neq(GR r[], S32 ri)
{
    S32 t = guru_cmp(r, r+1);
    RETURN_BOOL(t);
}

//================================================================
/*! (operator) <=>
 */
__CFUNC__
obj_cmp(GR r[], S32 ri)
{
    S32 t = guru_cmp(r, r+1);
    RETURN_INT(t);
}

//================================================================
/*! (operator) ===
 */
__CFUNC__
obj_eq3(GR r[], S32 ri)
{
    if (r->gt != GT_CLASS) {
    	RETURN_BOOL(guru_cmp(r, r+1)==0);
    }
    else {
#if GURU_CXX_CODEBASE
    	GR ret = ClassMgr::getInstance()->kind_of(r);
#else
    	GR ret = kind_of(r);
#endif // GURU_CXX_CODEBASE
    	RETURN_VAL(ret);
    }
}

//================================================================
/*! (method) class
 */
__CFUNC__
obj_class(GR r[], S32 ri)
{
	GP kls = GR_OBJ(r)->klass;
    GR ret { GT_CLASS, 0, 0, kls };

    RETURN_VAL(ret);
}

//================================================================
/*! (method) include
 */
__CFUNC__
obj_include(GR r[], S32 ri)
{
	ASSERT(r->gt==GT_CLASS && (r+1)->gt==GT_CLASS);

	guru_class_include(r->off, (r+1)->off);
}

//================================================================
/*! (method) extend
 */
__CFUNC__
obj_extend(GR r[], S32 ri)
{
	ASSERT(r->gt==GT_CLASS && (r+1)->gt==GT_CLASS);

#if GURU_CXX_CODEBASE
	ClassMgr::getInstance()->class_add_meta(r);			// lazily add metaclass if needed
#else
	guru_class *cx = GR_CLS(r);
	cx->kt   |= CLASS_EXTENDED;
	cx->klass = (r+1)->off;								// extend lexical scope
#endif // GURU_CXX_CODEBASE
}

//================================================================
/*! (method) instance variable getter
 */
__CFUNC__
obj_getiv(GR r[], S32 ri)
{
    RETURN_VAL(ostore_get(r, r->oid));				// attribute 'x'
}

//================================================================
/*! (method) instance variable setter
 */
__CFUNC__
obj_setiv(GR r[], S32 ri)
{
    GS oid = r->oid;							// attribute 'x='
    ostore_set(r, oid-1, r+1);					// attribute 'x' (hopefully at one entry before 'x=')
}


//================================================================
/*! append '=' to create name for attr_writer
 */
__GURU__ __INLINE__ U8*
_postfix_eq_sign(GR *buf, U8* s0)
{
	guru_str_clr(buf);
	guru_buf_add_cstr(buf, s0);
    guru_buf_add_cstr(buf, "=");

    return GR_RAW(buf);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__CFUNC__
obj_attr_reader(GR r[], S32 ri)
{
	ASSERT(r->gt==GT_CLASS);
	GP cls = find_class_by_obj(r);
	GR *s  = r+1;
    for (int i = 0; i < ri; i++, s++) {
        ASSERT(s->gt==GT_SYM);

        U8 *name = _RAW(s->i);								// SYM2RAW
        ASSERT(guru_define_method(cls, name, MEMOFF(obj_getiv)));
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
#define ATTR_BUFSIZE	255
__CFUNC__
obj_attr_accessor(GR r[], S32 ri)
{
	ASSERT(r->gt==GT_CLASS);

	GP cls = find_class_by_obj(r);
#if CC_DEBUG
	guru_class *cx = _CLS(cls);
    printf("%p:%s, sc=%d self=%d #attr_accessor\n", cx, _RAW(cx->cid), IS_SCLASS(r), IS_TCLASS(r));
#endif // CC_DEBUG
	GR *s  = r+1;
	GR buf = guru_str_buf(GURU_STRBUF_SIZE);
    for (int i=0; i < ri; i++, s++) {
        ASSERT(s->gt==GT_SYM);
        U8 *a0 = _RAW(s->i);										// reader
        U8 *a1 = _postfix_eq_sign(&buf, a0);						// writer
        ASSERT(guru_define_method(cls, a0, MEMOFF(obj_getiv)));
        ASSERT(guru_define_method(cls, a1, MEMOFF(obj_setiv)));
    }
    guru_str_del(&buf);
}

//================================================================
/*! (method) is_a, kind_of
 */
__CFUNC__
obj_kind_of(GR r[], S32 ri)
{
	ASSERT(r->gt==GT_OBJ);
    if ((r+1)->gt != GT_CLASS) {
        RETURN_BOOL(0);
    }

    GP kls = GR_OBJ(r)->klass;
    while (kls) {
        if (kls==(r+1)->off) break;
        kls = _CLS(kls)->super;
    }
}

//================================================================
/*! lambda function
 */
__CFUNC__
obj_lambda(GR r[], S32 ri)
{
	ASSERT(r->gt==GT_CLASS && (r+1)->gt==GT_PROC);		// ensure it is a proc

	guru_proc *px = GR_PRC(r+1);						// mark it as a lambda
	px->kt |= PROC_LAMBDA;

	U32	n    = px->n = ri+3;
	GR  *rf  = guru_gr_alloc(n);
	px->regs = MEMOFF(rf);

	GR  *r0 = r - n;
	for (int i=0; i<n; *rf++=*r0++, i++);

    *r = *(r+1);
	(r+1)->gt = GT_EMPTY;
}

//================================================================
/*! set constant
 */
__CFUNC__
obj_const_set(GR r[], S32 ri)
{
	GP cls = r->off;
	GS xid = (r+1)->off;
	(r+2)->acl &= ~ACL_HAS_REF;			// make it constant

	const_set(cls, xid, r+2);
}

//================================================================
/*! get constant
 */
__CFUNC__
obj_const_get(GR r[], S32 ri)
{
	GP cls = r->off;
	GS xid = (r+1)->off;
	GR ret = *const_get(cls, xid);

	RETURN_VAL(ret);
}

//================================================================
/*! pseudo random number generator
 */
__CFUNC__
obj_rand(GR r[], S32 ri)
{
	static int _seed = 123456789;

	_seed = 1103515245 * _seed + 12345;
	GR ret { GT_FLOAT, 0, 0, 0x3f000000 | (_seed & 0x7fffff) };

	if (ri && (r+1)->gt==GT_INT) {
		ret.gt = GT_INT;
		ret.i  = _seed % (r+1)->i;
	}
	RETURN_VAL(ret);
}

//=====================================================================
//! deprecated, use inspect#gr_to_s instead
__CFUNC__
obj_to_s(GR r[], S32 ri)
{
	ASSERT(1==0);				// handled in ucode
}

__GURU__ __const__ Vfunc obj_mtbl[] = {
	{ "puts",          	obj_puts 		},				// handled by state#_method_missing
	{ "initialize", 	obj_nop 		},
    { "private",		obj_nop			},				// do nothing now
    // comparators
	{ "!",				obj_not 		},
	{ "!=",          	obj_neq 		},
	{ "<=>",           	obj_cmp 		},
	{ "===",           	obj_eq3 		},
	// class ops
	{ "class",         	obj_class		},
	{ "include",		obj_include     },
	{ "extend",			obj_extend		},
//    	{ "new",           	obj_new 		},			// handled by state#_method_missing
//      { "raise",			obj_raise		},			// handled by state#_method_missing
	{ "attr_reader",   	obj_attr_reader 	},
	{ "attr_accessor", 	obj_attr_accessor	},
	{ "is_a?",         	obj_kind_of		},
    { "kind_of?",      	obj_kind_of		},
    { "lambda",			obj_lambda		},
    // misc functions
	{ "const_set",		obj_const_set	},
	{ "const_get",      obj_const_get   },
    { "rand",           obj_rand        },
    // print functions (puts is pulled to top)
    { "print",         	obj_puts		},
	{ "p", 				obj_p    		},
    // the following functions depends on string, implemented in inspect.cu
    { "to_s",          	gr_to_s  		},
    { "inspect",       	gr_to_s  		},
    { "sprintf",		gr_sprintf		},
    { "printf",			gr_printf		}
 };

//================================================================
// ProcClass
//================================================================

__GURU__ __const__ Vfunc prc_mtbl[] = {
//    	{ "call", 	prc_call	},		// handled  by ucode#uc_send
	{ "to_s", 	gr_to_s		},
	{ "inspect",gr_to_s		}
};

//================================================================
// Nil class
__CFUNC__
nil_false_not(GR r[], S32 ri)
{
    r->gt = GT_TRUE;
}

__CFUNC__
nil_inspect(GR r[], S32 ri)
{
    RETURN_VAL(guru_str_new("nil"));
}

//================================================================
/*! Nil class
 */
__GURU__ __const__ Vfunc nil_mtbl[] = {
	{ "!", 			nil_false_not	},
	{ "inspect", 	nil_inspect		},
	{ "to_s", 		gr_to_s			}
};

//================================================================
/*! False class
 */
__GURU__ __const__ Vfunc false_mtbl[] = {
	{ "!", 		nil_false_not	},
	{ "to_s",    gr_to_s		},
	{ "inspect", gr_to_s		}
};

__GURU__ __const__ Vfunc true_mtbl[] = {
	{ "to_s", 		gr_to_s 	},
	{ "inspect", 	gr_to_s		}
};

//================================================================
/*! Symbol class
 */
__CFUNC__ sym_nop(GR r[], S32 ri) {}

//================================================================
// initialize
__GURU__ __const__ Vfunc sym_mtbl[] = {
	{ "id2name", 	gr_to_s		},
	{ "to_sym",     sym_nop		},
	{ "to_s", 		sym_to_s	}, 	// no leading ':'
	{ "inspect", 	gr_to_s		}
};

//================================================================
/*! StandardError class
 */
__CFUNC__
err_new(GR r[], S32 ri)
{
	GR *r1 = r + 1;
	GS sid = r1->off;
	if (r1->gt==GT_STR) {
		U8 *s  = GR_RAW(r1);
		sid = guru_rom_add_sym((char*)s);
	}

	GR x { GT_ERROR, 0, 0, sid };
	ref_dec(r);
	*r     = x;
	*(r+1) = EMPTY;
}

__GURU__ __const__ Vfunc err_mtbl[] = {
	{ "new",        err_new     },
	{ "to_s", 		gr_to_s 	},
	{ "inspect", 	err_to_s	}
};

//================================================================
/*! Create a inner class
 */
__CFUNC__
cls_new(GR r[], S32 ri)
{
	ASSERT((r-2)->gt==GT_CLASS && r->gt==GT_CLASS);

	GP super = (r-2)->off;
	GS xid   = (r-1)->off;
	GP cls   = guru_define_class(NULL, xid, super);			// fill the ROM class storage

	_CLS(cls)->kt |= CLASS_SUBCLASS;

	r->off = cls;
}

//================================================================
/*! System class (guru only, i.e. non-Ruby)
 */
__CFUNC__
cls_mstat(GR r[], S32 ri)
{
	GR  si; { si.gt=GT_INT; si.acl=0; }
	GR  ret = guru_array_new(8);
	U32 s[8];
	guru_mmu_stat((guru_mstat*)s);
	for (int i=0; i<8; i++) {
		si.i = s[i];
		guru_array_push(&ret, &si);
	}
	*r = ret;
}

__GURU__ __const__ Vfunc cls_mtbl[] = {
    { "new",        cls_new     },
	{ "mstat", 		cls_mstat	}
};

//================================================================
// initialize
// TODO: move into ROM
__GURU__ void
_install_all_class(void)
{
    guru_rom_add_class(GT_OBJ,	"Object", 		(GT)0, 		obj_mtbl, 	VFSZ(obj_mtbl));		// xa

    guru_rom_add_class(GT_NIL, 	"NilClass", 	GT_OBJ, 	nil_mtbl,  	VFSZ(nil_mtbl));		// x1
    guru_rom_add_class(GT_FALSE,"FalseClass", 	GT_OBJ, 	false_mtbl,	VFSZ(false_mtbl));		// x2
    guru_rom_add_class(GT_TRUE, "TrueClass",  	GT_OBJ, 	true_mtbl, 	VFSZ(true_mtbl));		// x3
    guru_rom_add_class(GT_SYM,  "Symbol", 		GT_OBJ, 	sym_mtbl, 	VFSZ(sym_mtbl));		// x6
    guru_rom_add_class(GT_ERROR,"StandardError",GT_OBJ,     err_mtbl,   VFSZ(err_mtbl));		// x7
    guru_rom_add_class(GT_CLASS,"Class", 		GT_OBJ, 	cls_mtbl,  	VFSZ(cls_mtbl));		// x8
    guru_rom_add_class(GT_PROC, "Proc",     	GT_OBJ, 	prc_mtbl,  	VFSZ(prc_mtbl));		// x9

    guru_register_func(GT_OBJ, NULL, guru_obj_del, NULL);

    guru_init_class_int();			// c_fixnum.cu  x4
    guru_init_class_float();		// c_fixnum.cu	x5

    guru_init_class_range();		// c_range.cu	xb
    guru_init_class_string();		// c_string.cu	xc
    guru_init_class_array();		// c_array.cu	xd
    guru_init_class_hash();			// c_hash.cu	xe

#if GURU_USE_MATH
    guru_init_class_math();
#endif // GURU_USE_MATH
}

__GURU__ void
_setup_for_module(void)
{
    // duplicate module as class
	GS mid = guru_rom_add_sym("Module");
	GP obj = guru_rom_get_class(GT_OBJ);
	GP cls = guru_rom_get_class(GT_CLASS);

	GR *r  = (GR*)guru_alloc(sizeof(GR));		// TODO: utilize GT_ITER class block (which is not used)
	r->gt  = GT_CLASS;
	r->acl = 0;
	r->off = cls;

	const_set(obj, mid, r);
}

__GPU__ void
guru_core_init(void)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	guru_rom_init();
	_install_all_class();						// TODO: load image into context memory
	_setup_for_module();

#if CC_DEBUG
	guru_rom *rom = &guru_device_rom;
    	PRINTF("ROM createdd with ncls=%d, nprc=%d, nsym=%d, nstr=%d\n", rom->ncls, rom->nprc, rom->nsym, rom->nstr);
#endif // CC_DEBUG

    guru_rom_burn();
}
