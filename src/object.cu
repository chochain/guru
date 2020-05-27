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
obj_nop(GR r[], U32 ri)
{
	// do nothing
}

//================================================================
/*! (method) p
 */
__CFUNC__
obj_p(GR r[], U32 ri)
{
	guru_p(r, ri);
}

//================================================================
/*! (method) puts
 */
__CFUNC__
obj_puts(GR r[], U32 ri)
{
	guru_puts(r+1, ri);
}

//================================================================
/*! (method) print
 */
__CFUNC__
obj_print(GR r[], U32 ri)
{
	guru_puts(r+1, ri);
}

//================================================================
/*! (operator) !
 */
__CFUNC__
obj_not(GR r[], U32 ri)
{
    RETURN_FALSE();
}

//================================================================
/*! (operator) !=
 */
__CFUNC__
obj_neq(GR r[], U32 ri)
{
    S32 t = guru_cmp(r, r+1);
    RETURN_BOOL(t);
}

//================================================================
/*! (operator) <=>
 */
__CFUNC__
obj_cmp(GR r[], U32 ri)
{
    S32 t = guru_cmp(r, r+1);
    RETURN_INT(t);
}

//================================================================
/*! (operator) ===
 */
__CFUNC__
obj_eq3(GR r[], U32 ri)
{
    if (r->gt != GT_CLASS) {
    	RETURN_BOOL(guru_cmp(r, r+1)==0);
    }
    else {
    	GR ret = kind_of(r);
    	RETURN_VAL(ret);
    }
}

//================================================================
/*! (method) class
 */
__CFUNC__
obj_class(GR r[], U32 ri)
{
    GR ret { .gt=GT_CLASS, .acl=0, .oid=0, { .off=class_by_obj(r) }};

    RETURN_VAL(ret);
}

__GURU__ void
_extend(GP cls, GP mod)
{
	guru_class *cx  = _CLS(cls);
	guru_class *dup = (guru_class*)guru_alloc(sizeof(guru_class));
	MEMCPY(dup, _CLS(mod), sizeof(guru_class));	// TODO: deep copy so vtbl can be modified later

	dup->super = cx->super;							// put module as the super-class
	cx->super  = MEMOFF(dup);
}

//================================================================
/*! (method) include
 */
__CFUNC__
obj_include(GR r[], U32 ri)
{
	ASSERT(r->gt==GT_CLASS && (r+1)->gt==GT_CLASS);
	_extend(r->off, (r+1)->off);
}

//================================================================
/*! (method) extend
 */
__CFUNC__
obj_extend(GR r[], U32 ri)
{
	ASSERT(r->gt==GT_CLASS && (r+1)->gt==GT_CLASS);

	guru_class_add_meta(r);						// lazily add metaclass if needed
	_extend(GR_CLS(r)->meta, (r+1)->off);		// add to class methods
}

//================================================================
/*! (method) instance variable getter
 */
__CFUNC__
obj_getiv(GR r[], U32 ri)
{
    RETURN_VAL(ostore_get(r, r->oid));			// attribute 'x'
}

//================================================================
/*! (method) instance variable setter
 */
__CFUNC__
obj_setiv(GR r[], U32 ri)
{
    GS oid = r->oid;							// attribute 'x='
    ostore_set(r, oid-1, r+1);					// attribute 'x' (hopefully at one entry before 'x=')
}


//================================================================
/*! append '=' to create name for attr_writer
 */
__GURU__ U8*
_name_w_eq_sign(GR *buf, U8 *s0)
{
    guru_buf_add_cstr(buf, s0);
    guru_buf_add_cstr(buf, "=");

    U32 sid = guru_rom_add_sym((char*)GR_RAW(buf));		// create the symbol

    return _RAW(sid);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__CFUNC__
obj_attr_reader(GR r[], U32 ri)
{
	ASSERT(r->gt==GT_CLASS);
	GP cls = r->off;									// fetch class offset

	GR *s = r+1;
    for (int i = 0; i < ri; i++, s++) {
        ASSERT(s->gt==GT_SYM);

        U8 *name = _RAW(s->i);
        ASSERT(guru_define_method(cls, name, MEMOFF(obj_getiv)));
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
__CFUNC__
obj_attr_accessor(GR r[], U32 ri)
{
	ASSERT(r->gt==GT_CLASS);
	GP cls = IS_SCLASS(r) ? GR_CLS(r)->meta : r->off;	// fetch class offset
#if CC_DEBUG
	guru_class *cx = _CLS(cls);
    printf("%p:%s, sc=%d self=%d #attr_accessor\n", cx, MEMPTR(cx->cname), IS_SCLASS(r), IS_SELF(r));
#endif // CC_DEBUG
    GR buf = guru_str_buf(80);
	GR *s  = r+1;
    for (int i=0; i < ri; i++, s++) {
        ASSERT(s->gt==GT_SYM);
        U8 *a0  = _RAW(s->i);							// reader
        U8 *a1  = _name_w_eq_sign(&buf, a0);			// writer

        ASSERT(guru_define_method(cls, a0, MEMOFF(obj_getiv)));
        ASSERT(guru_define_method(cls, a1, MEMOFF(obj_setiv)));

        guru_str_clr(&buf);
    }
    guru_str_del(&buf);
}

//================================================================
/*! (method) is_a, kind_of
 */
__CFUNC__
obj_kind_of(GR r[], U32 ri)
{
	ASSERT(r->gt==GT_OBJ);
    if ((r+1)->gt != GT_CLASS) {
        RETURN_BOOL(0);
    }
    GP cls = class_by_obj(r);
    while (cls) {
        if (cls==(r+1)->off) break;
        cls = _CLS(cls)->super;
    }
}

//================================================================
/*! lambda function
 */
__CFUNC__
obj_lambda(GR r[], U32 ri)
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

//=====================================================================
//! deprecated, use inspect#gr_to_s instead
__CFUNC__
obj_to_s(GR r[], U32 ri)
{
	ASSERT(1==0);				// handled in ucode
}

__GURU__ __const__ Vfunc obj_vtbl[] = {
	{ "puts",          	obj_puts 		},
	{ "initialize", 	obj_nop 		},
    { "private",		obj_nop			},			// do nothing now
	{ "!",				obj_not 		},
	{ "!=",          	obj_neq 		},
	{ "<=>",           	obj_cmp 		},
	{ "===",           	obj_eq3 		},
	{ "class",         	obj_class		},
	{ "include",		obj_include     },
	{ "extend",			obj_extend		},
//    	{ "new",           	obj_new 		},			// handled by state#_method_missing
//      { "raise",			obj_raise		},			// handled by state#_method_missing
	{ "attr_reader",   	obj_attr_reader 	},
	{ "attr_accessor", 	obj_attr_accessor	},
    { "lambda",			obj_lambda		},
	{ "is_a?",         	obj_kind_of		},
    { "kind_of?",      	obj_kind_of		},
	{ "print",         	obj_print		},
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

__GURU__ __const__ Vfunc prc_vtbl[] = {
//    	{ "call", 	prc_call	},		// handled  by ucode#uc_send
	{ "to_s", 	gr_to_s		},
	{ "inspect",gr_to_s		}
};

//================================================================
// Nil class
__CFUNC__
nil_false_not(GR r[], U32 ri)
{
    r->gt = GT_TRUE;
}

__CFUNC__
nil_inspect(GR r[], U32 ri)
{
    RETURN_VAL(guru_str_new("nil"));
}

//================================================================
/*! Nil class
 */
__GURU__ __const__ Vfunc nil_vtbl[] = {
	{ "!", 			nil_false_not	},
	{ "inspect", 	nil_inspect		},
	{ "to_s", 		gr_to_s			}
};

//================================================================
/*! False class
 */
__GURU__ __const__ Vfunc false_vtbl[] = {
	{ "!", 		nil_false_not	},
	{ "to_s",    gr_to_s		},
	{ "inspect", gr_to_s		}
};

__GURU__ __const__ Vfunc true_vtbl[] = {
	{ "to_s", 		gr_to_s 	},
	{ "inspect", 	gr_to_s		}
};

//================================================================
/*! Symbol class
 */
__CFUNC__ sym_nop(GR r[], U32 ri) {}

__CFUNC__
sym_to_s(GR r[], U32 ri)
{
	U8 *s  = _RAW(r->i);
	GR ret = guru_str_new(s);
    RETURN_VAL(ret);
}

//================================================================
// initialize
__GURU__ __const__ Vfunc sym_vtbl[] = {
	{ "id2name", 	gr_to_s		},
	{ "to_sym",     sym_nop		},
	{ "to_s", 		sym_to_s	}, 	// no leading ':'
	{ "inspect", 	gr_to_s		}
};

#if GURU_DEBUG
//================================================================
/*! System class (guru only, i.e. non-Ruby)
 */
__CFUNC__
sys_mstat(GR r[], U32 ri)
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

__GURU__ __const__ Vfunc sys_vtbl[] = {
	{ "mstat", 		sys_mstat	}
};

#endif // GURU_DEBUG
//================================================================
// initialize
// TODO: move into ROM
__GURU__ void
_install_all_class(void)
{
    guru_rom_add_class(GT_OBJ,	"Object", 		GT_EMPTY, 	obj_vtbl, 	VFSZ(obj_vtbl));

    guru_rom_add_class(GT_NIL, 	"NilClass", 	GT_OBJ, 	nil_vtbl,  	VFSZ(nil_vtbl));
    guru_rom_add_class(GT_FALSE,"FalseClass", 	GT_OBJ, 	false_vtbl,	VFSZ(false_vtbl));
    guru_rom_add_class(GT_TRUE, "TrueClass",  	GT_OBJ, 	true_vtbl, 	VFSZ(true_vtbl));
    guru_rom_add_class(GT_SYM,  "Symbol", 		GT_OBJ, 	sym_vtbl, 	VFSZ(sym_vtbl));
    guru_rom_add_class(GT_PROC, "Proc",     	GT_OBJ, 	prc_vtbl,  	VFSZ(prc_vtbl));
#if GURU_DEBUG
    guru_rom_add_class(GT_SYS, 	"Sys", 			GT_OBJ, 	sys_vtbl,  	VFSZ(sys_vtbl));
#endif

    guru_register_func(GT_OBJ, NULL, guru_obj_del, NULL);

    guru_init_class_int();			// c_fixnum.cu
    guru_init_class_float();		// c_fixnum.cu

    guru_init_class_range();		// c_range.cu
    guru_init_class_string();		// c_string.cu
    guru_init_class_array();		// c_array.cu
    guru_init_class_hash();			// c_hash.cu

#if GURU_USE_MATH
    guru_init_class_math();
#endif // GURU_USE_MATH
}

__GPU__ void
guru_core_init(void)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	guru_rom_init();
	_install_all_class();		// TODO: load image into context memory

#if CC_DEBUG
	guru_rom *rom = &guru_device_rom;
    PRINTF("ROM createdd with ncls=%d, nprc=%d, nsym=%d, nstr=%d\n", rom->ncls, rom->nprc, rom->nsym, rom->nstr);
#endif // CC_DEBUG
	guru_rom_burn();
}
