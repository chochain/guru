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
#include "object.h"

#include "puts.h"
#include "inspect.h"

__GURU__ void
guru_obj_del(GV *v)
{
	ASSERT(v->gt==GT_OBJ);

	ostore_del(v);
}

//================================================================
__CFUNC__
obj_nop(GV v[], U32 vi)
{
	// do nothing
}

//================================================================
/*! (method) p
 */
__CFUNC__
obj_p(GV v[], U32 vi)
{
	guru_p(v, vi);
}

//================================================================
/*! (method) puts
 */
__CFUNC__
obj_puts(GV v[], U32 vi)
{
	guru_puts(v+1, vi);
}

//================================================================
/*! (method) print
 */
__CFUNC__
obj_print(GV v[], U32 vi)
{
	guru_puts(v+1, vi);
}

//================================================================
/*! (operator) !
 */
__CFUNC__
obj_not(GV v[], U32 vi)
{
    RETURN_FALSE();
}

//================================================================
/*! (operator) !=
 */
__CFUNC__
obj_neq(GV v[], U32 vi)
{
    S32 t = guru_cmp(v, v+1);
    RETURN_BOOL(t);
}

//================================================================
/*! (operator) <=>
 */
__CFUNC__
obj_cmp(GV v[], U32 vi)
{
    S32 t = guru_cmp(v, v+1);
    RETURN_INT(t);
}

//================================================================
/*! (operator) ===
 */
__CFUNC__
obj_eq3(GV v[], U32 vi)
{
    if (v->gt != GT_CLASS) {
    	RETURN_BOOL(guru_cmp(v, v+1)==0);
    }
    else {
    	GV ret = kind_of(v);
    	RETURN_VAL(ret);
    }
}

//================================================================
/*! (method) class
 */
__CFUNC__
obj_class(GV v[], U32 vi)
{
    GV ret;  { ret.gt = GT_CLASS; ret.acl=0; }
    ret.cls = class_by_obj(v);

    RETURN_VAL(ret);
}

__GURU__ void
_extend(guru_class *cls, guru_class *mod)
{
	guru_class *dup = (guru_class*)guru_alloc(sizeof(guru_class));
	MEMCPY(dup, mod, sizeof(guru_class));		// TODO: deep copy so vtbl can be modified later

	dup->super = cls->super;					// put module as the super-class
	cls->super = dup;
}

//================================================================
/*! (method) include
 */
__CFUNC__
obj_include(GV v[], U32 vi)
{
	ASSERT(v->gt==GT_CLASS && (v+1)->gt==GT_CLASS);
	_extend(v->cls, (v+1)->cls);
}

//================================================================
/*! (method) extend
 */
__CFUNC__
obj_extend(GV v[], U32 vi)
{
	ASSERT(v->gt==GT_CLASS && v[1].gt==GT_CLASS);

	guru_class_add_meta(v);						// lazily add metaclass if needed
	_extend(v->cls->meta, (v+1)->cls);			// add to class methods
}

//================================================================
/*! (method) instance variable getter
 */
__CFUNC__
obj_getiv(GV v[], U32 vi)
{
    RETURN_VAL(ostore_get(v, v->oid));			// attribute 'x'
}

//================================================================
/*! (method) instance variable setter
 */
__CFUNC__
obj_setiv(GV v[], U32 vi)
{
    GU oid = v->oid;							// attribute 'x='
    ostore_set(v, oid-1, v+1);					// attribute 'x' (hopefully at one entry before 'x=')
}


//================================================================
/*! append '=' to create name for attr_writer
 */
__GURU__ U8 *
_name_w_eq_sign(GV *buf, U8 *s0)
{
    guru_buf_add_cstr(buf, s0);
    guru_buf_add_cstr(buf, "=");

    U32 sid = create_sym((U8*)buf->str->raw);			// create the symbol

    return id2name(sid);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__CFUNC__
obj_attr_reader(GV v[], U32 vi)
{
	ASSERT(v->gt==GT_CLASS);
	guru_class *cls = v->cls;							// fetch class

	GV *s = v+1;
    for (U32 i = 0; i < vi; i++, s++) {
        ASSERT(s->gt==GT_SYM);

        U8 *name = id2name(s->i);						// reader only
        guru_define_method(cls, name, obj_getiv);
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
__CFUNC__
obj_attr_accessor(GV v[], U32 vi)
{
	ASSERT(v->gt==GT_CLASS);
	guru_class *cls = IS_SCLASS(v) ? v->cls->meta : v->cls;		// fetch class
#if CC_DEBUG
    printf("%p:%s, sc=%d self=%d #attr_accessor\n", cls, cls->name, IS_SCLASS(v), IS_SELF(v));
#endif // CC_DEBUG
    GV buf = guru_str_buf(80);
	GV *s  = v+1;
    for (U32 i=0; i < vi; i++, s++) {
        ASSERT(s->gt==GT_SYM);
        U8 *a0  = id2name(s->i);						// reader
        U8 *a1  = _name_w_eq_sign(&buf, a0);			// writer

        guru_define_method(cls, a0, obj_getiv);
        guru_define_method(cls, a1, obj_setiv);

        guru_str_clr(&buf);
    }
    guru_str_del(&buf);
}

//================================================================
/*! (method) is_a, kind_of
 */
__CFUNC__
obj_kind_of(GV v[], U32 vi)
{
	ASSERT(v->gt==GT_OBJ);
    if ((v+1)->gt != GT_CLASS) {
        RETURN_BOOL(0);
    }
    const guru_class *cls = class_by_obj(v);

    while (cls) {
        if (cls == (v+1)->cls) break;
        cls = cls->super;
    }
}

//================================================================
/*! lambda function
 */
__CFUNC__
obj_lambda(GV v[], U32 vi)
{
	ASSERT(v->gt==GT_CLASS && (v+1)->gt==GT_PROC);		// ensure it is a proc

	guru_proc *prc = (v+1)->proc;						// mark it as a lambda
	prc->kt |= PROC_LAMBDA;

	U32	n   = prc->n 	= vi+3;
	GV  *r  = prc->regs = guru_gv_alloc(n);
	GV  *r0 = v - n;
	for (U32 i=0; i<n; *r++=*r0++, i++);

    *v = *(v+1);
	(v+1)->gt = GT_EMPTY;
}

//=====================================================================
//! deprecated, use inspect#gv_to_s instead
__CFUNC__
obj_to_s(GV v[], U32 vi)
{
	ASSERT(1==0);				// handled in ucode
}

__GURU__ __const__ Vfunc obj_vtbl[] = {
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
    { "puts",          	obj_puts 		},
    { "print",         	obj_print		},
    { "p", 				obj_p    		},

    // the following functions depends on string, implemented in inspect.cu
    { "to_s",          	gv_to_s  		},
    { "inspect",       	gv_to_s  		},
    { "sprintf",		gv_sprintf		},
    { "printf",			gv_printf		}
 };

//================================================================
// ProcClass
//================================================================

__GURU__ __const__ Vfunc prc_vtbl[] = {
//    	{ "call", 	prc_call	},		// handled  by ucode#uc_send
	{ "to_s", 	gv_to_s		},
	{ "inspect",gv_to_s		}
};

//================================================================
// Nil class
__CFUNC__
nil_false_not(GV v[], U32 vi)
{
    v->gt = GT_TRUE;
}

__CFUNC__
nil_inspect(GV v[], U32 vi)
{
    RETURN_VAL(guru_str_new("nil"));
}

//================================================================
/*! Nil class
 */
__GURU__ __const__ Vfunc nil_vtbl[] = {
	{ "!", 			nil_false_not	},
	{ "inspect", 	nil_inspect		},
	{ "to_s", 		gv_to_s			}
};

//================================================================
/*! False class
 */
__GURU__ __const__ Vfunc false_vtbl[] = {
	{ "!", 		nil_false_not	},
	{ "to_s",    gv_to_s		},
	{ "inspect", gv_to_s		}
};

__GURU__ __const__ Vfunc true_vtbl[] = {
	{ "to_s", 		gv_to_s 	},
	{ "inspect", 	gv_to_s		}
};

//================================================================
/*! Symbol class
 */
__CFUNC__ sym_nop(GV v[], U32 vi) {}

__CFUNC__
sym_to_s(GV v[], U32 vi)
{
	GV ret = guru_str_new(id2name(v->i));
    RETURN_VAL(ret);
}

//================================================================
// initialize
__GURU__ __const__ Vfunc sym_vtbl[] = {
	{ "id2name", 	gv_to_s		},
	{ "to_sym",     sym_nop		},
	{ "to_s", 		sym_to_s	}, 	// no leading ':'
	{ "inspect", 	gv_to_s		}
};

#if GURU_DEBUG
//================================================================
/*! System class (guru only, i.e. non-Ruby)
 */
__CFUNC__
sys_mstat(GV v[], U32 vi)
{
	GV  si; { si.gt=GT_INT; si.acl=0; }
	GV  ret = guru_array_new(8);
	U32 s[8];
	guru_mmu_stat((guru_mstat*)s);
	for (U32 i=0; i<8; i++) {
		si.i = s[i];
		guru_array_push(&ret, &si);
	}
	*v = ret;
}

__GURU__ __const__ Vfunc sys_vtbl[] = {
	{ "mstat", 		sys_mstat	}
};

#endif // GURU_DEBUG
//================================================================
// initialize
// TODO: move into ROM
//
typedef struct Vclass {
	GT					cidx;
	U8 					*cname;
	guru_class 			*super;
	const struct Vfunc	*vtbl;
	U32					nfunc;
} guru_class_tbl;

//#define CLASS_DEF(cidx, cname, vtbl) { cidx, (U8*)cname, vtbl==obj_vtbl ? NULL : guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc) }
#define CLASS_DEF(cname, vtbl)	{ guru_add_class(cname, guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)) }

__GURU__ void
_init_all_class(void)
{
    guru_rom_set_class(GT_OBJ,	"Object", 		GT_EMPTY, 	obj_vtbl, 	VFSZ(obj_vtbl));

    guru_rom_set_class(GT_NIL, 	"NilClass", 	GT_OBJ, 	nil_vtbl,  	VFSZ(nil_vtbl));
    guru_rom_set_class(GT_FALSE,"FalseClass", 	GT_OBJ, 	false_vtbl,	VFSZ(false_vtbl));
    guru_rom_set_class(GT_TRUE, "TrueClass",  	GT_OBJ, 	true_vtbl, 	VFSZ(true_vtbl));
    guru_rom_set_class(GT_SYM,  "Symbol", 		GT_OBJ, 	sym_vtbl, 	VFSZ(sym_vtbl));
    guru_rom_set_class(GT_PROC, "Proc",     	GT_OBJ, 	prc_vtbl,  	VFSZ(prc_vtbl));
#if GURU_DEBUG
    guru_rom_set_class(GT_SYS, 	"Sys", 			GT_OBJ, 	sys_vtbl,  	VFSZ(sys_vtbl));
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

	//
	// TODO: load image into context memory
	//
	_init_all_class();
}
