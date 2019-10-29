/*! @file
  @brief
  GURU Object classes i.e. Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <assert.h>
#include <stdarg.h>

#include "guru.h"
#include "class.h"
#include "value.h"
#include "mmu.h"
#include "static.h"
#include "symbol.h"
#include "state.h"

#include "object.h"
#include "ostore.h"

#include "c_fixnum.h"
#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

#include "inspect.h"

//================================================================
/*! (BETA) Call any method of the object, but written by C.

  @param  vm		pointer to vm.
  @param  v		see bellow example.
  @param  reg_ofs	see bellow example.
  @param  recv		pointer to receiver.
  @param  name		method name.
  @param  argc		num of params.

  @example
  void int_to_s(GV v[], U32 vi)
  {
  	  GV *rcv = &v[1];
  	  GV ret  = _send(v, rcv, "to_s", argc);
  	  RETURN_VAL(ret);
  }
*/
__GURU__ GV
_send(GV v[], GV *rcv, const U8 *method, U32 argc, ...)
{
    GV *regs = v + 2;	     		// allocate 2 for stack
    GS sid   = name2id(method);

    guru_proc  *m = proc_by_sid(v, sid);		// find method for receiver object
    assert(m);

    // create call stack.
    regs[0] = *ref_inc(rcv);		// create call stack, start with receiver object

    va_list ap;						// setup calling registers
    va_start(ap, argc);
    for (U32 i = 1; i <= argc+1; i++) {
        regs[i] = (i>argc) ? NIL() : *va_arg(ap, GV *);
    }
    va_end(ap);

    m->func(regs, argc);			// call method, put return value in regs[0]

#if GURU_DEBUG
    GV *r = v;						// _wipe_stack
    for (U32 i=1; i<=argc+1; i++) {
    	*r++ = EMPTY();				// clean up the stack
    }
#endif
    return regs[0];
}

__GURU__ void
guru_class_add_meta(GV *v)			// lazy add metaclass to a class
{
	assert(v->gt==GT_CLASS);

	if (v->cls->cls!=NULL) return;

	// lazily create the metaclass
	const U8	*name = (U8*)"_meta";
	guru_class 	*cls  = guru_define_class(name, guru_class_object);
	cls->meta |= CLASS_META;		// mark it as a metaclass
	v->cls->cls = cls;				// self pointing =~ metaclass
}

__GURU__ GV
guru_inspect(GV v[], GV *obj)
{
	return _send(v, obj, (U8*)"inspect", 0);
}

__GURU__ GV
guru_kind_of(GV v[])		// whether v1 is a kind of v0
{
	return _send(v, v+1, (U8*)"kind_of?", 1, v);
}

__GURU__ void
guru_obj_del(GV *v)
{
	assert(v->gt==GT_OBJ);

	ostore_del(v);
}

//================================================================
__CFUNC__
obj_nop(GV v[], U32 vi)
{
	// do nothing
}

#if GURU_DEBUG
//================================================================
/*! (method) p
 */
__CFUNC__
obj_p(GV v[], U32 vi)
{
	guru_p(v, vi);
}
#endif

//================================================================
/*! (method) puts
 */
__CFUNC__
obj_puts(GV v[], U32 vi)
{
	guru_puts(v, vi);
}

//================================================================
/*! (method) print
 */
__CFUNC__
obj_print(GV v[], U32 vi)
{
	guru_puts(v, vi);
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
    	GV ret = guru_kind_of(v);
    	RETURN_VAL(ret);
    }
}

//================================================================
/*! (method) class
 */
__CFUNC__
obj_class(GV v[], U32 vi)
{
	assert(v->gt==GT_OBJ);

    GV ret;  { ret.gt = GT_CLASS; ret.acl=0; }
    ret.cls = class_by_obj(v);

    RETURN_VAL(ret);
}

__GURU__ void
_extend(guru_class *cls, guru_class *mod)
{
	guru_class *dup = (guru_class*)guru_alloc(sizeof(guru_class));
	MEMCPY(dup, mod, sizeof(guru_class));		// deep copy so vtbl can be modified later

	dup->super = cls->super;					// put module as the super-class
	cls->super = dup;
}

//================================================================
/*! (method) include
 */
__CFUNC__
obj_include(GV v[], U32 vi)
{
	assert(v->gt==GT_CLASS && (v+1)->gt==GT_CLASS);
	_extend(v->cls, (v+1)->cls);
}

//================================================================
/*! (method) extend
 */
__CFUNC__
obj_extend(GV v[], U32 vi)
{
	assert(v->gt==GT_CLASS && v[1].gt==GT_CLASS);

	guru_class_add_meta(v);						// lazily add metaclass if needed
	_extend(v->cls->cls, (v+1)->cls);			// add to class methods
}

//================================================================
/*! (method) instance variable getter
 */
__CFUNC__
obj_getiv(GV v[], U32 vi)
{
    RETURN_VAL(ostore_get(v, v->vid));			// attribute 'x'
}

//================================================================
/*! (method) instance variable setter
 */
__CFUNC__
obj_setiv(GV v[], U32 vi)
{
    GS vid = v->vid;							// attribute 'x='
    ostore_set(v, vid-1, v+1);					// attribute 'x'
}


//================================================================
/*! append '=' to create name for attr_writer
 */
__GURU__ U8 *
_name_w_eq_sign(GV *buf, U8 *s0)
{
    guru_str_add_cstr(buf, s0);
    guru_str_add_cstr(buf, "=");

    U32 sid = name2id((U8*)buf->str->raw);

    return id2name(sid);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__CFUNC__
obj_attr_reader(GV v[], U32 vi)
{
	assert(v->gt==GT_CLASS);
	guru_class *cls = v->cls;							// fetch class

	GV *s = v+1;
    for (U32 i = 0; i < vi; i++, s++) {
        assert(s->gt==GT_SYM);

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
	assert(v->gt==GT_CLASS);
	guru_class *cls = IS_SCLASS(v) ? v->cls->cls : v->cls;		// fetch class
#if CC_DEBUG
    printf("%p:%s, sc=%d self=%d #attr_accessor\n", cls, cls->name, IS_SCLASS(v), IS_SELF(v));
#endif // CC_DEBUG
    GV buf = guru_str_buf(80);
	GV *s  = v+1;
    for (U32 i=0; i < vi; i++, s++) {
        assert(s->gt==GT_SYM);
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
	assert(v->gt==GT_OBJ);
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
	assert(v->gt==GT_CLASS && (v+1)->gt==GT_PROC);		// ensure it is a proc

	guru_proc *prc = (v+1)->proc;						// mark it as a lambda
	prc->meta |= PROC_LAMBDA;

	U32	n   = prc->n 	= vi+3;
	GV  *r  = prc->regs = (GV*)guru_alloc(sizeof(GV)*n);
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
	assert(1==0);				// handled in ucode
}

__GURU__ void
_init_class_object()
{
    static Vfunc vtbl[] = {
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
        { "to_s",          	gv_to_s  		},
        { "inspect",       	gv_to_s  		},
#if GURU_DEBUG
        { "p", 				obj_p    		},
        { "sprintf",		str_sprintf		},
        { "printf",			str_printf		}
#endif
     };
    guru_class_object = guru_add_class(
    	"Object", NULL, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}

//================================================================
// ProcClass
//================================================================

__GURU__ void
_init_class_proc()
{
    static Vfunc vtbl[] = {
//    	{ "call", 	prc_call	},		// handled  by ucode#uc_send
    	{ "to_s", 	gv_to_s		},
    	{ "inspect",gv_to_s		}
    };
    guru_class_proc = guru_add_class(
    	"Proc", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}

//================================================================
// Nil class
__CFUNC__
nil_false_not(GV v[], U32 vi)
{
    v->gt = GT_TRUE;
}

#if !GURU_USE_STRING
__CFUNC__ nil_inspect(GV v[], U32 vi) {}
#else
__CFUNC__
nil_inspect(GV v[], U32 vi)
{
    RETURN_VAL(guru_str_new("nil"));
}
#endif	// GURU_USE_STRING

//================================================================
/*! Nil class
 */
__GURU__ void
_init_class_nil()
{
    static Vfunc vtbl[] = {
    	{ "!", 			nil_false_not	},
    	{ "inspect", 	nil_inspect		},
    	{ "to_s", 		gv_to_s			}
    };
    guru_class_nil = guru_add_class(
    	"NilClass", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}

//================================================================
/*! False class
 */
__GURU__ void
_init_class_false()
{
    static Vfunc vtbl[] = {
    	{ "!", 		nil_false_not	},
    	{ "to_s",    gv_to_s		},
    	{ "inspect", gv_to_s		}
    };
    guru_class_false = guru_add_class(
    	"FalseClass", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}

__GURU__ void
_init_class_true()
{
	static Vfunc vtbl[] = {
		{ "to_s", 		gv_to_s 	},
		{ "inspect", 	gv_to_s		}
	};
    guru_class_true = guru_add_class(
    	"TrueClass", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}

//================================================================
// initialize

__GURU__ void
_init_all_class(void)
{
    _init_class_object();
    _init_class_nil();
    _init_class_proc();
    _init_class_false();
    _init_class_true();

    guru_init_class_symbol();		// symbol.cu
    guru_init_class_int();			// c_fixnum.cu
    guru_init_class_float();		// c_fixnum.cu

    guru_init_class_string();		// c_string.cu
    guru_init_class_array();		// c_array.cu
    guru_init_class_range();		// c_range.cu
    guru_init_class_hash();			// c_hash.cu

#if GURU_USE_MATH
    guru_init_class_math();
#endif // GURU_USE_MATH
}

__GPU__ void
guru_class_init(void)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	//
	// TODO: load image into context memory
	//
	_init_all_class();
}
