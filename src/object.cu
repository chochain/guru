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

    guru_proc *m = proc_by_sid(rcv, sid);	// method of receiver object

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
    GV ret;  { ret.gt = GT_CLASS; }
    ret.cls = class_by_obj(v);

    RETURN_VAL(ret);
}

//================================================================
/*! get callee name

  @param  vm	Pointer to VM
  @return	string
*/
__GURU__ U8*
_get_callee(GV v[])
{
#if 0
    U32 code = *((U32*)VM_ISEQ(vm) + vm->state->pc - 1);

    int rb = GETARG_B(code);  		// index of method sym

    return _vm_symbol(vm, rb);
#endif
    guru_na("method not supported: callee\n");

    return NULL;
}

//================================================================
/*! (method) instance variable getter
 */
__CFUNC__
obj_getiv(GV v[], U32 vi)
{
    const U8 *name = _get_callee(v);			// TODO:
    GS sid = name2id(name);

    RETURN_VAL(ostore_get(v, sid));
}

//================================================================
/*! (method) instance variable setter
 */
__CFUNC__
obj_setiv(GV v[], U32 vi)
{
    const U8 *name = _get_callee(v);			// TODO:
    GS sid = name2id(name);

    ostore_set(v, sid, &v[1]);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__CFUNC__
obj_attr_reader(GV v[], U32 vi)
{
	GV *v0 = v;
	GV *p  = v+1;
    for (U32 i = 0; i < vi; i++, p++) {
        if (p->gt != GT_SYM) continue;	// TypeError raise?

        // define reader method
        U8 *name = id2name(p->i);
        guru_define_method(v0->cls, name, obj_getiv);
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
__CFUNC__
obj_attr_accessor(GV v[], U32 vi)
{
	GV *v0 = v;
	GV *p  = v+1;
    for (U32 i=0; i < vi; i++, p++) {
        if (p->gt != GT_SYM) continue;				// TypeError raise?

        // define reader method
        U8 *name = id2name(p->i);
        guru_define_method(v0->cls, name, obj_getiv);

        U32 asz = STRLEN(name);	ALIGN(asz);				// 8-byte aligned

        // make string "....=" and define writer method.
        // TODO: consider using static buffer
        U8 *buf = (U8*)guru_alloc(asz);

        STRCPY(buf, name);
        STRCAT(buf, "=");
        guru_sym_new(buf);
        guru_define_method(v0->cls, buf, obj_setiv);

        guru_free(buf);
    }
}

//================================================================
/*! (method) is_a, kind_of
 */
__CFUNC__
obj_kind_of(GV v[], U32 vi)
{
    if ((v+1)->gt != GT_CLASS) {
        RETURN_BOOL(0);
    }
    const guru_class *cls = class_by_obj(v);

    while (cls) {
        if (cls == (v+1)->cls) break;
        cls = cls->super;
    }
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
    	{ "!",				obj_not 		},
    	{ "!=",          	obj_neq 		},
    	{ "<=>",           	obj_cmp 		},
    	{ "===",           	obj_eq3 		},
    	{ "class",         	obj_class		},
//    	{ "new",           	obj_new 		},		// handled by vm#vm_method_exec
//      { "raise",			obj_raise		},		// handled by vm#vm_method_exec
    	{ "attr_reader",   	obj_attr_reader 	},
    	{ "attr_accessor", 	obj_attr_accessor	},
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
