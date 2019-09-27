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
#include "value.h"
#include "alloc.h"
#include "static.h"
#include "symbol.h"
//#include "global.h"
//#include "vm.h"
#include "class.h"

//#include "ucode.h"
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
_send(GV v[], GV *rcv, const U8P method, U32 argc, ...)
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
        regs[i] = (i>argc) ? GURU_NIL_NEW() : *va_arg(ap, GV *);
    }
    va_end(ap);

    m->func(regs, argc);			// call method, put return value in regs[0]

#if GURU_DEBUG
    GV *r = v;						// _wipe_stack
    for (U32 i=1; i<=argc+1; i++, r++) {
    	r->gt 	= GT_EMPTY;			// clean up the stack
    	r->self = NULL;
    }
#endif
    return regs[0];
}

__GURU__ GV
guru_inspect(GV v[], GV *obj)
{
	return _send(v, obj, (U8P)"inspect", 0);
}

__GURU__ GV
guru_kind_of(GV v[])		// whether v1 is a kind of v0
{
	return _send(v, v+1, (U8P)"kind_of?", 1, v);
}

__GURU__ void
guru_obj_del(GV *v)
{
	assert(v->gt==GT_OBJ);

	ostore_del(v);
}

//================================================================
__GURU__ void
obj_nop(GV v[], U32 vi)
{
	// do nothing
}

#if GURU_DEBUG
//================================================================
/*! (method) p
 */
__GURU__ void
obj_p(GV v[], U32 vi)
{
	guru_p(v, vi);
}
#endif

//================================================================
/*! (method) puts
 */
__GURU__ void
obj_puts(GV v[], U32 vi)
{
	guru_puts(v, vi);
}

//================================================================
/*! (method) print
 */
__GURU__ void
obj_print(GV v[], U32 vi)
{
	guru_puts(v, vi);
}

//================================================================
/*! (operator) !
 */
__GURU__ void
obj_not(GV v[], U32 vi)
{
    RETURN_FALSE();
}

//================================================================
/*! (operator) !=
 */
__GURU__ void
obj_neq(GV v[], U32 vi)
{
    S32 t = guru_cmp(&v[0], &v[1]);
    RETURN_BOOL(t);
}

//================================================================
/*! (operator) <=>
 */
__GURU__ void
obj_cmp(GV v[], U32 vi)
{
    S32 t = guru_cmp(&v[0], &v[1]);
    RETURN_INT(t);
}

//================================================================
/*! (operator) ===
 */
__GURU__ void
obj_eq3(GV v[], U32 vi)
{
    if (v[0].gt != GT_CLASS) {
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
__GURU__ void
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
__GURU__ U8P
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
__GURU__ void
obj_getiv(GV v[], U32 vi)
{
    const U8P name = _get_callee(v);			// TODO:
    GS sid = name2id(name);

    RETURN_VAL(ostore_get(&v[0], sid));
}

//================================================================
/*! (method) instance variable setter
 */
__GURU__ void
obj_setiv(GV v[], U32 vi)
{
    const U8P name = _get_callee(v);			// TODO:
    GS sid = name2id(name);

    ostore_set(&v[0], sid, &v[1]);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__GURU__ void
obj_attr_reader(GV v[], U32 vi)
{
    for (U32 i = 1; i <= vi; i++) {
        if (v[i].gt != GT_SYM) continue;	// TypeError raise?

        // define reader method
        U8P name = id2name(v[i].i);
        guru_define_method(v[0].cls, name, obj_getiv);
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
__GURU__ void
obj_attr_accessor(GV v[], U32 vi)
{
    for (U32 i = 1; i <= vi; i++) {
        if (v[i].gt != GT_SYM) continue;				// TypeError raise?

        // define reader method
        U8P name = id2name(v[i].i);
        guru_define_method(v[0].cls, name, obj_getiv);

        U32 asz = STRLEN(name);	ALIGN(asz);				// 8-byte aligned

        // make string "....=" and define writer method.
        // TODO: consider using static buffer
        U8P buf = (U8P)guru_alloc(asz);
        
        STRCPY(buf, name);
        STRCAT(buf, "=");
        guru_sym_new(buf);
        guru_define_method(v[0].cls, buf, obj_setiv);

        guru_free(buf);
    }
}

//================================================================
/*! (method) is_a, kind_of
 */
__GURU__ void
obj_kind_of(GV v[], U32 vi)
{
    if (v[1].gt != GT_CLASS) {
        RETURN_BOOL(0);
    }
    const guru_class *cls = class_by_obj(&v[0]);

    while (cls) {
        if (cls == v[1].cls) break;
        cls = cls->super;
    }
}

__GURU__ void
obj_new(GV v[], U32 vi)
{
	assert(1==0);				// handled in ucode
}

//=====================================================================
//! deprecated, use inspect#gv_to_s instead
__GURU__ void
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
    	{ "new",           	obj_new 		},
    	{ "attr_reader",   	obj_attr_reader },
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
prc_call(GV v[], U32 vi)
{
	// not suppose to come here
	assert(1==0);		// taken care by vm#op_send
}

__GURU__ void
_init_class_proc()
{
    static Vfunc vtbl[] = {
    	{ "call", 	prc_call	},
    	{ "to_s", 	gv_to_s		},
    	{ "inspect",gv_to_s		}
    };
    guru_class_proc = guru_add_class(
    	"Proc", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}

//================================================================
// Nil class
__GURU__ void
nil_false_not(GV v[], U32 vi)
{
    v[0].gt = GT_TRUE;
}

#if !GURU_USE_STRING
__GURU__ void nil_inspect(GV v[], U32 vi) {}
#else
__GURU__ void
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

    guru_init_class_symbol();
    guru_init_class_int();
    guru_init_class_float();

    guru_init_class_string();
    guru_init_class_array();
    guru_init_class_range();
    guru_init_class_hash();

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
