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
#include "store.h"
#include "global.h"
#include "vm.h"
#include "class.h"

#include "opcode.h"
#include "object.h"

#include "c_fixnum.h"

#if GURU_USE_STRING
#include "c_string.h"
#endif

#if GURU_USE_ARRAY
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"
#endif

#include "puts.h"

//================================================================
/*! (BETA) Call any method of the object, but written by C.

  @param  vm		pointer to vm.
  @param  v		see bellow example.
  @param  reg_ofs	see bellow example.
  @param  recv		pointer to receiver.
  @param  name		method name.
  @param  argc		num of params.

  @example
  // (int).to_s(16)
  void c_int_to_s(GV v[], U32 argc)
  {
  GV *recv = &v[1];
  GV arg1 = guru_int_value(16);
  GV ret  = _send(vm, v, argc, recv, "to_s", 1, &arg1);
  RETURN_VAL(ret);
  }
*/
__GURU__ GV
_send(GV v[], GV *rcv, const U8P method, U32 argc, ...)
{
    GV *regs = v + 2;	     // allocate 2 for stack
    GS   sid   = name2id(method);
    guru_proc  *m    = proc_by_sid(rcv, sid);

    if (m == 0) {
        PRINTF("No method. vtype=%d method='%s'\n", rcv->gt, method);
        return GURU_NIL_NEW();
    }
    if (!IS_CFUNC(m)) {
        PRINTF("Method '%s' is not a C function\n", method);
        return GURU_NIL_NEW();
    }

    // create call stack.
    ref_clr(&regs[0]);
    regs[0] = *rcv;					// create call stack, start with receiver object
    ref_inc(rcv);

    va_list ap;						// setup calling registers
    va_start(ap, argc);
    for (U32 i = 1; i <= argc+1; i++) {
    	ref_clr(&regs[i]);
        regs[i] = (i>argc) ? GURU_NIL_NEW() : *va_arg(ap, GV *);
    }
    va_end(ap);

    m->func(regs, argc);			// call method
    GV ret = regs[0];		// copy result

#if GURU_DEBUG
    for (U32 i=0; i<=argc+1; i++) {	// not really needed!
    	regs[i].gt = GT_EMPTY;	// but, clean up the stack before returning
    }
#endif
    return ret;
}

__GURU__ GV
guru_inspect(GV v[], GV *rcv)
{
	return _send(v, rcv, (U8P)"inspect", 0);
}

__GURU__ GV
guru_kind_of(GV v[], U32 argc)		// whether v1 is a kind of v0
{
	return _send(v+argc, v+1, (U8P)"kind_of?", 1, v);
}

//================================================================
__GURU__ void
c_nop(GV v[], U32 argc)
{
	// do nothing
}

#if GURU_DEBUG
//================================================================
/*! (method) p
 */
__GURU__ void
c_p(GV v[], U32 argc)
{
	guru_p(v, argc);
}
#endif

//================================================================
/*! (method) puts
 */
__GURU__ void
c_puts(GV v[], U32 argc)
{
	guru_puts(v, argc);
}

//================================================================
/*! (method) print
 */
__GURU__ void
c_print(GV v[], U32 argc)
{
	guru_puts(v, argc);
}

//================================================================
/*! (operator) !
 */
__GURU__ void
c_object_not(GV v[], U32 argc)
{
    RETURN_FALSE();
}

//================================================================
/*! (operator) !=
 */
__GURU__ void
c_object_neq(GV v[], U32 argc)
{
    S32 t = guru_cmp(&v[0], &v[1]);
    RETURN_BOOL(t);
}

//================================================================
/*! (operator) <=>
 */
__GURU__ void
c_object_compare(GV v[], U32 argc)
{
    S32 t = guru_cmp(&v[0], &v[1]);
    RETURN_INT(t);
}

//================================================================
/*! (operator) ===
 */
__GURU__ void
c_object_equal3(GV v[], U32 argc)
{
	S32 ret = guru_cmp(v, v+1);

    if (v[0].gt != GT_CLASS) RETURN_BOOL(ret==0);
    else 					 RETURN_VAL(guru_kind_of(v, argc));
}

//================================================================
/*! (method) class
 */
__GURU__ void
c_object_class(GV v[], U32 argc)
{
    GV ret = {.gt = GT_CLASS };
    ret.cls = class_by_obj(v);

    RETURN_VAL(ret);
}

//================================================================
/*! get callee name

  @param  vm	Pointer to VM
  @return	string
*/
__GURU__ U8P
_get_callee(guru_vm *vm)
{
#if 0
    U32 code = *((U32*)VM_ISEQ(vm) + vm->state->pc - 1);

    int rb = GETARG_B(code);  // index of method sym

    return _vm_symbol(vm, rb);
#endif
    guru_na("method not supported: callee\n");

    return NULL;
}

//================================================================
/*! (method) instance variable getter
 */
__GURU__ void
c_object_getiv(GV v[], U32 argc)
{
    const U8P name = _get_callee(NULL);			// TODO:
    GS  sid  = name2id(name);

    RETURN_VAL(guru_store_get(&v[0], sid));
}

//================================================================
/*! (method) instance variable setter
 */
__GURU__ void
c_object_setiv(GV v[], U32 argc)
{
    U8P name = _get_callee(NULL);			// CC TODO: another way
    GS  sid  = name2id(name);

    guru_store_set(&v[0], sid, &v[1]);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__GURU__ void
c_object_attr_reader(GV v[], U32 argc)
{
    for (U32 i = 1; i <= argc; i++) {
        if (v[i].gt != GT_SYM) continue;	// TypeError raise?

        // define reader method
        U8P name = id2name(v[i].i);
        guru_define_method(v[0].cls, name, c_object_getiv);
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
__GURU__ void
c_object_attr_accessor(GV v[], U32 argc)
{
    for (U32 i = 1; i <= argc; i++) {
        if (v[i].gt != GT_SYM) continue;				// TypeError raise?

        // define reader method
        U8P name = id2name(v[i].i);
        guru_define_method(v[0].cls, name, c_object_getiv);

        U32 sz = STRLEN(name);

        // make string "....=" and define writer method.
        // TODO: consider using static buffer
        U8P buf = (U8P)guru_alloc(sz + (-sz & 7));		// 8-byte aligned
        
        STRCPY(buf, name);
        STRCAT(buf, "=");
        guru_sym_new(buf);
        guru_define_method(v[0].cls, buf, c_object_setiv);

        guru_free(buf);
    }
}

//================================================================
/*! (method) is_a, kind_of
 */
__GURU__ void
c_object_kind_of(GV v[], U32 argc)
{
    if (v[1].gt != GT_CLASS) {
        RETURN_BOOL(0);
    }
    const guru_class *cls = class_by_obj(&v[0]);

    do {
        if (cls == v[1].cls) break;

        cls = cls->super;
    } while (cls != NULL);
}

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
c_object_to_s(GV v[], U32 argc)
{
	GV ret;
	U8P name;

    switch (v->gt) {
    case GT_CLASS:
    	name = id2name(v->cls->sid);
    	ret = guru_str_new(name);
    	break;
    case GT_OBJ:
    	name = id2name(v->self->cls->sid);
    	ret  = guru_str_new("#<0x");
    	guru_str_append_cstr(&ret, name);
    	guru_str_append_cstr(&ret, guru_i2s((U64)v->self, 16));
    	guru_str_append_cstr(&ret, ">");
    	break;
    default:
    	ret = guru_str_new("");
    	break;
    }
    RETURN_VAL(ret);
}
#endif

__GURU__ void
c_object_new(GV v[], U32 argc)
{
	assert(1==0);		// taken cared in opcode handler
}

__GURU__ void
_init_class_object()
{
    // Class
    guru_class *c = guru_class_object = guru_add_class("Object", NULL);

    // Methods
    guru_add_proc(c, "initialize",    	c_nop);
#if GURU_DEBUG
    guru_add_proc(c, "p", 				c_p);
#endif
    guru_add_proc(c, "puts",          	c_puts);
    guru_add_proc(c, "print",         	c_print);
    guru_add_proc(c, "!",             	c_object_not);
    guru_add_proc(c, "!=",            	c_object_neq);
    guru_add_proc(c, "<=>",           	c_object_compare);
    guru_add_proc(c, "===",           	c_object_equal3);
    guru_add_proc(c, "class",         	c_object_class);
    guru_add_proc(c, "new",           	c_object_new);
    guru_add_proc(c, "attr_reader",   	c_object_attr_reader);
    guru_add_proc(c, "attr_accessor", 	c_object_attr_accessor);
    guru_add_proc(c, "is_a?",         	c_object_kind_of);
    guru_add_proc(c, "kind_of?",      	c_object_kind_of);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect",       	c_object_to_s);
    guru_add_proc(c, "to_s",          	c_object_to_s);
#endif
}

//================================================================
// ProcClass
//================================================================
__GURU__ void
c_proc_call(GV v[], U32 argc)
{
	// not suppose to come here
	assert(1==0);		// taken care by vm#op_send
}

#if GURU_USE_STRING
__GURU__ void
c_proc_inspect(GV v[], U32 argc)
{
	GV ret = guru_str_new("<#Proc:");
	guru_str_append_cstr(&ret, guru_i2s((U64)v->proc, 16));

    RETURN_VAL(ret);
}
#endif

__GURU__ void
_init_class_proc()
{
    // Class
    guru_class *c = guru_class_proc = guru_add_class("Proc", guru_class_object);
    // Methods
    guru_add_proc(c, "call", 	c_proc_call);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect", c_proc_inspect);
    guru_add_proc(c, "to_s", 	c_proc_inspect);
#endif
}

//================================================================
// Nil class

//================================================================
/*! (method) !
 */
__GURU__ void
c_nil_false_not(GV v[], U32 argc)
{
    v[0].gt = GT_TRUE;
}

#if GURU_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__ void
c_nil_inspect(GV v[], U32 argc)
{
    v[0] = guru_str_new("nil");
}

//================================================================
/*! (method) to_s
 */
__GURU__ void
c_nil_to_s(GV v[], U32 argc)
{
    v[0] = guru_str_new(NULL);
}
#endif

//================================================================
/*! Nil class
 */
__GURU__ void
_init_class_nil()
{
    // Class
    guru_class *c = guru_class_nil = guru_add_class("NilClass", guru_class_object);
    // Methods
    guru_add_proc(c, "!", 			c_nil_false_not);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect", 	c_nil_inspect);
    guru_add_proc(c, "to_s", 		c_nil_to_s);
#endif
}

//================================================================
// False class

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
c_false_to_s(GV v[], U32 argc)
{
    v[0] = guru_str_new("false");
}
#endif

//================================================================
/*! False class
 */
__GURU__ void
_init_class_false()
{
    // Class
    guru_class *c = guru_class_false = guru_add_class("FalseClass", guru_class_object);
    // Methods
    guru_add_proc(c, "!", 		c_nil_false_not);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect", c_false_to_s);
    guru_add_proc(c, "to_s",    c_false_to_s);
#endif
}

//================================================================
// True class

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
c_true_to_s(GV v[], U32 argc)
{
    v[0] = guru_str_new("true");
}
#endif

__GURU__ void
_init_class_true()
{
    // Class
    guru_class_true = guru_add_class("TrueClass", guru_class_object);
    // Methods
#if GURU_USE_STRING
    guru_add_proc(guru_class_true, "inspect", 	c_true_to_s);
    guru_add_proc(guru_class_true, "to_s", 		c_true_to_s);
#endif
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
#if GURU_USE_FLOAT
    guru_init_class_float();
#if GURU_USE_MATH
    guru_init_class_math();
#endif
#endif
    
#if GURU_USE_STRING
    guru_init_class_string();
#endif
#if GURU_USE_ARRAY
    guru_init_class_array();
    guru_init_class_range();
    guru_init_class_hash();
#endif
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
