/*! @file
  @brief
  Guru Object, Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

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
#include "instance.h"
#include "global.h"
#include "vm.h"
#include "class.h"

#include "console.h"
#include "puts.h"
#include "opcode.h"
#include "object.h"

#include "c_fixnum.h"

#if GURU_USE_STRING
#include "sprintf.h"
#include "c_string.h"
#endif

#if GURU_USE_ARRAY
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"
#endif


//================================================================
/*! (BETA) Call any method of the object, but written by C.

  @param  vm		pointer to vm.
  @param  v		see bellow example.
  @param  reg_ofs	see bellow example.
  @param  recv		pointer to receiver.
  @param  name		method name.
  @param  argc		num of params.

  @example
  // (Fixnum).to_s(16)
  void c_fixnum_to_s(mrbc_value v[], U32 argc)
  {
  mrbc_value *recv = &v[1];
  mrbc_value arg1 = mrbc_fixnum_value(16);
  mrbc_value ret = mrbc_send(vm, v, argc, recv, "to_s", 1, &arg1);
  SET_RETURN(ret);
  }
*/
__GURU__ mrbc_value
mrbc_send(mrbc_value v[], mrbc_value *rcv, const U8 *method, U32 argc, ...)
{
    mrbc_value *regs = v + 2;	     // allocate 2 for stack
    mrbc_sym   sid   = name2symid(method);
    mrbc_proc  *m    = mrbc_get_class_method(*rcv, sid);

    if (m == 0) {
        console_str("No method. vtype=");
        console_int(rcv->tt);
        console_str(" method='");
        console_str(method);
        console_str("'\n");
        return mrbc_nil_value();
    }
    if (!IS_CFUNC(m)) {
        console_str("Method is not C function: ");
        console_str(method);
        console_str("\n");
        return mrbc_nil_value();
    }

    // create call stack.
    mrbc_release(&regs[0]);
    regs[0] = *rcv;					// create call stack, start with receiver object
    mrbc_retain(rcv);

    va_list ap;						// setup calling registers
    va_start(ap, argc);
    for (int i = 1; i <= argc+1; i++) {
    	mrbc_release(&regs[i]);
        regs[i] = (i>argc) ? mrbc_nil_value() : *va_arg(ap, mrbc_value *);
    }
    va_end(ap);

    m->func(regs, argc);			// call method
    mrbc_value ret = regs[0];		// copy result

#ifdef GURU_DEBUG
    for (int i=0; i<=argc+1; i++) {	// not really needed!
    	regs[i].tt = GURU_TT_EMPTY;	// but, clean up the stack before returning
    }
#endif
    return ret;
}

#ifdef GURU_DEBUG
//================================================================
/*! (method) p
 */
__GURU__ void
c_p(mrbc_value v[], U32 argc)
{
    for (int i = 1; i <= argc; i++) {
        mrbc_p_sub(&v[i]);
        console_char('\n');
    }
}
#endif

//================================================================
/*! (method) puts
 */
__GURU__ void
c_puts(mrbc_value v[], U32 argc)
{
    if (argc) {
    	for (int i = 1; i <= argc; i++) {
    		if (mrbc_print_sub(&v[i]) == 0) console_char('\n');
    	}
    }
    else console_char('\n');
}

//================================================================
/*! (method) print
 */
__GURU__ void
c_print(mrbc_value v[], U32 argc)
{
    for (int i = 1; i <= argc; i++) {
        mrbc_print_sub(&v[i]);
    }
}

//================================================================
/*! (operator) !
 */
__GURU__ void
c_object_not(mrbc_value v[], U32 argc)
{
    SET_FALSE_RETURN();
}

//================================================================
/*! (operator) !=
 */
__GURU__ void
c_object_neq(mrbc_value v[], U32 argc)
{
    S32 t = mrbc_compare(&v[0], &v[1]);
    SET_BOOL_RETURN(t);
}

//================================================================
/*! (operator) <=>
 */
__GURU__ void
c_object_compare(mrbc_value v[], U32 argc)
{
    S32 t = mrbc_compare(&v[0], &v[1]);
    SET_INT_RETURN(t);
}

//================================================================
/*! (operator) ===
 */
__GURU__ void
c_object_equal3(mrbc_value v[], U32 argc)
{
	S32 ret = mrbc_compare(&v[0], &v[1]);

    if (v[0].tt != GURU_TT_CLASS) SET_BOOL_RETURN(ret==0);
    else 						  SET_RETURN(mrbc_send(v+argc, &v[1], "kind_of?", 1, &v[0]));
}

//================================================================
/*! (method) class
 */
__GURU__ void
c_object_class(mrbc_value v[], U32 argc)
{
    mrbc_value ret = {.tt = GURU_TT_CLASS };
    ret.cls = mrbc_get_class_by_object(v);

    SET_RETURN(ret);
}

// Object.new
__GURU__ void
c_object_new(mrbc_value v[], U32 argc)
{
	assert(1==0);			// taken cared in opcode
}

//================================================================
/*! get callee name

  @param  vm	Pointer to VM
  @return	string
*/
__GURU__ const U8*
_get_callee(guru_vm *vm)
{
#if 0
    U32 code = *(VM_ISEQ(vm) + vm->state->pc - 1);

    int rb = GETARG_B(code);  // index of method sym

    return _vm_symbol(vm, rb);
#endif
    console_na("callee");

    return NULL;
}

//================================================================
/*! (method) instance variable getter
 */
__GURU__ void
c_object_getiv(mrbc_value v[], U32 argc)
{
    const U8 *name = _get_callee(NULL);			// TODO:
    mrbc_sym sid   = name2symid(name);

    SET_RETURN(mrbc_instance_getiv(&v[0], sid));
}

//================================================================
/*! (method) instance variable setter
 */
__GURU__ void
c_object_setiv(mrbc_value v[], U32 argc)
{
    const U8 *name = _get_callee(NULL);			// CC TODO: another way
    mrbc_sym sid   = name2symid(name);

    mrbc_instance_setiv(&v[0], sid, &v[1]);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__GURU__ void
c_object_attr_reader(mrbc_value v[], U32 argc)
{
    for (int i = 1; i <= argc; i++) {
        if (v[i].tt != GURU_TT_SYMBOL) continue;	// TypeError raise?

        // define reader method
        const U8 *name = VSYM(&v[i]);
        mrbc_define_method(v[0].cls, name, (mrbc_func_t)c_object_getiv);
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
__GURU__ void
c_object_attr_accessor(mrbc_value v[], U32 argc)
{
    for (int i = 1; i <= argc; i++) {
        if (v[i].tt != GURU_TT_SYMBOL) continue;	// TypeError raise?

        // define reader method
        U8 *name = VSYM(&v[i]);
        mrbc_define_method(v[0].cls, name, (mrbc_func_t)c_object_getiv);

        // make string "....=" and define writer method.
        U8 *buf = (U8 *)mrbc_alloc(STRLEN(name)+2);
        if (!buf) return;
        
        STRCPY(buf, name);
        STRCAT(buf, "=");
        mrbc_symbol_new(buf);
        mrbc_define_method(v[0].cls, buf, (mrbc_func_t)c_object_setiv);
        mrbc_free(buf);
    }
}

//================================================================
/*! (method) is_a, kind_of
 */
__GURU__ void
c_object_kind_of(mrbc_value v[], U32 argc)
{
    if (v[1].tt != GURU_TT_CLASS) {
        SET_BOOL_RETURN(0);
        return;
    }
    const mrbc_class *cls = mrbc_get_class_by_object(&v[0]);

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
c_object_to_s(mrbc_value v[], U32 argc)
{
	const U8 *str;
	U8    buf[20];

    switch (v->tt) {
    case GURU_TT_CLASS:
    	str = symid2name(v->cls->sym_id); 								break;
    case GURU_TT_OBJECT:
    	str = symid2name(v->self->cls->sym_id);
    	str = guru_sprintf(buf, "#<%s:%08x>", str, (uintptr_t)v->self); break;
    default: str = ""; break;
    }
    SET_RETURN(mrbc_string_new(str));
}
#endif

__GURU__ void
_init_class_object()
{
    // Class
    mrbc_class *c = mrbc_class_object = mrbc_define_class("Object", NULL);

    // Methods
    mrbc_define_method(c, "initialize",    	c_nop);
    mrbc_define_method(c, "puts",          	c_puts);
    mrbc_define_method(c, "print",         	c_print);
    mrbc_define_method(c, "!",             	c_object_not);
    mrbc_define_method(c, "!=",            	c_object_neq);
    mrbc_define_method(c, "<=>",           	c_object_compare);
    mrbc_define_method(c, "===",           	(mrbc_func_t)c_object_equal3);
    mrbc_define_method(c, "class",         	c_object_class);
    mrbc_define_method(c, "new",           	(mrbc_func_t)c_object_new);
    mrbc_define_method(c, "attr_reader",   	c_object_attr_reader);
    mrbc_define_method(c, "attr_accessor", 	c_object_attr_accessor);
    mrbc_define_method(c, "is_a?",         	c_object_kind_of);
    mrbc_define_method(c, "kind_of?",      	c_object_kind_of);
#if GURU_USE_STRING
    mrbc_define_method(c, "inspect",       	c_object_to_s);
    mrbc_define_method(c, "to_s",          	c_object_to_s);
#endif
#ifdef GURU_DEBUG
    mrbc_define_method(c, "p", 				c_p);
#endif
}

#if GURU_USE_STRING
__GURU__ void
c_proc_inspect(mrbc_value v[], U32 argc)
{
	U8 buf[20];
    const U8 *str = guru_sprintf(buf, "<#Proc:%08x>", (U32P)v->proc);

    SET_RETURN(mrbc_string_new(str));
}
#endif

__GURU__ void
_init_class_proc()
{
    // Class
    mrbc_class *c = mrbc_class_proc = mrbc_define_class("Proc", mrbc_class_object);
    // Methods
    mrbc_define_method(c, "call", (mrbc_func_t)c_proc_call);
#if GURU_USE_STRING
    mrbc_define_method(c, "inspect", 	c_proc_inspect);
    mrbc_define_method(c, "to_s", 		c_proc_inspect);
#endif
}

//================================================================
// Nil class

//================================================================
/*! (method) !
 */
__GURU__ void
c_nil_false_not(mrbc_value v[], U32 argc)
{
    v[0].tt = GURU_TT_TRUE;
}

#if GURU_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__ void
c_nil_inspect(mrbc_value v[], U32 argc)
{
    v[0] = mrbc_string_new("nil");
}

//================================================================
/*! (method) to_s
 */
__GURU__ void
c_nil_to_s(mrbc_value v[], U32 argc)
{
    v[0] = mrbc_string_new(NULL);
}
#endif

//================================================================
/*! Nil class
 */
__GURU__ void
_init_class_nil()
{
    // Class
    mrbc_class *c = mrbc_class_nil = mrbc_define_class("NilClass", mrbc_class_object);
    // Methods
    mrbc_define_method(c, "!", 			c_nil_false_not);
#if GURU_USE_STRING
    mrbc_define_method(c, "inspect", 	c_nil_inspect);
    mrbc_define_method(c, "to_s", 		c_nil_to_s);
#endif
}

//================================================================
// False class

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
c_false_to_s(mrbc_value v[], U32 argc)
{
    v[0] = mrbc_string_new("false");
}
#endif

//================================================================
/*! False class
 */
__GURU__ void
_init_class_false()
{
    // Class
    mrbc_class_false = mrbc_define_class("FalseClass", mrbc_class_object);
    // Methods
    mrbc_define_method(mrbc_class_false, "!", 		c_nil_false_not);
#if GURU_USE_STRING
    mrbc_define_method(mrbc_class_false, "inspect", c_false_to_s);
    mrbc_define_method(mrbc_class_false, "to_s",    c_false_to_s);
#endif
}

//================================================================
// True class

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
c_true_to_s(mrbc_value v[], U32 argc)
{
    v[0] = mrbc_string_new("true");
}
#endif

__GURU__ void
_init_class_true()
{
    // Class
    mrbc_class_true = mrbc_define_class("TrueClass", mrbc_class_object);
    // Methods
#if GURU_USE_STRING
    mrbc_define_method(mrbc_class_true, "inspect", 	c_true_to_s);
    mrbc_define_method(mrbc_class_true, "to_s", 	c_true_to_s);
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

    mrbc_init_class_symbol();
    mrbc_init_class_fixnum();
#if GURU_USE_FLOAT
    mrbc_init_class_float();
#if GURU_USE_MATH
    mrbc_init_class_math();
#endif
#endif
    
#if GURU_USE_STRING
    mrbc_init_class_string();
#endif
#if GURU_USE_ARRAY
    mrbc_init_class_array();
    mrbc_init_class_range();
    mrbc_init_class_hash();
#endif
}

__GPU__ void
guru_class_init(void)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	_init_all_class();
}
