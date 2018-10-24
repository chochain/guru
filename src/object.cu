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

#if MRBC_USE_STRING
#include "sprintf.h"
#include "c_string.h"
#endif

#if MRBC_USE_ARRAY
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
  void c_fixnum_to_s(mrbc_value v[], int argc)
  {
  mrbc_value *recv = &v[1];
  mrbc_value arg1 = mrbc_fixnum_value(16);
  mrbc_value ret = mrbc_send(vm, v, argc, recv, "to_s", 1, &arg1);
  SET_RETURN(ret);
  }
*/
__GURU__
mrbc_value _mrbc_send(mrbc_value *v, int reg_ofs,
                      mrbc_value *recv, const char *method, int argc, ...)
{
    mrbc_sym  sym_id = name2symid(method);
    mrbc_proc *m     = mrbc_get_class_method(*recv, sym_id);
    mrbc_value *regs = v + reg_ofs + 2;

    if (m == 0) {
        console_str("No method. vtype=");
        console_int(recv->tt);
        console_str(" method='");
        console_str(method);
        console_str("'\n");
        return mrbc_nil_value();
    }
    if (!IS_C_FUNC(m)) {
        console_str("Method is not C function: ");
        console_str(method);
        console_str("\n");
        return mrbc_nil_value();
    }

    // create call stack.
    mrbc_release(&regs[0]);
    regs[0] = *recv;
    mrbc_dup(recv);

    va_list ap;
    va_start(ap, argc);

    int i;
    for (i = 1; i <= argc; i++) {
        mrbc_release(&regs[i]);
        regs[i] = *va_arg(ap, mrbc_value *);
    }
    mrbc_release(&regs[i]);
    regs[i] = mrbc_nil_value();
    va_end(ap);

    // call method.
    m->func(regs, argc);
    mrbc_value ret = regs[0];

    for (; i >= 0; i--) {
        regs[i].tt = MRBC_TT_EMPTY;
    }
    return ret;
}

#ifdef MRBC_DEBUG
//================================================================
/*! (method) p
 */
__GURU__
void c_p(mrbc_value v[], int argc)
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
__GURU__
void c_puts(mrbc_value v[], int argc)
{
    if (argc) {
    	for (int i = 1; i <= argc; i++) {
    		if (mrbc_puts_sub(&v[i]) == 0) console_char('\n');
    	}
    }
    else {
    	console_char('\n');
    }
}

//================================================================
/*! (method) print
 */
__GURU__
void c_print(mrbc_value v[], int argc)
{
    for (int i = 1; i <= argc; i++) {
        mrbc_print_sub(&v[i]);
    }
}

//================================================================
/*! (operator) !
 */
__GURU__ void c_object_not(mrbc_value v[], int argc)
{
    SET_FALSE_RETURN();
}

//================================================================
/*! (operator) !=
 */
__GURU__
void c_object_neq(mrbc_value v[], int argc)
{
    int result = mrbc_compare(&v[0], &v[1]);
    SET_BOOL_RETURN(result != 0);
}

//================================================================
/*! (operator) <=>
 */
__GURU__
void c_object_compare(mrbc_value v[], int argc)
{
    int result = mrbc_compare(&v[0], &v[1]);
    SET_INT_RETURN(result);
}

//================================================================
/*! (operator) ===
 */
__GURU__
void c_object_equal3(mrbc_value v[], int argc)
{
    if (v[0].tt == MRBC_TT_CLASS) {
        mrbc_value result = _mrbc_send(v, argc, &v[1], "kind_of?", 1, &v[0]);
        SET_RETURN(result);
    }
    else {
        int result = mrbc_compare(&v[0], &v[1]);
        SET_BOOL_RETURN(result == 0);
    }
}

//================================================================
/*! (method) class
 */
__GURU__
void c_object_class(mrbc_value v[], int argc)
{
    mrbc_value value = {.tt = MRBC_TT_CLASS};
    value.cls = mrbc_get_class_by_object(v);
    SET_RETURN(value);
}

// Object.new
__GURU__
void c_object_new(mrbc_value v[], int argc)
{
    mrbc_value new_obj = mrbc_instance_new(v->cls, 0);

    char sym[]="______initialize";
    _uint32_to_bin(1, (uint8_t*)&sym[0]);
    _uint16_to_bin(10,(uint8_t*)&sym[4]);

    uint32_t code[2] = {
        (uint32_t)(MKOPCODE(OP_SEND) | MKARG_A(0) | MKARG_B(0) | MKARG_C(argc)),
        (uint32_t)(MKOPCODE(OP_ABORT))
    };
    mrbc_irep irep = {		// where does this go?
        0,     				// nlv
        0,     				// nregs
        0,     				// rlen
        2,     				// ilen
        0,     				// plen
        (uint8_t *)code,   	// iseq
        (uint8_t *)sym,  	// ptr_to_sym
        NULL,  				// object pool
        NULL,  				// irep_list
    };

    mrbc_release(&v[0]);
    v[0] = new_obj;
    mrbc_dup(&new_obj);

    SET_RETURN(new_obj);
}

//================================================================
/*! (method) instance variable getter
 */
__GURU__
void c_object_getiv(mrbc_value v[], int argc)
{
    const char *name = mrbc_get_callee_name(NULL);

    mrbc_sym sym_id = name2symid(name);
    mrbc_value ret = mrbc_instance_getiv(&v[0], sym_id);

    SET_RETURN(ret);
}

//================================================================
/*! (method) instance variable setter
 */
__GURU__
void c_object_setiv(mrbc_value v[], int argc)
{
    const char *name = mrbc_get_callee_name(NULL);

    char *namebuf = (char *)mrbc_alloc(STRLEN(name));
    
    if (!namebuf) return;
    STRCPY(namebuf, name);
    namebuf[STRLEN(name)-1] = '\0';	// delete '='
    mrbc_sym sym_id = name2symid(namebuf);

    mrbc_instance_setiv(&v[0], sym_id, &v[1]);
    mrbc_free(namebuf);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__GURU__
void c_object_attr_reader(mrbc_value v[], int argc)
{
    for (int i = 1; i <= argc; i++) {
        if (v[i].tt != MRBC_TT_SYMBOL) continue;	// TypeError raise?

        // define reader method
        const char *name = VSYM(&v[i]);
        mrbc_define_method(v[0].cls, name, (mrbc_func_t)c_object_getiv);
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
__GURU__
void c_object_attr_accessor(mrbc_value v[], int argc)
{
    for (int i = 1; i <= argc; i++) {
        if (v[i].tt != MRBC_TT_SYMBOL) continue;	// TypeError raise?

        // define reader method
        char *name = VSYM(&v[i]);
        mrbc_define_method(v[0].cls, name, (mrbc_func_t)c_object_getiv);

        // make string "....=" and define writer method.
        char *namebuf = (char *)mrbc_alloc(STRLEN(name)+2);
        if (!namebuf) return;
        
        STRCPY(namebuf, name);
        STRCAT(namebuf, "=");
        mrbc_symbol_new(namebuf);
        mrbc_define_method(v[0].cls, namebuf, (mrbc_func_t)c_object_setiv);
        mrbc_free(namebuf);
    }
}

//================================================================
/*! (method) is_a, kind_of
 */
__GURU__
void c_object_kind_of(mrbc_value v[], int argc)
{
    int result = 0;
    if (v[1].tt != MRBC_TT_CLASS) {
        SET_BOOL_RETURN(result);
        return;
    }
    const mrbc_class *cls = mrbc_get_class_by_object(&v[0]);

    do {
        result = (cls == v[1].cls);
        if (result) break;

        cls = cls->super;
    } while (cls != NULL);
}

#if MRBC_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__
void c_object_to_s(mrbc_value v[], int argc)
{
	const char *str;

    switch (v->tt) {
    case MRBC_TT_CLASS:  str = symid2name(v->cls->sym_id); break;
    case MRBC_TT_OBJECT:
    	str = guru_sprintf("#<%s:%08x>", symid2name(v->cls->sym_id), (uintptr_t)v->self); break;
    default: str = ""; break;
    }
    SET_RETURN(mrbc_string_new_cstr(str));
}
#endif

__GURU__
void mrbc_init_class_object()
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
#if MRBC_USE_STRING
    mrbc_define_method(c, "inspect",       	c_object_to_s);
    mrbc_define_method(c, "to_s",          	c_object_to_s);
#endif
#ifdef MRBC_DEBUG
    mrbc_define_method(c, "p", 				c_p);
#endif
}

#if MRBC_USE_STRING
__GURU__
void c_proc_to_s(mrbc_value v[], int argc)
{
    char *str = guru_sprintf("<#Proc:%08x>", (uintptr_t)v->proc);

    SET_RETURN(mrbc_string_new_cstr(str));
}
#endif

__GURU__
void mrbc_init_class_proc()
{
    // Class
    mrbc_class *c = mrbc_class_proc = mrbc_define_class("Proc", mrbc_class_object);
    // Methods
    mrbc_define_method(c, "call", (mrbc_func_t)c_proc_call);
#if MRBC_USE_STRING
    mrbc_define_method(c, "inspect", 	c_proc_to_s);
    mrbc_define_method(c, "to_s", 		c_proc_to_s);
#endif
}

//================================================================
// Nil class

//================================================================
/*! (method) !
 */
__GURU__
void c_nil_false_not(mrbc_value v[], int argc)
{
    v[0].tt = MRBC_TT_TRUE;
}

#if MRBC_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__
void c_nil_inspect(mrbc_value v[], int argc)
{
    v[0] = mrbc_string_new_cstr("nil");
}

//================================================================
/*! (method) to_s
 */
__GURU__
void c_nil_to_s(mrbc_value v[], int argc)
{
    v[0] = mrbc_string_new(NULL, 0);
}
#endif

//================================================================
/*! Nil class
 */
__GURU__
void mrbc_init_class_nil()
{
    // Class
    mrbc_class *c = mrbc_class_nil = mrbc_define_class("NilClass", mrbc_class_object);
    // Methods
    mrbc_define_method(c, "!", 			c_nil_false_not);
#if MRBC_USE_STRING
    mrbc_define_method(c, "inspect", 	c_nil_inspect);
    mrbc_define_method(c, "to_s", 		c_nil_to_s);
#endif
}

//================================================================
// False class

#if MRBC_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__
void c_false_to_s(mrbc_value v[], int argc)
{
    v[0] = mrbc_string_new_cstr("false");
}
#endif

//================================================================
/*! False class
 */
__GURU__
void mrbc_init_class_false()
{
    // Class
    mrbc_class_false = mrbc_define_class("FalseClass", mrbc_class_object);
    // Methods
    mrbc_define_method(mrbc_class_false, "!", c_nil_false_not);
#if MRBC_USE_STRING
    mrbc_define_method(mrbc_class_false, "inspect", c_false_to_s);
    mrbc_define_method(mrbc_class_false, "to_s", c_false_to_s);
#endif
}

//================================================================
// True class

#if MRBC_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__
void c_true_to_s(mrbc_value v[], int argc)
{
    v[0] = mrbc_string_new_cstr("true");
}
#endif

__GURU__
void mrbc_init_class_true()
{
    // Class
    mrbc_class_true = mrbc_define_class("TrueClass", mrbc_class_object);
    // Methods
#if MRBC_USE_STRING
    mrbc_define_method(mrbc_class_true, "inspect", 	c_true_to_s);
    mrbc_define_method(mrbc_class_true, "to_s", 	c_true_to_s);
#endif
}

//================================================================
/*! initialize
 */
__GURU__ void mrbc_init_class_symbol()  // << from symbol.cu
{
    mrbc_class *c = mrbc_class_symbol = mrbc_define_class("Symbol", mrbc_class_object);

#if MRBC_USE_ARRAY
    mrbc_define_method(o, "all_symbols", 	c_all_symbols);
#endif
#if MRBC_USE_STRING
    mrbc_define_method(c, "inspect", 		c_inspect);
    mrbc_define_method(c, "to_s", 			c_to_s);
    mrbc_define_method(c, "id2name", 		c_to_s);
#endif
    mrbc_define_method(c, "to_sym", 		c_nop);
}

//================================================================
// initialize

__GURU__
void mrbc_init_class(void)
{
    mrbc_init_class_object();
    mrbc_init_class_nil();
    mrbc_init_class_proc();
    mrbc_init_class_false();
    mrbc_init_class_true();

    mrbc_init_class_symbol();
    mrbc_init_class_fixnum();
#if MRBC_USE_FLOAT
    mrbc_init_class_float();
#if MRBC_USE_MATH
    mrbc_init_class_math();
#endif
#endif
    
#if MRBC_USE_STRING
    mrbc_init_class_string();
#endif
#if MRBC_USE_ARRAY
    mrbc_init_class_array(0);
    mrbc_init_class_range(0);
    mrbc_init_class_hash(0);
#endif
}
