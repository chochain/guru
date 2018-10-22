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
#include "symbol.h"
#include "vmalloc.h"
#include "global.h"
#include "static.h"
#include "console.h"
#include "opcode.h"
#include "class.h"

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
/*! print - sub function
  @param  v	pointer to target value.
  @retval 0	normal return.
  @retval 1	already output LF.
*/
__GURU__
int mrbc_print_sub(mrbc_value *v)
{
    int ret = 0;

    switch (v->tt){
    case MRBC_TT_EMPTY:	 console_str("(empty)");					break;
    case MRBC_TT_NIL:					                			break;
    case MRBC_TT_FALSE:	 console_str("false");						break;
    case MRBC_TT_TRUE:	 console_str("true");						break;
    case MRBC_TT_FIXNUM: console_int(v->i);							break;
#if MRBC_USE_FLOAT
    case MRBC_TT_FLOAT:  console_float(v->f);						break;
#endif
    case MRBC_TT_SYMBOL: console_str(VSYM(v)); 						break;
    case MRBC_TT_CLASS:  console_str(symid_to_str(v->cls->sym_id)); break;
    case MRBC_TT_OBJECT:
        console_str(
            guru_vprintf("#<%s:0x%x>",
            symid_to_str(find_class_by_object(v)->sym_id),
                         (mrbc_int)v->instance));
        break;
    case MRBC_TT_PROC:   console_str("#<Proc:0x%x>", (mrbc_int)v->proc)); break;
#if MRBC_USE_STRING
    case MRBC_TT_STRING:
        console_str(VSTR(v));
        if (VSTRLEN(v) != 0 && VSTR(v)[VSTRLEN(v) - 1]=='\n') {
        	ret = 1;
        }
        break;
#endif
#if MRBC_USE_ARRAY
    case MRBC_TT_ARRAY: {
        console_char('[');
        for (int i = 0; i < mrbc_array_size(v); i++) {
            if (i != 0) console_str(", ");
            mrbc_value v1 = mrbc_array_get(v, i);
            mrbc_p_sub(&v1);
        }
        console_char(']');
    } break;
    case MRBC_TT_RANGE:{
        mrbc_value v1 = mrbc_range_first(v);
        mrbc_print_sub(&v1);
        console_str(mrbc_range_exclude_end(v) ? "..." : "..");
        v1 = mrbc_range_last(v);
        mrbc_print_sub(&v1);
    } break;
    case MRBC_TT_HASH:{
        console_char('{');
        mrbc_hash_iterator ite = mrbc_hash_iterator_new(v);
        while (mrbc_hash_i_has_next(&ite)) {
            mrbc_value *vk = mrbc_hash_i_next(&ite);
            mrbc_p_sub(vk);
            console_str("=>");
            mrbc_p_sub(vk+1);
            if (mrbc_hash_i_has_next(&ite)) console_str(", ");
        }
        console_char('}');
    } break;
#endif
    default:
    	console_str("Not support MRBC_TT_XX: ");
    	console_int((mrbc_int)v->tt);
    	break;
    }
    return ret;
}

//================================================================
/*! puts - sub function

  @param  v	pointer to target value.
  @retval 0	normal return.
  @retval 1	already output LF.
*/
__GURU__
int mrbc_puts_sub(mrbc_value *v)
{
    if (v->tt == MRBC_TT_ARRAY) {
#if MRBC_USE_ARRAY
        for (int i = 0; i < mrbc_array_size(v); i++) {
            if (i != 0) console_char('\n');
            mrbc_value v1 = mrbc_array_get(v, i);
            mrbc_puts_sub(&v1);
        }
#endif
        return 0;
    }
    return mrbc_print_sub(v);
}

//================================================================
/*! p - sub function
 */
__GURU__
int mrbc_p_sub(mrbc_value *v)
{
    switch (v->tt){
    case MRBC_TT_NIL: console_str("nil");		break;
    case MRBC_TT_SYMBOL:{
        const char *s   = VSYM(v);
        const char *fmt = STRCHR(s, ':') ? "\":%s\"" : ":%s";
        console_strf(s, fmt);
    } break;

#if MRBC_USE_STRING
    case MRBC_TT_STRING:{
        console_char('"');
        const char *s = VSTR(v);

        for (int i = 0; i < VSTRLEN(v); i++) {
            if (s[i] < ' ' || 0x7f <= s[i]) {		// tiny isprint()
                console_hex(s[i]);
            } else {
                console_char(s[i]);
            }
        }
        console_char('"');
    } break;
#endif
#if MRBC_USE_ARRAY
    case MRBC_TT_RANGE:{
        mrbc_value v1 = mrbc_range_first(v);
        mrbc_p_sub(&v1);
        console_str(mrbc_range_exclude_end(v) ? "..." : "..");
        v1 = mrbc_range_last(v);
        mrbc_p_sub(&v1);
    } break;
#endif
    default:
        mrbc_print_sub(v);
        break;
    }
    return 0;
}

//================================================================
/*!@brief
  find class by object

  @param  vm
  @param  obj
  @return pointer to mrbc_class
*/
__GURU__
mrbc_class *find_class_by_object(mrbc_object *obj)
{
    mrbc_class *cls;

    switch (obj->tt) {
    case MRBC_TT_TRUE:	  cls = mrbc_class_true;		break;
    case MRBC_TT_FALSE:	  cls = mrbc_class_false; 	    break;
    case MRBC_TT_NIL:	  cls = mrbc_class_nil;		    break;
    case MRBC_TT_FIXNUM:  cls = mrbc_class_fixnum;	    break;
#if MRBC_USE_FLOAT
    case MRBC_TT_FLOAT:	  cls = mrbc_class_float; 	    break;
#endif
    case MRBC_TT_SYMBOL:  cls = mrbc_class_symbol;	    break;

    case MRBC_TT_OBJECT:  cls = obj->instance->cls;     break;
    case MRBC_TT_CLASS:   cls = obj->cls;               break;
    case MRBC_TT_PROC:	  cls = mrbc_class_proc;		break;
#if MRBC_USE_STRING
    case MRBC_TT_STRING:  cls = mrbc_class_string;	    break;
#endif
#if MRBC_USE_ARRAY
    case MRBC_TT_ARRAY:   cls = mrbc_class_array; 	    break;
    case MRBC_TT_RANGE:	  cls = mrbc_class_range; 	    break;
    case MRBC_TT_HASH:	  cls = mrbc_class_hash;		break;
#endif
    default:		      cls = mrbc_class_object;	    break;
    }
    return cls;
}

//================================================================
/*!@brief
  find method from

  @param  vm
  @param  recv
  @param  sym_id
  @return
*/
__GURU__
mrbc_proc *find_method(mrbc_value recv, mrbc_sym sym_id)
{
    mrbc_class *cls = find_class_by_object(&recv);

    while (cls != 0) {
        mrbc_proc *proc = cls->procs;
        while (proc != 0) {
            if (proc->sym_id == sym_id) {
                return proc;
            }
            proc = proc->next;
        }
        cls = cls->super;
    }
    return 0;
}

//================================================================
/*!@brief
  define class

  @param  vm		pointer to vm.
  @param  name		class name.
  @param  super		super class.
*/
__GURU__
mrbc_class * mrbc_define_class(const char *name, mrbc_class *super)
{
    if (super == NULL) super = mrbc_class_object;  // set default to Object.

    mrbc_class *cls = mrbc_get_class_by_name(name);
    if (cls) return cls;

    // create a new class?
    cls = (mrbc_class *)mrbc_alloc(sizeof(mrbc_class));
    if (!cls) return NULL;			// ENOMEM

    mrbc_sym sym_id = str_to_symid(name);

    cls->sym_id = sym_id;
    cls->super 	= super;
    cls->procs 	= 0;
#ifdef MRBC_DEBUG
    cls->names 	= name;			// for debug; delete soon.
#endif

    // register to global constant.
    mrbc_value v = { .tt = MRBC_TT_CLASS };
    v.cls = cls;
    const_object_add(sym_id, &v);

    return cls;
}

//================================================================
/*! get class by name

  @param  name		class name.
  @return		pointer to class object.
*/
__GURU__
mrbc_class * mrbc_get_class_by_name(const char *name)
{
    mrbc_sym sym_id = str_to_symid(name);
    mrbc_object obj = const_object_get(sym_id);

    return (obj.tt == MRBC_TT_CLASS) ? obj.cls : NULL;
}

//================================================================
/*!@brief
  define class method or instance method.

  @param  vm		pointer to vm.
  @param  cls		pointer to class.
  @param  name		method name.
  @param  cfunc		pointer to function.
*/
__GURU__
void mrbc_define_method(mrbc_class *cls, const char *name, mrbc_func_t cfunc)
{
    if (cls==NULL) cls = mrbc_class_object;	// set default to Object.

    mrbc_proc *proc = mrbc_proc_alloc(name);

    proc->c_func	= 1;  			// c-func
    proc->func 		= cfunc;
    proc->next 		= cls->procs;

    cls->procs 		= proc;
}

// Call a method
// v[0]: receiver
// v[1..]: params
//================================================================
/*!@brief
  call a method with params

  @param  vm		pointer to vm
  @param  name		method name
  @param  v		receiver and params
  @param  argc		num of params
*/
__GURU__
void mrbc_funcall(mrbc_vm *vm, const char *name, mrbc_value *v, int argc)
{
    mrbc_sym  sym_id = str_to_symid(name);
    mrbc_proc *m     = find_method(v[0], sym_id);

    if (m==0) return;   	// no method

    mrbc_callinfo *ci = (mrbc_callinfo *)mrbc_alloc(sizeof(mrbc_callinfo));

    ci->reg		= vm->reg;
    ci->pc_irep = vm->pc_irep;
    ci->pc   	= vm->pc;
    ci->argc 	= 0;
    ci->klass	= vm->klass;

    ci->prev = vm->calltop;	// push call stack
    vm->calltop = ci;

    vm->pc_irep = m->irep;	// target irep
    vm->pc 		= 0;
    // new regs
    vm->reg 	+= 2;   	// recv and symbol
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
mrbc_value mrbc_send(mrbc_value *v, int reg_ofs,
                      mrbc_value *recv, const char *method, int argc, ...)
{
    mrbc_sym  sym_id = str_to_symid(method);
    mrbc_proc *m     = find_method(*recv, sym_id);
    mrbc_value *regs = v + reg_ofs + 2;

    if (m == 0) {
        console_str("No method. vtype=");
        console_int(recv->tt);
        console_str(" method='");
        console_str(method);
        console_str("'\n");
        return mrbc_nil_value();
    }
    if (!m->c_func) {
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

//================================================================
// Object class
//================================================================
/*! Nop operator / method
 */
__GURU__
void c_nop(mrbc_value v[], int argc)
{
    // nothing to do.
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
        mrbc_value result = mrbc_send(v, argc, &v[1], "kind_of?", 1, &v[0]);
        SET_RETURN(result);

    } else {
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
    value.cls = find_class_by_object(v);
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
        0,     // nlocals
        0,     // nregs
        0,     // rlen
        2,     // ilen
        0,     // plen
        (uint8_t *)code,   	// iseq
        (uint8_t *)sym,  	// ptr_to_sym
        NULL,  // pools
        NULL,  // reps
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

    mrbc_sym sym_id = str_to_symid(name);
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
    mrbc_sym sym_id = str_to_symid(namebuf);

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
    const mrbc_class *cls = find_class_by_object(&v[0]);

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
    case MRBC_TT_CLASS:  str = symid_to_str(v->cls->sym_id);						 break;
    case MRBC_TT_OBJECT: str = guru_vprintf("#<%s:%08x>",v, (uintptr_t)v->instance); break;
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

// =============== ProcClass
__GURU__
void c_proc_call(mrbc_value v[], int argc)
{
    // push callinfo, but not release regs
    mrbc_push_callinfo(NULL, argc);
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
