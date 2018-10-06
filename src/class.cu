/*! @file
  @brief
  Guru Object, Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include "vm_config.h"
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
#include "vm.h"
#include "class.h"

#if MRBC_USE_STRING
#include "c_string.h"
#endif

#if MRBC_USE_ARRAY
#include "c_array.h"
#include "c_hash.h"
#include "c_numeric.h"
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
    case MRBC_TT_EMPTY:	    console_print("(empty)");	break;
    case MRBC_TT_NIL:					                break;
    case MRBC_TT_FALSE:	    console_print("false");		break;
    case MRBC_TT_TRUE:	    console_print("true");		break;
    case MRBC_TT_FIXNUM:	console_printf("%d", v->i);	break;
#if MRBC_USE_FLOAT
    case MRBC_TT_FLOAT:    console_printf("%g", v->d);	break;
#endif
    case MRBC_TT_SYMBOL:
        console_print(mrbc_symbol_cstr(v));             break;
    case MRBC_TT_CLASS:
        console_print(symid_to_str(v->cls->sym_id));    break;
    case MRBC_TT_OBJECT:
        console_printf("#<%s:", symid_to_str(find_class_by_object(v)->sym_id));
        console_printf("%08x>", v->instance);
        break;

    case MRBC_TT_PROC:
        console_printf("#<Proc:%08x>", v->proc);      break;

#if MRBC_USE_STRING
    case MRBC_TT_STRING:
        console_nprint(mrbc_string_cstr(v), mrbc_string_size(v));
        if (mrbc_string_size(v) != 0 &&
            mrbc_string_cstr(v)[ mrbc_string_size(v) - 1 ] == '\n') ret = 1;
        break;
#endif
#if MRBC_USE_ARRAY
    case MRBC_TT_ARRAY:{
        console_putchar('[');
        int i;
        for(i = 0; i < mrbc_array_size(v); i++) {
            if (i != 0) console_print(", ");
            mrbc_value v1 = mrbc_array_get(v, i);
            mrbc_p_sub(&v1);
        }
        console_putchar(']');
    } break;
    case MRBC_TT_RANGE:{
        mrbc_value v1 = mrbc_range_first(v);
        mrbc_print_sub(&v1);
        console_print(mrbc_range_exclude_end(v) ? "..." : "..");
        v1 = mrbc_range_last(v);
        mrbc_print_sub(&v1);
    } break;
    case MRBC_TT_HASH:{
        console_putchar('{');
        mrbc_hash_iterator ite = mrbc_hash_iterator_new(v);
        while(mrbc_hash_i_has_next(&ite)) {
            mrbc_value *vk = mrbc_hash_i_next(&ite);
            mrbc_p_sub(vk);
            console_print("=>");
            mrbc_p_sub(vk+1);
            if (mrbc_hash_i_has_next(&ite)) console_print(", ");
        }
        console_putchar('}');
    } break;
#endif
    default:
        console_printf("Not support MRBC_TT_XX(%d)", v->tt);
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
#if MRBC_USE_ARRAY
    if (v->tt == MRBC_TT_ARRAY) {
        int i;
        for(i = 0; i < mrbc_array_size(v); i++) {
            if (i != 0) console_putchar('\n');
            mrbc_value v1 = mrbc_array_get(v, i);
            mrbc_puts_sub(&v1);
        }
        return 0;
    }
    return mrbc_print_sub(v);
#else
    return 0;
#endif
}

//================================================================
/*! p - sub function
 */
__GURU__
int mrbc_p_sub(mrbc_value *v)
{
    switch (v->tt){
    case MRBC_TT_NIL:
        console_print("nil");
        break;

    case MRBC_TT_SYMBOL:{
        const char *s   = mrbc_symbol_cstr(v);
        const char *fmt = STRCHR(s, ':') ? "\":%s\"" : ":%s";
        console_printf(fmt, s);
    } break;

#if MRBC_USE_STRING
    case MRBC_TT_STRING:{
        console_putchar('"');
        const char *s = mrbc_string_cstr(v);
        int i;
        for(i = 0; i < mrbc_string_size(v); i++) {
            if (s[i] < ' ' || 0x7f <= s[i]) {	// tiny isprint()
                console_printf("\\x%02x", s[i]);
            } else {
                console_putchar(s[i]);
            }
        }
        console_putchar('"');
    } break;
#endif
#if MRBC_USE_ARRAY
    case MRBC_TT_RANGE:{
        mrbc_value v1 = mrbc_range_first(v);
        mrbc_p_sub(&v1);
        console_print(mrbc_range_exclude_end(v) ? "..." : "..");
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

    while(cls != 0) {
        mrbc_proc *proc = cls->procs;
        while(proc != 0) {
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

    mrbc_sym sym_id = str_to_symid(name);
    mrbc_object obj = const_object_get(sym_id);

    // create a new class?
    if (obj.tt == MRBC_TT_NIL) {
        mrbc_class *cls = (mrbc_class *)mrbc_alloc(sizeof(mrbc_class));
        if (!cls) return cls;	// ENOMEM

        cls->sym_id = sym_id;
#ifdef MRBC_DEBUG
        cls->names = name;	// for debug; delete soon.
#endif
        cls->super = super;
        cls->procs = 0;

        // register to global constant.
        mrbc_value v = {.tt = MRBC_TT_CLASS};
        v.cls = cls;
        const_object_add(sym_id, &v);

        return cls;
    }

    // already?
    if (obj.tt == MRBC_TT_CLASS) {
        return obj.cls;
    }

    // error.
    assert(1==0);   // raise TypeError
    return NULL;
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
    if (cls == NULL) cls = mrbc_class_object;	// set default to Object.

    mrbc_proc *rproc = mrbc_rproc_alloc(name);
    rproc->c_func = 1;  // c-func
    rproc->next = cls->procs;
    cls->procs = rproc;
    rproc->func = cfunc;
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
void mrbc_funcall(struct VM *vm, const char *name, mrbc_value *v, int argc)
{
    mrbc_sym sym_id = str_to_symid(name);
    mrbc_proc *m = find_method(v[0], sym_id);

    if (m==0) return;   // no method

    mrbc_callinfo *callinfo = (mrbc_callinfo *)mrbc_alloc(sizeof(mrbc_callinfo));
    callinfo->current_regs = vm->current_regs;
    callinfo->pc_irep = vm->pc_irep;
    callinfo->pc = vm->pc;
    callinfo->n_args = 0;
    callinfo->target_class = vm->target_class;
    callinfo->prev = vm->callinfo_tail;
    vm->callinfo_tail = callinfo;

    // target irep
    vm->pc = 0;
    vm->pc_irep = m->irep;

    // new regs
    vm->current_regs += 2;   // recv and symbol
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
    mrbc_sym sym_id = str_to_symid(method);
    mrbc_proc *m = find_method(*recv, sym_id);
    mrbc_value *regs = v + reg_ofs + 2;

    if (m == 0) {
        console_printf("No method. vtype=%d method='%s'\n", recv->tt, method);
        return mrbc_nil_value();
    }
    if (!m->c_func) {
        console_printf("Method %s is not C function\n", method);
        return mrbc_nil_value();
    }

    // create call stack.
    mrbc_release(&regs[0]);
    regs[0] = *recv;
    mrbc_dup(recv);

    va_list ap;
    va_start(ap, argc);
    int i;
    for(i = 1; i <= argc; i++) {
        mrbc_release(&regs[i]);
        regs[i] = *va_arg(ap, mrbc_value *);
    }
    mrbc_release(&regs[i]);
    regs[i] = mrbc_nil_value();
    va_end(ap);

    // call method.
    m->func(regs, argc);
    mrbc_value ret = regs[0];

    for(; i >= 0; i--) {
        regs[i].tt = MRBC_TT_EMPTY;
    }
    return ret;
}

//================================================================
// Object class

#ifdef MRBC_DEBUG
//================================================================
/*! (method) p
 */
__GURU__
void c_p(mrbc_value v[], int argc)
{
    int i;
    for(i = 1; i <= argc; i++) {
        mrbc_p_sub(&v[i]);
        console_putchar('\n');
    }
}
#endif

//================================================================
/*! (method) puts
 */
__GURU__
void c_puts(mrbc_value v[], int argc)
{
    int i;
    if (argc){
        for(i = 1; i <= argc; i++) {
            if (mrbc_puts_sub(&v[i]) == 0) console_putchar('\n');
        }
    } else {
        console_putchar('\n');
    }
}

//================================================================
/*! (method) print
 */
__GURU__
void c_print(mrbc_value v[], int argc)
{
    int i;
    for(i = 1; i <= argc; i++) {
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
void c_object_equal3(struct VM *vm, mrbc_value v[], int argc)
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
void c_object_new(struct VM *vm, mrbc_value v[], int argc)
{
    mrbc_value new_obj = mrbc_instance_new(v->cls, 0);

    char syms[]="______initialize";
    uint32_to_bin(1,(uint8_t*)&syms[0]);
    uint16_to_bin(10,(uint8_t*)&syms[4]);

    uint32_t code[2] = {
        (uint32_t)(MKOPCODE(OP_SEND) | MKARG_A(0) | MKARG_B(0) | MKARG_C(argc)),
        (uint32_t)MKOPCODE(OP_ABORT)
    };
    mrbc_irep irep = {
        0,     // nlocals
        0,     // nregs
        0,     // rlen
        2,     // ilen
        0,     // plen
        (uint8_t *)code,   // iseq
        NULL,  // pools
        (uint8_t *)syms,  // ptr_to_sym
        NULL,  // reps
    };

    mrbc_release(&v[0]);
    v[0] = new_obj;
    mrbc_dup(&new_obj);

    mrbc_irep *org_pc_irep = vm->pc_irep;
    uint16_t  org_pc = vm->pc;
    mrbc_value* org_regs = vm->current_regs;
    vm->pc = 0;
    vm->pc_irep = &irep;
    vm->current_regs = v;

    mrbc_vm_run(vm);

    vm->pc = org_pc;
    vm->pc_irep = org_pc_irep;
    vm->current_regs = org_regs;

    SET_RETURN(new_obj);
}

//================================================================
/*! (method) instance variable getter
 */
__GURU__
void c_object_getiv(struct VM *vm, mrbc_value v[], int argc)
{
    const char *name = mrbc_get_callee_name(vm);
    mrbc_sym sym_id = str_to_symid(name);
    mrbc_value ret = mrbc_instance_getiv(&v[0], sym_id);

    SET_RETURN(ret);
}

//================================================================
/*! (method) instance variable setter
 */
__GURU__
void c_object_setiv(struct VM *vm, mrbc_value v[], int argc)
{
    const char *name = mrbc_get_callee_name(vm);

    char *namebuf = (char *)mrbc_alloc(STRLEN(name));
    
    if (!namebuf) return;
    STRCPY(namebuf, name);
    namebuf[STRLEN(name)-1] = '\0';	// delete '='
    mrbc_sym sym_id = str_to_symid(namebuf);

    mrbc_instance_setiv(&v[0], sym_id, &v[1]);
    mrbc_raw_free(namebuf);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__GURU__
void c_object_attr_reader(mrbc_value v[], int argc)
{
    int i;
    for(i = 1; i <= argc; i++) {
        if (v[i].tt != MRBC_TT_SYMBOL) continue;	// TypeError raise?

        // define reader method
        const char *name = mrbc_symbol_cstr(&v[i]);
        mrbc_define_method(v[0].cls, name, (mrbc_func_t)c_object_getiv);
    }
}

//================================================================
/*! (class method) access method 'attr_accessor'
 */
__GURU__
void c_object_attr_accessor(mrbc_value v[], int argc)
{
    int i;
    for(i = 1; i <= argc; i++) {
        if (v[i].tt != MRBC_TT_SYMBOL) continue;	// TypeError raise?

        // define reader method
        char *name = (char *)mrbc_symbol_cstr(&v[i]);
        mrbc_define_method(v[0].cls, name, (mrbc_func_t)c_object_getiv);

        // make string "....=" and define writer method.
        char *namebuf = (char *)mrbc_alloc(STRLEN(name)+2);
        if (!namebuf) return;
        
        STRCPY(namebuf, name);
        STRCAT(namebuf, "=");
        mrbc_symbol_new(namebuf);
        mrbc_define_method(v[0].cls, namebuf, (mrbc_func_t)c_object_setiv);
        mrbc_raw_free(namebuf);
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
    } while(cls != NULL);
}

#if MRBC_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__
void c_object_to_s(mrbc_value v[], int argc)
{
    char buf[32];
    const char *s = buf;

    switch (v->tt) {
    case MRBC_TT_CLASS:
        s = symid_to_str(v->cls->sym_id);
        break;

    case MRBC_TT_OBJECT:{
        // (NOTE) address part assumes 32bit. but enough for this.
        mrbc_printf pf;

        mrbc_printf_init(&pf, buf, sizeof(buf), "#<%s:%08x>");
        while(mrbc_printf_main(&pf) > 0) {
            switch (pf.fmt.type) {
            case 's':
                mrbc_printf_str(&pf, symid_to_str(v->instance->cls->sym_id), ' ');
                break;
            case 'x':
                mrbc_printf_int(&pf, (uintptr_t)v->instance, 16);
                break;
            }
        }
        mrbc_printf_end(&pf);
    } break;

    default:
        s = "";
        break;
    }

    SET_RETURN(mrbc_string_new_cstr(vm, s));
}
#endif


#ifdef MRBC_DEBUG
__GURU__
void c_object_instance_methods(mrbc_value v[], int argc)
{
    // TODO: check argument.

    // temporary code for operation check.
    console_printf("[");
    int flag_first = 1;

    mrbc_class *cls = find_class_by_object(vm, v);
    mrbc_proc *proc = cls->procs;
    while(proc) {
        console_printf("%s:%s", (flag_first ? "" : ", "),
                        symid_to_str(proc->sym_id));
        flag_first = 0;
        proc = proc->next;
    }

    console_printf("]");

    SET_NIL_RETURN();
}
#endif

__GURU__
void mrbc_init_class_object(struct VM *vm)
{
    // Class
    mrbc_class_object = mrbc_define_class("Object",        0);
    // Methods
    mrbc_define_method(mrbc_class_object, "initialize",    c_ineffect);
    mrbc_define_method(mrbc_class_object, "puts",          c_puts);
    mrbc_define_method(mrbc_class_object, "print",         c_print);
    mrbc_define_method(mrbc_class_object, "!",             c_object_not);
    mrbc_define_method(mrbc_class_object, "!=",            c_object_neq);
    mrbc_define_method(mrbc_class_object, "<=>",           c_object_compare);
    mrbc_define_method(mrbc_class_object, "===",           (mrbc_func_t)c_object_equal3);
    mrbc_define_method(mrbc_class_object, "class",         c_object_class);
    mrbc_define_method(mrbc_class_object, "new",           (mrbc_func_t)c_object_new);
    mrbc_define_method(mrbc_class_object, "attr_reader",   c_object_attr_reader);
    mrbc_define_method(mrbc_class_object, "attr_accessor", c_object_attr_accessor);
    mrbc_define_method(mrbc_class_object, "is_a?",         c_object_kind_of);
    mrbc_define_method(mrbc_class_object, "kind_of?",      c_object_kind_of);

#if MRBC_USE_STRING
    mrbc_define_method(mrbc_class_object, "inspect",       c_object_to_s);
    mrbc_define_method(mrbc_class_object, "to_s",          c_object_to_s);
#endif

#ifdef MRBC_DEBUG
    mrbc_define_method(mrbc_class_object, "instance_methods", c_object_instance_methods);
    mrbc_define_method(mrbc_class_object, "p", c_p);
#endif
}

// =============== ProcClass
__GURU__
void c_proc_call(struct VM *vm, mrbc_value v[], int argc)
{
    // push callinfo, but not release regs
    mrbc_push_callinfo(vm, argc);

    // target irep
    vm->pc = 0;
    vm->pc_irep = v[0].proc->irep;

    vm->current_regs = v;
}


#if MRBC_USE_STRING
__GURU__
void c_proc_to_s(mrbc_value v[], int argc)
{
    // (NOTE) address part assumes 32bit. but enough for this.
    char buf[32];
    mrbc_printf pf;

    mrbc_printf_init(&pf, buf, sizeof(buf), "<#Proc:%08x>");
    while(mrbc_printf_main(&pf) > 0) {
        mrbc_printf_int(&pf, (uintptr_t)v->proc, 16);
    }
    mrbc_printf_end(&pf);

    SET_RETURN(mrbc_string_new_cstr(vm, buf));
}
#endif

__GURU__
void mrbc_init_class_proc(struct VM *vm)
{
    // Class
    mrbc_class_proc= mrbc_define_class("Proc", mrbc_class_object);
    // Methods
    mrbc_define_method(mrbc_class_proc, "call", (mrbc_func_t)c_proc_call);
#if MRBC_USE_STRING
    mrbc_define_method(mrbc_class_proc, "inspect", c_proc_to_s);
    mrbc_define_method(mrbc_class_proc, "to_s", c_proc_to_s);
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
void mrbc_init_class_nil(struct VM *vm)
{
    // Class
    mrbc_class_nil = mrbc_define_class("NilClass", mrbc_class_object);
    // Methods
    mrbc_define_method(mrbc_class_nil, "!", c_nil_false_not);
#if MRBC_USE_STRING
    mrbc_define_method(mrbc_class_nil, "inspect", c_nil_inspect);
    mrbc_define_method(mrbc_class_nil, "to_s", c_nil_to_s);
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
    mrbc_define_method(mrbc_class_true, "inspect", c_true_to_s);
    mrbc_define_method(mrbc_class_true, "to_s", c_true_to_s);
#endif
}

//================================================================
/*! Ineffect operator / method
 */
__GURU__
void c_ineffect(mrbc_value v[], int argc)
{
    // nothing to do.
}

//================================================================
// initialize

__GURU__
void mrbc_init_class(void)
{
    mrbc_init_class_object(0);
    mrbc_init_class_nil(0);
    mrbc_init_class_proc(0);
    mrbc_init_class_false();
    mrbc_init_class_true();

    mrbc_init_class_symbol();
#if MRBC_USE_FLOAT
    mrbc_init_class_fixnum();
    mrbc_init_class_float();
#if MRBC_USE_MATH
    mrbc_init_class_math();
#endif
#endif
    
#if MRBC_USE_STRING
    mrbc_init_class_string(0);
#endif
#if MRBC_USE_ARRAY
    mrbc_init_class_array(0);
    mrbc_init_class_range(0);
    mrbc_init_class_hash(0);
#endif
}

//================================================================
/*! initialize
 */
__GURU__ void mrbc_init_class_symbol()  // << from symbol.cu
{
    mrbc_class_symbol = mrbc_define_class("Symbol", mrbc_class_object);

#if MRBC_USE_ARRAY
    mrbc_define_method(mrbc_class_symbol, "all_symbols", c_all_symbols);
#endif
#if MRBC_USE_STRING
    mrbc_define_method(mrbc_class_symbol, "inspect", c_inspect);
    mrbc_define_method(mrbc_class_symbol, "to_s", c_to_s);
    mrbc_define_method(mrbc_class_symbol, "id2name", c_to_s);
#endif
    mrbc_define_method(mrbc_class_symbol, "to_sym", c_ineffect);
}


