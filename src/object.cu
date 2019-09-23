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
#include "global.h"
#include "vm.h"
#include "class.h"

#include "ucode.h"
#include "object.h"
#include "ostore.h"

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
    GV *regs = v + 2;	     		// allocate 2 for stack
    GS sid   = name2id(method);

    guru_proc *m = proc_by_sid(rcv, sid);	// method of receiver object

    if (m==0) {
        PRINTF("No method. vtype=%d method='%s'\n", rcv->gt, method);
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
    GV ret = regs[0];				// copy result

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

__GURU__ void
guru_obj_del(GV *v)
{
	assert(v->gt==GT_OBJ);

	ostore_del(v);
}

//================================================================
__GURU__ void
obj_nop(GV v[], U32 argc)
{
	// do nothing
}

#if GURU_DEBUG
//================================================================
/*! (method) p
 */
__GURU__ void
obj_p(GV v[], U32 argc)
{
	guru_p(v, argc);
}
#endif

//================================================================
/*! (method) puts
 */
__GURU__ void
obj_puts(GV v[], U32 argc)
{
	guru_puts(v, argc);
}

//================================================================
/*! (method) print
 */
__GURU__ void
obj_print(GV v[], U32 argc)
{
	guru_puts(v, argc);
}

//================================================================
/*! (operator) !
 */
__GURU__ void
obj_not(GV v[], U32 argc)
{
    RETURN_FALSE();
}

//================================================================
/*! (operator) !=
 */
__GURU__ void
obj_neq(GV v[], U32 argc)
{
    S32 t = guru_cmp(&v[0], &v[1]);
    RETURN_BOOL(t);
}

//================================================================
/*! (operator) <=>
 */
__GURU__ void
obj_cmp(GV v[], U32 argc)
{
    S32 t = guru_cmp(&v[0], &v[1]);
    RETURN_INT(t);
}

//================================================================
/*! (operator) ===
 */
__GURU__ void
obj_eq3(GV v[], U32 argc)
{
    if (v[0].gt != GT_CLASS) {
    	RETURN_BOOL(guru_cmp(v, v+1)==0);
    }
    else {
    	GV ret = guru_kind_of(v, argc);
    	RETURN_VAL(ret);
    }
}

//================================================================
/*! (method) class
 */
__GURU__ void
obj_class(GV v[], U32 argc)
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
obj_getiv(GV v[], U32 argc)
{
    const U8P name = _get_callee(NULL);			// TODO:
    GS  sid  = name2id(name);

    RETURN_VAL(ostore_get(&v[0], sid));
}

//================================================================
/*! (method) instance variable setter
 */
__GURU__ void
obj_setiv(GV v[], U32 argc)
{
    U8P name = _get_callee(NULL);			// CC TODO: another way
    GS  sid  = name2id(name);

    ostore_set(&v[0], sid, &v[1]);
}

//================================================================
/*! (class method) access method 'attr_reader'
 */
__GURU__ void
obj_attr_reader(GV v[], U32 argc)
{
    for (U32 i = 1; i <= argc; i++) {
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
obj_attr_accessor(GV v[], U32 argc)
{
    for (U32 i = 1; i <= argc; i++) {
        if (v[i].gt != GT_SYM) continue;				// TypeError raise?

        // define reader method
        U8P name = id2name(v[i].i);
        guru_define_method(v[0].cls, name, obj_getiv);

        U32 sz = STRLEN(name);

        // make string "....=" and define writer method.
        // TODO: consider using static buffer
        U8P buf = (U8P)guru_alloc(sz + (-sz & 7));		// 8-byte aligned
        
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
obj_kind_of(GV v[], U32 argc)
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

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
obj_to_s(GV v[], U32 argc)
{
	GV  ret;
	U8  *name;
	GV  iv[2] = { { .gt=GT_INT }, { .gt=GT_INT } };

    switch (v->gt) {
    case GT_CLASS:
    	name = id2name(v->cls->sid);
    	ret = guru_str_new(name);
    	break;
    case GT_OBJ:
    	iv[1].i = 16;
    	iv[0].i = (U32A)v->self;
    	int_to_s(iv, 1);

    	name = id2name(v->self->cls->sid);
    	ret  = guru_str_new("#<");
    	guru_str_append_cstr(&ret, name);
    	guru_str_append_cstr(&ret, ":0x");
    	guru_str_append_cstr(&ret, (U8*)iv[0].str->data);
    	guru_str_append_cstr(&ret, ">");

    	ref_clr(&iv[0]);
    	break;
    default:
    	ret = guru_str_new("");
    	break;
    }
    RETURN_VAL(ret);
}
#endif

__GURU__ void
obj_new(GV v[], U32 argc)
{
	assert(1==0);				// handled in ucode
}

__GURU__ void
_init_class_object()
{
    // Class
    guru_class *c = guru_class_object = NEW_CLASS("Object", NULL);

    // Methods
    NEW_PROC("initialize",    	obj_nop);
#if GURU_DEBUG
    NEW_PROC("p", 				obj_p);
#endif
    NEW_PROC("puts",          	obj_puts);
    NEW_PROC("print",         	obj_print);
    NEW_PROC("!",             	obj_not);
    NEW_PROC("!=",            	obj_neq);
    NEW_PROC("<=>",           	obj_cmp);
    NEW_PROC("===",           	obj_eq3);
    NEW_PROC("class",         	obj_class);
    NEW_PROC("new",           	obj_new);
    NEW_PROC("attr_reader",   	obj_attr_reader);
    NEW_PROC("attr_accessor", 	obj_attr_accessor);
    NEW_PROC("is_a?",         	obj_kind_of);
    NEW_PROC("kind_of?",      	obj_kind_of);
#if GURU_USE_STRING
    NEW_PROC("inspect",       	obj_to_s);
    NEW_PROC("to_s",          	obj_to_s);
#endif
}

//================================================================
// ProcClass
//================================================================
__GURU__ void
prc_call(GV v[], U32 argc)
{
	// not suppose to come here
	assert(1==0);		// taken care by vm#op_send
}

#if GURU_USE_STRING
__GURU__ void
prc_inspect(GV v[], U32 argc)
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
    guru_class *c = guru_class_proc = NEW_CLASS("Proc", guru_class_object);
    // Methods
    NEW_PROC("call", 	prc_call);
#if GURU_USE_STRING
    NEW_PROC("inspect", prc_inspect);
    NEW_PROC("to_s", 	prc_inspect);
#endif
}

//================================================================
// Nil class

//================================================================
/*! (method) !
 */
__GURU__ void
nil_false_not(GV v[], U32 argc)
{
    v[0].gt = GT_TRUE;
}

#if GURU_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__ void
nil_inspect(GV v[], U32 argc)
{
    v[0] = guru_str_new("nil");
}

//================================================================
/*! (method) to_s
 */
__GURU__ void
nil_to_s(GV v[], U32 argc)
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
    guru_class *c = guru_class_nil = NEW_CLASS("NilClass", guru_class_object);
    // Methods
    NEW_PROC("!", 			nil_false_not);
#if GURU_USE_STRING
    NEW_PROC("inspect", 	nil_inspect);
    NEW_PROC("to_s", 		nil_to_s);
#endif
}

//================================================================
// False class

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
false_to_s(GV v[], U32 argc)
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
    guru_class *c = guru_class_false = NEW_CLASS("FalseClass", guru_class_object);
    // Methods
    NEW_PROC("!", 		nil_false_not);
#if GURU_USE_STRING
    NEW_PROC("inspect", false_to_s);
    NEW_PROC("to_s",    false_to_s);
#endif
}

//================================================================
// True class

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
true_to_s(GV v[], U32 argc)
{
    v[0] = guru_str_new("true");
}
#endif

__GURU__ void
_init_class_true()
{
    // Class
    guru_class *c = guru_class_true = NEW_CLASS("TrueClass", guru_class_object);
    // Methods
#if GURU_USE_STRING
    NEW_PROC("inspect", 	true_to_s);
    NEW_PROC("to_s", 		true_to_s);
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
