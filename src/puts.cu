/*! @file
  @brief
  Guru Object, Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include "guru.h"
#include "value.h"
#include "alloc.h"
#include "symbol.h"
#include "static.h"
#include "global.h"
#include "class.h"

#include "console.h"
#include "puts.h"

#if GURU_USE_ARRAY
#include "c_array.h"
#include "c_range.h"
#include "c_hash.h"
#endif

//================================================================
/*! print - sub function
  @param  v	pointer to target value.
  @retval 0	normal return.
  @retval 1	already output LF.
*/
__GURU__ U32
mrbc_print_sub(mrbc_value *v)
{
    mrbc_value *p;
    U32 ret = 0;

    switch (v->tt){
    case GURU_TT_EMPTY:	 console_str("(empty)");					break;
    case GURU_TT_NIL:					                			break;
    case GURU_TT_FALSE:	 console_str("false");						break;
    case GURU_TT_TRUE:	 console_str("true");						break;
    case GURU_TT_FIXNUM: console_int(v->i);							break;
#if GURU_USE_FLOAT
    case GURU_TT_FLOAT:  console_float(v->f);						break;
#endif
    case GURU_TT_SYMBOL: console_str(VSYM(v));						break;
    case GURU_TT_CLASS:  console_str(symid2name(v->cls->sym_id));   break;
    case GURU_TT_OBJECT:
    	console_str("#<");
    	console_str(symid2name(mrbc_get_class_by_object(v)->sym_id));
        console_str(":");
        console_hex((uintptr_t)v->self>>16);
        console_hex((uintptr_t)v->self&0xffff);
        console_str(">");
        break;
    case GURU_TT_PROC:
    	console_str("#<Proc:");
    	console_hex((uintptr_t)v->proc>>16);
    	console_hex((uintptr_t)v->proc&0xffff);
    	break;
#if GURU_USE_STRING
    case GURU_TT_STRING:
        console_str(VSTR(v));
        if (VSTRLEN(v) != 0 && VSTR(v)[VSTRLEN(v) - 1]=='\n') {
        	ret = 1;
        }
        break;
#endif
#if GURU_USE_ARRAY
    case GURU_TT_ARRAY:
        p = v->array->data;
        for (U32 i=0; i < v->array->n; i++, p++) {
            if (i!=0) console_str("\n");
            mrbc_p_sub(p);
        }
        break;
    case GURU_TT_RANGE:
        mrbc_print_sub(&v->range->first);
        console_str((const U8 *)(IS_EXCLUDE_END(v->range) ? "..." : ".."));
        mrbc_print_sub(&v->range->last);
        break;
    case GURU_TT_HASH:
        console_char('{');
        p = v->hash->data;
        for (U32 i=0; i < v->hash->n; i+=2, p+=2) {
            if (i!=0) console_str(", ");
        	mrbc_p_sub(p);
            console_str("=>");
            mrbc_p_sub(p+1);
        }
        console_char('}');
        break;
#endif
    default:
    	console_str("?vtype: ");
    	console_int((mrbc_int)v->tt);
    	break;
    }
    return ret;
}

//================================================================
/*! p - sub function
 */
__GURU__ U32
mrbc_p_sub(mrbc_value *v)
{
	mrbc_value *p;
	U8P        s;

    switch (v->tt){		// only when output different from print_sub
    case GURU_TT_NIL: console_str("nil");		break;
    case GURU_TT_SYMBOL:
        s = VSYM(v);
        if (STRCHR(s, ':')) {
        	console_str("\":");
        	console_str(s);
        	console_str("\"");
        }
        else {
        	console_str(":");
        	console_str(s);
        }
        break;
#if GURU_USE_ARRAY
    case GURU_TT_ARRAY:
        console_char('[');
        p = v->array->data;
        for (U32 i=0; i < v->array->n; i++, p++) {
            if (i!=0) console_str(", ");
            mrbc_p_sub(p);
        }
        console_char(']');
        break;
#endif
#if GURU_USE_STRING
    case GURU_TT_STRING:
        s = VSTR(v);
        console_char('"');
        for (U32 i=0; i < VSTRLEN(v); i++, s++) {
            if (*s>=' ' && *s < 0x80) console_char(*s);
            else 					  console_hex(*s);
        }
        console_char('"');
        break;
#endif
    default:
        mrbc_print_sub(v);
        break;
    }
    return 0;
}

