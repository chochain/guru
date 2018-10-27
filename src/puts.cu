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

#if MRBC_USE_ARRAY
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
    case MRBC_TT_CLASS:  console_str(symid2name(v->cls->sym_id));   break;
    case MRBC_TT_OBJECT:
    	console_str("#<");
    	console_str(symid2name(mrbc_get_class_by_object(v)->sym_id));
        console_str(":0x");
        console_hex((mrbc_int)v->self);
        console_str(">");
        break;
    case MRBC_TT_PROC:
    	console_str("#<Proc:0x");
    	console_hex((mrbc_int)v->proc);
    	break;
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
        console_str(IS_EXCLUDE_END(v->range) ? "..." : "..");
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
        console_printf(s, fmt);
    } break;

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
#if MRBC_USE_ARRAY
    case MRBC_TT_RANGE:{
        mrbc_value v1 = mrbc_range_first(v);
        mrbc_p_sub(&v1);
        console_str(IS_EXCLUDE_END(v->range) ? "..." : "..");
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

