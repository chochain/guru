/*! @file
  @brief
  Guru Object, Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <stdio.h>

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

__GURU__ U32 _p(mrbc_value *v);				// forward declaration

//================================================================
/*! print - sub function
  @param  v	pointer to target value.
  @retval 0	normal return.
  @retval 1	already output LF.
*/
__GURU__ U32
_print(mrbc_value *v)
{
    mrbc_value *p;
    U32 ret = 0;

    switch (v->tt){
    case GURU_TT_EMPTY:	 printf("(empty)");		break;
    case GURU_TT_NIL:					    	break;
    case GURU_TT_FALSE:	 printf("false");		break;
    case GURU_TT_TRUE:	 printf("true");		break;
    case GURU_TT_FIXNUM: printf("%d", v->i);	break;
#if GURU_USE_FLOAT
    case GURU_TT_FLOAT:  printf("%f", v->f);	break;
#endif
    case GURU_TT_SYMBOL: printf("%s", VSYM(v));	break;
    case GURU_TT_CLASS:  printf("%s", symid2name(v->cls->sym_id));   break;
    case GURU_TT_OBJECT:
    	printf("#<%04d:0x%08x>",
    		symid2name(mrbc_get_class_by_object(v)->sym_id),
    		v->self
    	);
        break;
    case GURU_TT_PROC: 	printf("#<Proc:0x%08x", v->proc); break;
#if GURU_USE_STRING
    case GURU_TT_STRING:
        printf("%s", VSTR(v));
        if (VSTRLEN(v) != 0 && VSTR(v)[VSTRLEN(v) - 1]=='\n') {
        	ret = 1;
        }
        break;
#endif
#if GURU_USE_ARRAY
    case GURU_TT_ARRAY:
        p = v->array->data;
        for (U32 i=0; i < v->array->n; i++, p++) {
            if (i!=0) printf("\n");
            _p(p);
        }
        break;
    case GURU_TT_RANGE:
        _print(&v->range->first);
        printf((IS_EXCLUDE_END(v->range) ? "..." : ".."));
        _print(&v->range->last);
        break;
    case GURU_TT_HASH:
        printf("%c", '{');
        p = v->hash->data;
        for (U32 i=0; i < v->hash->n; i+=2, p+=2) {
            if (i!=0) printf(", ");
        	_p(p);
            printf("=>");
            _p(p+1);
        }
        printf("%c", '}');
        break;
#endif
    default: printf("?vtype: %d", (int)v->tt); break;
    }
    return ret;
}

//================================================================
/*! p - sub function
 */
__GURU__ U32
_p(mrbc_value *v)
{
	mrbc_value *p;
	U8P        s;

    switch (v->tt){		// only when output different from print_sub
    case GURU_TT_NIL: printf("nil");		break;
    case GURU_TT_SYMBOL:
        s = VSYM(v);
        printf(STRCHR(s, ';') ? "\":%s\"" : ":%s", s);
        break;
#if GURU_USE_ARRAY
    case GURU_TT_ARRAY:
        printf("%c", '[');
        p = v->array->data;
        for (U32 i=0; i < v->array->n; i++, p++) {
            if (i!=0) printf(", ");
            _p(p);
        }
        printf("%c", ']');
        break;
#endif
#if GURU_USE_STRING
    case GURU_TT_STRING:
    	printf("\"%s\"", VSTR(v));
    	break;
#endif
    default:
        _print(v);
        break;
    }
    return 0;
}

__GURU__ void
guru_puts(mrbc_value *v, U32 argc)
{
    if (argc) {
    	for (U32 i = 1; i <= argc; i++) {
    		if (_print(&v[i]) == 0) printf("%c", '\n');
    	}
    }
    else printf("%c", '\n');
}

__GURU__ void
guru_p(mrbc_value *v, U32 argc)
{
    for (U32 i = 1; i <= argc; i++) {
        _p(&v[i]);
        printf("%c", '\n');
    }
}

__GURU__ void
guru_na(const U8 *msg)
{
	printf("method not supported: %s\n", msg);
}
