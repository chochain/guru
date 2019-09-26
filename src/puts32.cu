/*! @file
  @brief
  GURU puts functions for Object, Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2019- GreenII

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

#include "puts.h"

#if GURU_USE_ARRAY
#include "c_array.h"
#include "c_range.h"
#include "c_hash.h"
#endif // GURU_USE_ARRAY

__GURU__ U32 _p(GV *v);				// forward declaration

//================================================================
/*! print - sub function
  @param  v	pointer to target value.
  @retval 0	normal return.
  @retval 1	already output LF.
*/
__GURU__ U32
_print(GV *v)
{
    GV *p;
    U32 ret = 0;

    switch (v->gt){
    case GT_EMPTY:	PRINTF("(empty)");		break;
    case GT_NIL:					    	break;
    case GT_FALSE:	PRINTF("false");		break;
    case GT_TRUE:	PRINTF("true");			break;
    case GT_INT: 	PRINTF("%d", v->i);		break;
#if GURU_USE_FLOAT
    case GT_FLOAT:  PRINTF("%f", v->f);		break;
#endif // GURU_USE_FLOAT
    case GT_SYM: 	PRINTF(":%s", id2name(v->i));			break;
    case GT_CLASS:  PRINTF("%s",  id2name(v->cls->sid));  	break;
    case GT_OBJ:
    	PRINTF("#<%s:0x%08x>",
    		id2name(class_by_obj(v)->sid),
    		(U32A)v->self
    	);
        break;
    case GT_PROC: 	PRINTF("#<Proc:0x%08x", v->proc); break;
#if GURU_USE_STRING
    case GT_STR: {
    	U8P s   = GVSTR(v);
    	U32 len = STRLEN(s);
        PRINTF("%s", s);
        if (len && s[len-1]=='\n') {
        	ret = 1;
        }
    } break;
#endif  // GURU_USE_STRING
#if GURU_USE_ARRAY
    case GT_ARRAY:
        p = v->array->data;
        for (U32 i=0; i < v->array->n; i++, p++) {
            if (i!=0) PRINTF("\n");
            _p(p);
        }
        break;
    case GT_RANGE:
        _print(&v->range->first);
        PRINTF("%s", IS_EXCLUDE_END(v->range) ? "..." : "..");
        _print(&v->range->last);
        break;
    case GT_HASH:
        PRINTF("%c", '{');
        p = v->hash->data;
        for (U32 i=0; i < v->hash->n; i+=2, p+=2) {
            if (i!=0) PRINTF(", ");
        	_p(p);
            PRINTF("=>");
            _p(p+1);
        }
        PRINTF("%c", '}');
        break;
#endif	// GURU_USE_ARRAY
    default: PRINTF("?vtype: %d", (int)v->gt); break;
    }
    return ret;
}

//================================================================
/*! p - sub function
 */
__GURU__ U32
_p(GV *v)
{
	GV  *p;
	U8P name;

    switch (v->gt){		// only when output different from print_sub
    case GT_NIL: PRINTF("nil");		break;
    case GT_SYM:
        name = id2name(v->i);
        STRCHR(name, ';') ? PRINTF("\"%s\"", name) : PRINTF(":%s", name);
        break;
#if GURU_USE_ARRAY
    case GT_ARRAY:
        PRINTF("%c", '[');
        p = v->array->data;
        for (U32 i=0; i < v->array->n; i++, p++) {
            if (i!=0) PRINTF(", ");
            _p(p);
        }
        PRINTF("%c", ']');
        break;
#endif // GURU_USE_ARRAY
#if GURU_USE_STRING
    case GT_STR:
    	PRINTF("\"%s\"", GVSTR(v));
    	break;
#endif // GURU_USE_STRING
    default:
        _print(v);
        break;
    }
    return 0;
}

__GURU__ void
guru_puts(GV v[], U32 vi)
{
    if (vi) {
    	for (U32 i=1; i <= vi; i++) {
    		if (_print(&v[i])==0) PRINTF("\n");
    	}
    }
    else PRINTF("\n");
}

__GURU__ void
guru_p(GV v[], U32 vi)
{
    for (U32 i=1; i <= vi; i++) {
        _p(&v[i]);
        PRINTF("\n");
    }
}

