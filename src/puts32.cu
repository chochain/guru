/*! @file
  @brief
  GURU puts functions for Object, Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include "guru.h"
#include "class.h"
#include "value.h"
#include "mmu.h"
#include "symbol.h"
//#include "static.h"
#include "global.h"

#include "puts.h"

#if GURU_USE_ARRAY
#include "c_array.h"
#include "c_range.h"
#include "c_hash.h"
#endif // GURU_USE_ARRAY

//================================================================
/*! p - sub function
 */
__GURU__ U32
_p(GV *v)
{
	U32 cr = 1;

	switch (v->gt){		// only when output different from print_sub
    case GT_NIL: 	PRINTF("nil");			break;
    case GT_EMPTY:	PRINTF("(empty)");		break;
    case GT_FALSE:	PRINTF("false");		break;
    case GT_TRUE:	PRINTF("true");			break;
    case GT_INT: 	PRINTF("%d", v->i);		break;
    case GT_FLOAT:  PRINTF("%.7g", v->f);	break;		// 23-digit fraction ~= 1/16M => 7 digit
    case GT_CLASS: {
    	U8 *name = id2name(v->self->sid);
    	PRINTF("%s", name);
    } break;
    case GT_OBJ: {
    	U8 *name = id2name(class_by_obj(v)->sid);
    	PRINTF("#<%s:%08x>", name, (U32A)v->self);
    } break;
    case GT_PROC:
    	PRINTF("#<Proc:%08x>", v->proc);
    	break;
    case GT_SYM: {
        U8 *name = id2name(v->i);
        STRCHR(name, ';') ? PRINTF("\"%s\"", name) : PRINTF(":%s", name);
    } break;
    case GT_STR:
    	PRINTF("\"%s\"", v->str->raw);
    	break;
    case GT_ARRAY: {
        GV *p = v->array->data;
    	U32 n = v->array->n;
        for (U32 i=0; i < n; i++, p++) {
            PRINTF(i==0 ? "[" : ", ");
            _p(p);			// recursive call
        }
        PRINTF(n==0 ? "[]" : "]");
        cr = 0;
    } break;
    case GT_HASH: {
        GV *p = v->hash->data;
    	U32 n = v->hash->n;
        for (U32 i=0; i < n; i+=2, p+=2) {
        	PRINTF(i==0 ? "{" : ", ");
        	_p(p);
            PRINTF("=>");
            _p(p+1);
        }
        PRINTF(n==0 ? "{}" : "}");
    } break;
    case GT_RANGE:
        _p(&v->range->first);
        PRINTF("%s", IS_INCLUDE(v->range) ? ".." : "...");
        _p(&v->range->last);
        break;
    default: PRINTF("?vtype: %d", (int)v->gt); break;
    }
	return cr;
}

//================================================================
/*! print - sub function
  @param  v	pointer to target value.
  @retval 0	normal return.
  @retval 1	already output LF.
*/
__GURU__ U32
_print(GV *v)
{
	U32 cr = 1;

    switch (v->gt){		// somehow, Ruby handled the following differently
    case GT_NIL: 		/* print blank */    	break;
    case GT_SYM: PRINTF(":%s", id2name(v->i));	break;
    case GT_STR: {
    	U8  *s  = (U8*)v->str->raw;
    	U32 len = STRLENB(s);
        PRINTF("%s", s);						// no double quote around
        if (len && s[len-1]=='\n') {
        	cr = 0;								// capture line-feed
        }
    } break;
    case GT_ARRAY: {							// =~ruby2.0  !~mruby1.4
        GV *p = v->array->data;
    	U32 n = v->array->n;
        for (U32 i=0; i < n; i++, p++) {
            if (_print(p)) PRINTF("\n");		// recursive call
        }
        cr = 0;
    } break;
    default: cr = _p(v); break;
    }
    return cr;
}

__GURU__ void
guru_puts(GV v[], U32 vi)
{
    for (U32 i=0; vi>0 && i < vi; i++) {
    	if (_print(&v[i])) PRINTF("\n");
    }
}

__GURU__ void
guru_p(GV v[], U32 vi)
{
    for (U32 i=1; vi>0 && i <= vi; i++) {
        _p(&v[i]);
        PRINTF("\n");
    }
}

