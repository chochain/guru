/*! @file
  @brief
  GURU puts functions for Object, Proc, Nil, False and True class and class specific functions.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include "guru.h"
#include "util.h"
#include "static.h"
#include "mmu.h"
#include "class.h"		// class_by_obj

#if GURU_USE_ARRAY
#include "c_array.h"
#include "c_range.h"
#include "c_hash.h"
#endif // GURU_USE_ARRAY

#include "puts.h"

//================================================================
/*! p - sub function
 */
__GURU__ U32
_p(GR *r)
{
	U32 cr = 1;

	switch (r->gt){		// only when output different from print_sub
    case GT_NIL: 	PRINTF("nil");			break;
    case GT_EMPTY:	PRINTF("(empty)");		break;
    case GT_FALSE:	PRINTF("false");		break;
    case GT_TRUE:	PRINTF("true");			break;
    case GT_INT: 	PRINTF("%d", r->i);		break;
    case GT_FLOAT:  PRINTF("%.7g", r->f);	break;				// 23-digit fraction ~= 1/16M => 7 digit
    case GT_CLASS: {
    	U8 *name = _RAW(GR_CLS(r)->cid);						// ~= class->cname in GURU_DEBUG mode
    	PRINTF("%s", name);
    } break;
    case GT_OBJ: {
    	U8 *name = _RAW(_CLS(class_by_obj(r))->cid);			// ~= class->cname
    	PRINTF("#<%s:%p>", name, GR_OBJ(r));
    } break;
    case GT_PROC:
    	PRINTF("#<Proc:%p>", GR_PRC(r));
    	break;
    case GT_SYM: {
        U8 *name = _RAW(r->i);
        STRCHR(name, ';') ? PRINTF("\"%s\"", name) : PRINTF(":%s", name);
    } break;
    case GT_STR:
    	PRINTF("\"%s\"", GR_RAW(r));
    	break;
    case GT_ARRAY: {
    	guru_array *ary = GR_ARY(r);
        GR *p = ary->data;
    	U32 n = ary->n;
        for (int i=0; i < n; i++, p++) {
            PRINTF(i==0 ? "[" : ", ");
            _p(p);			// recursive call
        }
        PRINTF(n==0 ? "[]" : "]");
        cr = 0;
    } break;
    case GT_HASH: {
    	guru_hash *hsh = GR_HSH(r);
        GR *p = hsh->data;
    	U32 n = hsh->n;
        for (int i=0; i < n; i+=2, p+=2) {
        	PRINTF(i==0 ? "{" : ", ");
        	_p(p);
            PRINTF("=>");
            _p(p+1);
        }
        PRINTF(n==0 ? "{}" : "}");
    } break;
    case GT_RANGE: {
    	guru_range *rng = GR_RNG(r);
        _p(&rng->first);
        PRINTF("%s", IS_INCLUDE(rng) ? ".." : "...");
        _p(&rng->last);
    } break;
    default: PRINTF("?vtype: %d", (int)r->gt); break;
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
_print(GR *r)
{
	U32 cr = 1;

    switch (r->gt){		// somehow, Ruby handled the following differently
    case GT_NIL: 		/* print blank */   break;
    case GT_SYM: PRINTF(":%s", _RAW(r->i));	break;
    case GT_STR: {
    	U8  *s  = GR_RAW(r);
    	U32 len = STRLENB(s);
        PRINTF("%s", s);						// no double quote around
        if (len && s[len-1]=='\n') {
        	cr = 0;								// capture line-feed
        }
    } break;
    case GT_ARRAY: {							// =~ruby2.0  !~mruby1.4
    	guru_array *ary = GR_ARY(r);
        GR *p = ary->data;
    	U32 n = ary->n;
        for (int i=0; i < n; i++, p++) {
            if (_print(p)) PRINTF("\n");		// recursive call
        }
        cr = 0;
    } break;
    default: cr = _p(r); break;
    }
    return cr;
}

__GURU__ void
guru_puts(GR r[], U32 ri)
{
    for (int i=0; ri>0 && i < ri; i++) {
    	if (_print(&r[i])) PRINTF("\n");
    }
}

__GURU__ void
guru_p(GR r[], U32 ri)
{
    for (int i=1; ri>0 && i <= ri; i++) {
        _p(&r[i]);
        PRINTF("\n");
    }
}

