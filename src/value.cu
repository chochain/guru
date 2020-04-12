/*! @file
  @brief
  GURU value and macro definitions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include "util.h"
#include "value.h"

#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

//================================================================
// common values
// CC: which should be defined in guru.h or some earlier module
//
__GURU__ GV 	NIL()	{ GV v; { v.gt=GT_NIL;   v.acl=0; } return v; }
__GURU__ GV 	EMPTY()	{ GV v; { v.gt=GT_EMPTY; v.acl=0; } return v; }

//================================================================
/*! compare
 */
__GURU__ S32
_string_cmp(const GV *v0, const GV *v1)
{
	S32 x  = (U32)v0->str->bsz - (U32)v1->str->bsz;
	if (x) return x;

	return STRCMP(v0->str->raw, v1->str->raw);
}

//================================================================
/*! compare two GVs

  @param  v1	Pointer to GV
  @param  v2	Pointer to another GV
  @retval 0	v1 == v2
  @retval plus	v1 >  v2
  @retval minus	v1 <  v2
*/
__GURU__ S32
guru_cmp(const GV *v0, const GV *v1)
{
    if (v0->gt != v1->gt) { 						// GT different
#if GURU_USE_FLOAT
    	GF f0, f1;

        if (v0->gt==GT_INT && v1->gt==GT_FLOAT) {
            f0 = v0->i;
            f1 = v1->f;
            return -1 + (f0 == f1) + (f0 > f1)*2;	// caution: NaN == NaN is false
        }
        if (v0->gt==GT_FLOAT && v1->gt==GT_INT) {
            f0 = v0->f;
            f1 = v1->i;
            return -1 + (f0 == f1) + (f0 > f1)*2;	// caution: NaN == NaN is false
        }
#endif // GURU_USE_FLOAT
        // leak Empty?
        if ((v0->gt==GT_EMPTY && v1->gt==GT_NIL) ||
            (v0->gt==GT_NIL   && v1->gt==GT_EMPTY)) return 0;

        // other case
        return v0->gt - v1->gt;
    }

    // check value
    switch(v1->gt) {
    case GT_NIL:
    case GT_FALSE:
    case GT_TRUE:   return 0;
    case GT_INT:
    case GT_SYM: 	return -1 + (v0->i==v1->i) + (v0->i > v1->i)*2;
#if GURU_USE_FLOAT
    case GT_FLOAT:  return -1 + (v0->f==v1->f) + (v0->f > v1->f)*2;	// caution: NaN == NaN is false
#endif // GURU_USE_FLOAT

    case GT_CLASS:
    case GT_OBJ:
    case GT_PROC:   return -1 + (v0->self==v1->self) + (v0->self > v1->self)*2;
    case GT_STR: 	return _string_cmp(v0, v1);
#if GURU_USE_ARRAY
    case GT_ARRAY:  return guru_array_cmp(v0, v1);
    case GT_RANGE:  return guru_range_cmp(v0, v1);
    case GT_HASH:   return guru_hash_cmp(v0, v1);
#endif // GURU_USE_ARRAY
    default:
        return 1;
    }
}

