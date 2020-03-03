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
#include "object.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

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

//================================================================
/*!@brief

  convert ASCII string to integer Guru version

  @param  s	source string.
  @param  base	n base.
  @return	result.
*/
__GURU__ GI
guru_atoi(const U8 *s, U32 base)
{
    GI  ret  = 0;
    U32 sign = 0;

REDO:
    switch(*s) {
    case '-': sign = 1;		// fall through.
    case '+': s++;	        break;
    case ' ': s++;          goto REDO;
    }

    U8  ch;
    U32 n;
    while ((ch = *s++) != '\0') {
        if      ('a' <= ch) 			 n = ch - 'a' + 10;
        else if ('A' <= ch) 			 n = ch - 'A' + 10;
        else if ('0' <= ch && ch <= '9') n = ch - '0';
        else break;

        if (n >= base) break;

        ret = ret * base + n;
    }
    return (sign) ? -ret : ret;
}

__GURU__ GF
guru_atof(const U8 *s)
{
#if GURU_USE_FLOAT
    int sign = 1, esign = 1, state=0;
    int r = 0, e = 0;
    long v = 0L, f = 0L;

    while ((*s<'0' || *s>'9') && *s!='+' && *s!='-') s++;

    if (*s=='+' || *s=='-') sign = *s++=='-' ? -1 : 1;

    while (*s!='\0' && *s!='\n' && *s!=' ' && *s!='\t') {
    	if      (state==0 && *s>='0' && *s<='9') {	// integer
    		v = (*s - '0') + v * 10;
    	}
    	else if (state==1 && *s>='0' && *s<='9') {	// decimal
    			f = (*s - '0') + f * 10;
    			r--;
        }
    	else if (state==2) {						// exponential
            if (*s=='-') {
                esign = -1;
                s++;
            }
            if (*s>='0' && *s<='9') e = (*s - '0') + e * 10;
        }
        state = (*s=='e' || *s=='E') ? 2 : ((*s=='.') ? 1 : state);
        s++;
    }
    GF ret = sign
    		* (v + (f==0 ? 0.0 : f * exp10((double)r)))
    		* (e==0 ? 1.0 : exp10((double)esign * e));

    return ret;
#else
    return 0.0;
#endif // GURU_USE_FLOAT
}

__GURU__ GV NIL() 	{ GV v; { v.gt=GT_NIL;   v.acl=0; } return v; }
__GURU__ GV EMPTY()	{ GV v; { v.gt=GT_EMPTY; v.acl=0; } return v; }


