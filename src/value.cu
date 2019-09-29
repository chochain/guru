/*! @file
  @brief
  GURU value and macro definitions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include <assert.h>
#include "value.h"
#include "alloc.h"
#include "object.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

//extern "C" __GURU__ void guru_obj_del(GV *v);		// object.cu

//================================================================
/*! compare
 */
__GURU__ S32
_string_cmp(const GV *v0, const GV *v1)
{
	if (v0->str->n != v1->str->n) return -1;

	return STRCMP(v0->str->data, v1->str->data);
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
    case GT_SYM: 	return v0->i - v1->i;

    case GT_CLASS:
    case GT_OBJ:
    case GT_PROC:   return -1 + (v0->self==v1->self) + (v0->self > v1->self)*2;
    case GT_STR: 	return _string_cmp(v0, v1);

#if GURU_USE_FLOAT
    case GT_FLOAT:  return -1 + (v0->f==v1->f) + (v0->f > v1->f)*2;	// caution: NaN == NaN is false
#endif // GURU_USE_FLOAT

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
guru_atoi(U8P s, U32 base)
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
guru_atof(U8P s)
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

__GURU__ U8P guru_i2s(U64 i, U32 base)
{
    U32 bias = 'a' - 10;		// for base > 10
    U8  buf[64+2];				// int64 + terminate + 1
    U8P p = buf + sizeof(buf) - 1;
    U32 x;
    *p = '\0';
    do {
        x = i % 10;
        *--p = (x < 10)? x + '0' : x + bias;
        i /= base;
    } while (i != 0);

    return p;
}

__GURU__ void
guru_memcpy(U8P d, U8P s, U32 sz)
{
    for (U32 i=0; s && d && i<sz; i++, *d++ = *s++);
}

__GURU__ void
guru_memset(U8P d, U8 v,  U32 sz)
{
    for (U32 i=0; d && i<sz; i++, *d++ = v);
}

__GURU__ int
guru_memcmp(U8P d, U8P s, U32 sz)
{
	U32 i;
    for (i=0; s && d && i<sz && *d++==*s++; i++);

    return i<sz;
}

__GURU__ U32
guru_strlen(const U8P str)
{
	U32 i;
	for (i=0; str && str[i]!='\0'; i++);
    return i;
}

__GURU__ void
guru_strcpy(const U8P d, const U8P s)
{
    guru_memcpy(d, s, guru_strlen(s));
}

__GURU__ S32
guru_strcmp(const U8P s1, const U8P s2)
{
    return guru_memcmp(s1, s2, guru_strlen(s1));
}

__GURU__ U8P
guru_strchr(U8P s, const U8 c)
{
    while (s && *s!='\0' && *s!=c) s++;

    return (U8P)((*s==c) ? &s : NULL);
}

__GURU__ U8P
guru_strcat(U8P d, const U8P s)
{
    return d;
}

//================================================================
/*!@brief
  Decrement reference counter

  @param   v     Pointer to target GV
*/
__GURU__ GV *
ref_dec(GV *v)
{
    if ((v->gt & GT_HAS_REF)==0) return v;			// simple objects

    assert(v->self->rc);							// rc > 0
    if (--v->self->rc > 0) return v;				// still used, keep going

    switch(v->gt) {
    case GT_OBJ:		guru_obj_del(v);	break;	// delete object instance
    case GT_PROC:	    guru_free(v->proc);	break;

#if GURU_USE_STRING
    case GT_STR:		guru_str_del(v);	break;
#endif // GURU_USE_STRING

#if GURU_USE_ARRAY
    case GT_ARRAY:	    guru_array_del(v);	break;
    case GT_RANGE:	    guru_range_del(v);	break;
    case GT_HASH:	    guru_hash_del(v);	break;
#endif // GURU_USE_ARRAY

    default: break;
    }
    return v;
}

//================================================================
/*!@brief
  Duplicate GV

  @param   v     Pointer to GV
*/
__GURU__ GV *
ref_inc(GV *v)
{
	if (v->gt & GT_HAS_REF) (v->self->rc++);

	return v;
}

//================================================================
/*!@brief
  Release object related memory

  @param   v     Pointer to target GV
*/
__GURU__ void
ref_clr(GV *v)
{
    if (v->gt & GT_HAS_REF) (v->fil = v->self->rc = 0);
    v->gt = GT_EMPTY;
}

__GURU__ GV
GURU_NIL_NEW()
{
	GV ret; { ret.gt = GT_NIL; } return ret;
}
