#include <assert.h>
#include "value.h"
#include "alloc.h"

#if GURU_USE_STRING
#include "c_string.h"
#endif

#if GURU_USE_ARRAY
#include "c_range.h"
#include "c_array.h"
#include "c_hash.h"
#endif

extern "C" __GURU__ void guru_store_delete(GV *v);		// store.cu

//================================================================
/*! compare
 */
__GURU__ S32
_string_compare(const GV *v1, const GV *v2)
{
	if (v1->str->len != v2->str->len) return -1;

	return STRCMP(v1->str->data, v2->str->data);
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
guru_cmp(const GV *v1, const GV *v2)
{
    if (v1->gt != v2->gt) { 						// guru_vtype different
#if GURU_USE_FLOAT
    	guru_float f1, f2;

        if (v1->gt == GT_INT && v2->gt == GT_FLOAT) {
            f1 = v1->i;
            f2 = v2->f;
            return -1 + (f1 == f2) + (f1 > f2)*2;	// caution: NaN == NaN is false
        }
        if (v1->gt == GT_FLOAT && v2->gt == GT_INT) {
            f1 = v1->f;
            f2 = v2->i;
            return -1 + (f1 == f2) + (f1 > f2)*2;	// caution: NaN == NaN is false
        }
#endif
        // leak Empty?
        if ((v1->gt == GT_EMPTY && v2->gt == GT_NIL) ||
            (v1->gt == GT_NIL   && v2->gt == GT_EMPTY)) return 0;

        // other case
        return v1->gt - v2->gt;
    }

    // check value
    switch(v1->gt) {
    case GT_NIL:
    case GT_FALSE:
    case GT_TRUE:   return 0;
    case GT_INT:
    case GT_SYM: 	return v1->i - v2->i;

    case GT_CLASS:
    case GT_OBJ:
    case GT_PROC:   return -1 + (v1->self == v2->self) + (v1->self > v2->self)*2;
    case GT_STR: 	return _string_compare(v1, v2);

#if GURU_USE_FLOAT
    case GT_FLOAT:  return -1 + (v1->f==v2->f) + (v1->f > v2->f)*2;	// caution: NaN == NaN is false
#endif
#if GURU_USE_ARRAY
    case GT_ARRAY:  return guru_array_compare(v1, v2);
    case GT_RANGE:  return guru_range_compare(v1, v2);
    case GT_HASH:   return guru_hash_compare(v1, v2);
#endif
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
__GURU__ guru_int
guru_atoi(U8P s, U32 base)
{
    U32 ret  = 0;
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

__GURU__ guru_float
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
    double ret = sign
    		* (v + (f==0 ? 0.0 : f * exp10((double)r)))
    		* (e==0 ? 1.0 : exp10((double)esign * e));

    return (guru_float)ret;
#else
    return 0.0;
#endif
}

__GURU__ U8P guru_i2s(U64 i, U32 base)
{
    U32 bias = 'a' - 10;		// for base > 10

    U8  buf[20+2];				// int64 + terminate + 1
    U8P p = buf + sizeof(buf) - 1;
    U32 x;
    *p = '\0';
    do {
        x = i % 10;
        *--p = (x < 10)? x + '0' : x + bias;
        x /= base;
    } while (x != 0);

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
guru_strcpy(const U8P s1, const U8P s2)
{
    guru_memcpy(s1, s2, guru_strlen(s1));
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
__GURU__ void
ref_dec(GV *v)
{
	if (!(v->gt & GT_HAS_REF)) return;

	assert(v->self->refc > 0);

    if (--v->self->refc > 0) return;		// still used, keep going

    switch(v->gt) {
    case GT_OBJ:		guru_store_delete(v);	break;
    case GT_PROC:	    mrbc_free(v->proc);		break;
#if GURU_USE_STRING
    case GT_STR:		guru_str_delete(v);		break;
#endif
#if GURU_USE_ARRAY
    case GT_ARRAY:	    guru_array_delete(v);	break;
    case GT_RANGE:	    guru_range_delete(v);	break;
    case GT_HASH:	    guru_hash_delete(v);	break;
#endif
    default: break;
    }
}

//================================================================
/*!@brief
  Duplicate GV

  @param   v     Pointer to GV
*/
__GURU__ void
ref_inc(GV *v)         			// CC: was mrbc_inc_refc() 20181101
{
	if (!(v->gt & GT_HAS_REF)) return;

	v->self->refc++;
}

//================================================================
/*!@brief
  Release object related memory

  @param   v     Pointer to target GV
*/
__GURU__ void
ref_clr(GV *v)
{
    ref_dec(v);
    v->gt = GT_EMPTY;
}
