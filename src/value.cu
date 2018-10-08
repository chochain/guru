/*! @file
  @brief
  Guru value definitions

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/

#include <assert.h>
#include "value.h"

#if MRBC_USE_STRING
#include "c_string.h"
#endif
#if MRGC_USE_ARRAY
#include "c_range.h"
#include "c_array.h"
#include "c_hash.h"
#endif

//================================================================
/*! compare two mrbc_values

  @param  v1	Pointer to mrbc_value
  @param  v2	Pointer to another mrbc_value
  @retval 0	v1 == v2
  @retval plus	v1 >  v2
  @retval minus	v1 <  v2
*/
__GURU__
int mrbc_compare(const mrbc_value *v1, const mrbc_value *v2)
{
    // if TT_XXX is different
    if (v1->tt != v2->tt) {
#if MRBC_USE_FLOAT
        // but Numeric?
        if (v1->tt == MRBC_TT_FIXNUM && v2->tt == MRBC_TT_FLOAT) {
            d1 = v1->i;
            d2 = v2->d;
            goto CMP_FLOAT;
        }
        if (v1->tt == MRBC_TT_FLOAT && v2->tt == MRBC_TT_FIXNUM) {
            d1 = v1->d;
            d2 = v2->i;
            goto CMP_FLOAT;
        }
#endif
        // leak Empty?
        if ((v1->tt == MRBC_TT_EMPTY && v2->tt == MRBC_TT_NIL) ||
            (v1->tt == MRBC_TT_NIL   && v2->tt == MRBC_TT_EMPTY)) return 0;

        // other case
        return v1->tt - v2->tt;
    }

    // check value
    switch(v1->tt) {
    case MRBC_TT_NIL:
    case MRBC_TT_FALSE:
    case MRBC_TT_TRUE:
        return 0;

    case MRBC_TT_FIXNUM:
    case MRBC_TT_SYMBOL:
        return v1->i - v2->i;

    case MRBC_TT_CLASS:
    case MRBC_TT_OBJECT:
    case MRBC_TT_PROC:
        return -1 + (v1->handle == v2->handle) + (v1->handle > v2->handle)*2;
    
#if MRBC_USE_STRING
    case MRBC_TT_STRING:
        return mrbc_string_compare(v1, v2);
#endif
#if MRBC_USE_FLOAT
    case MRBC_TT_FLOAT:
        d1 = v1->d;
        d2 = v2->d;
        goto CMP_FLOAT;
#endif
#if MRBC_USE_ARRAY
    case MRBC_TT_ARRAY:
        return mrbc_array_compare(v1, v2);

    case MRBC_TT_RANGE:
        return mrbc_range_compare(v1, v2);

    case MRBC_TT_HASH:
        return mrbc_hash_compare(v1, v2);
#endif
    default:
        return 1;
    }

#if MRBC_USE_FLOAT
    mrbc_float d1, d2;

CMP_FLOAT:
    return -1 + (d1 == d2) + (d1 > d2)*2;	// caution: NaN == NaN is false
#endif
}

//================================================================
/*!@brief
  Duplicate mrbc_value

  @param   v     Pointer to mrbc_value
*/
__GURU__
void mrbc_dup(mrbc_value *v)
{
    switch(v->tt){
    case MRBC_TT_OBJECT:
    case MRBC_TT_PROC:
    case MRBC_TT_ARRAY:
    case MRBC_TT_STRING:
    case MRBC_TT_RANGE:
    case MRBC_TT_HASH:
        assert(v->instance->ref_count > 0);
        assert(v->instance->ref_count != 0xff);	// check max value.
        v->instance->ref_count++;
        break;

    default:
        // Nothing
        break;
    }
}

//================================================================
/*!@brief

  convert ASCII string to integer Guru version

  @param  s	source string.
  @param  base	n base.
  @return	result.
*/
__GURU__
mrbc_int mrbc_atoi(const char *s, int base)
{
    int ret = 0;
    int sign = 0;

REDO:
    switch(*s) {
    case '-':
        sign = 1;
        // fall through.
    case '+':
        s++;
        break;

    case ' ':
        s++;
        goto REDO;
    }

    int ch;
    while((ch = *s++) != '\0') {
        int n;

        if ('a' <= ch) {
            n = ch - 'a' + 10;
        }
        else {
            if ('A' <= ch) {
                n = ch - 'A' + 10;
            }
            else {
                if ('0' <= ch && ch <= '9') {
                    n = ch - '0';
                }
                else break;
            }
        }
        if (n >= base) break;

        ret = ret * base + n;
    }
    if (sign) ret = -ret;

    return ret;
}

__GURU__ void guru_memcpy(uint8_t *d, const uint8_t *s, size_t sz)
{
    for (int i=0; i<sz; i++, *d++ = *s++);
}

__GURU__ void guru_memset(uint8_t *d, const uint8_t v,  size_t sz)
{
    for (int i=0; i<sz; i++, *d++ = v);
}

__GURU__ int guru_memcmp(const uint8_t *d, const uint8_t *s, size_t sz)
{
    int i;

    for (i=0; i<sz && *d++==*s++; i++);

    return i<sz;
}

__GURU__ long guru_atol(const char *s)
{
    return 0L;
}

__GURU__ size_t guru_strlen(const char *str)
{
    int i=0;
    while (str[++i]!='\0');
    return (size_t)i;
}

__GURU__ void  guru_strcpy(const char *s1, const char *s2)
{
    guru_memcpy((uint8_t *)s1, (uint8_t *)s2, guru_strlen(s1));
}

__GURU__ size_t  guru_strcmy(const char *s1, const char *s2)
{
    return guru_memcmp((uint8_t *)s1, (uint8_t *)s2, guru_strlen(s1));
}

__GURU__ char   *guru_strchr(const char *s, const char c)
{
    while (*s!='\0' && *s!=c) s++;

    return (char *)((*s==c) ? &s : NULL);
}

__GURU__ char   *guru_strcat(char *d, const char *s)
{
    return d;
}


