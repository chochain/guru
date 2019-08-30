/*! @file
  @brief
  Guru value definitions

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_VALUE_H_
#define GURU_SRC_VALUE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif
__GURU__ S32  mrbc_compare(const mrbc_value *v1, const mrbc_value *v2);
__GURU__ void mrbc_retain(mrbc_value *v);
__GURU__ void mrbc_release(mrbc_value *v);
__GURU__ void mrbc_dec_refc(mrbc_value *v);

// for C call
#define SET_RETURN(n)		do { mrbc_value nnn = (n); mrbc_dec_refc(v); v[0] = nnn; 	} while (0)
#define SET_NIL_RETURN()	do { mrbc_dec_refc(v); v[0].tt = GURU_TT_NIL;   			} while (0)

#define SET_FALSE_RETURN()	do { mrbc_dec_refc(v); v[0].tt = GURU_TT_FALSE; 			} while (0)
#define SET_TRUE_RETURN()	do { mrbc_dec_refc(v); v[0].tt = GURU_TT_TRUE;  			} while (0)
#define SET_BOOL_RETURN(n)	do { mrbc_dec_refc(v); v[0].tt = (n)?GURU_TT_TRUE:GURU_TT_FALSE; } while (0)

#define SET_INT_RETURN(n)	do { mrbc_int nnn = (n);					\
		mrbc_dec_refc(v); v[0].tt = GURU_TT_FIXNUM; v[0].i = nnn; } while (0)
#define SET_FLOAT_RETURN(n)	do { mrbc_float nnn = (n);                  \
        mrbc_dec_refc(v); v[0].tt = GURU_TT_FLOAT;  v[0].f = nnn; } while (0)

#define GET_TT_ARG(n)			(v[(n)].tt)
#define GET_INT_ARG(n)			(v[(n)].i)
#define GET_ARY_ARG(n)			(v[(n)])
#define GET_FLOAT_ARG(n)		(v[(n)].f)
#define GET_STRING_ARG(n)		(v[(n)].string->data)

#define mrbc_fixnum_value(n)	((mrbc_value){.tt = GURU_TT_FIXNUM, .i=(n)})
#define mrbc_float_value(n)	    ((mrbc_value){.tt = GURU_TT_FLOAT,  .f=(n)})
#define mrbc_nil_value()	    ((mrbc_value){.tt = GURU_TT_NIL})
#define mrbc_true_value()	    ((mrbc_value){.tt = GURU_TT_TRUE})
#define mrbc_false_value()	    ((mrbc_value){.tt = GURU_TT_FALSE})
#define mrbc_bool_value(n)	    ((mrbc_value){.tt = (n)?GURU_TT_TRUE:GURU_TT_FALSE})

#ifdef __GURU_CUDA__
__GURU__ mrbc_int   guru_atoi(U8P s, U32 base);
__GURU__ mrbc_float	guru_atof(U8P s);

__GURU__ U8P 		guru_i2s(U64 i, U32 base);

__GURU__ void 		guru_memcpy(U8P d, U8P s, U32 sz);
__GURU__ void    	guru_memset(U8P d, U8  v, U32 sz);
__GURU__ S32     	guru_memcmp(U8P d, U8P s, U32 sz);

__GURU__ U32  		guru_strlen(const U8P s);
__GURU__ void    	guru_strcpy(const U8P s1, const U8P s2);
__GURU__ S32     	guru_strcmp(const U8P s1, const U8P s2);
__GURU__ U8P		guru_strchr(U8P d, const U8 c);
__GURU__ U8P		guru_strcat(U8P d, const U8P s);

#define ATOI(s)           guru_atoi(s, 10)
#define ATOF(s)			  guru_atof(s)

#define MEMCPY(d, s, sz)  guru_memcpy((U8P)d, (U8P)s, (U32)sz)
#define MEMSET(d, v, sz)  guru_memset((U8P)d, (U8)v,  (U32)sz)
#define MEMCMP(d, s, sz)  guru_memcmp((U8P)d, (U8P)s, (U32)sz)

#define STRLEN(s)		  guru_strlen((U8P)s)
#define STRCPY(d, s)	  guru_strcpy((U8P)d, (U8P)s)
#define STRCMP(d, s)      guru_strcmp((U8P)d, (U8P)s)
#define STRCHR(d, c)      guru_strchr((U8P)d, (U8)c)
#define STRCAT(d, s)      guru_strcat((U8P)d, (U8P)s)
#else
#define ATOI(s)			  atol(s)
#define ATOF(s)			  atof(s)

#define MEMCPY(d, s, sz)  memcpy(d, s, sz)
#define MEMSET(d, v, sz)  memset(d, v, sz)
#define MEMCMP(d, s, sz)  memcmp(d, s, sz)

#define STRLEN(s)		  strlen(s)
#define STRCPY(s1, s2)	  strcpy(s1, s2)
#define STRCMP(s1, s2)    strcmp(s1, s2)
#define STRCHR(s, c)      strchr(s, c)
#define STRCAT(d, s)      strcat(d, s)
#endif

// basic C string functions for mrbc_value
#define VSTRLEN(v)		((v)->str->len)
#define VSTR(v)			((U8P)(v)->str->data)
#define VSTRCMP(v1, v2) (STRCMP((v1)->str->data, (v2)->str->data))
#define VSYM(v)			((U8P)symid2name((v)->i))

#ifdef __cplusplus
}
#endif
#endif
