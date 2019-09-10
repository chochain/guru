/*! @file
  @brief
  GURU value and macro definitions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_VALUE_H_
#define GURU_SRC_VALUE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif
__GURU__ S32  guru_cmp(const GV *v1, const GV *v2);

__GURU__ void ref_clr(GV *v);
__GURU__ GV   *ref_dec(GV *v);
__GURU__ GV   *ref_inc(GV *v);

// macro for C call returns
// Note: becareful, the following macros assume a "v" pointer to top of stack
//
#define RETURN_VAL(n)	{ *v=(n); 		  return; }
#define RETURN_NIL()	{ v->gt=GT_NIL;   return; }
#define RETURN_FALSE()	{ v->gt=GT_FALSE; return; }
#define RETURN_TRUE()	{ v->gt=GT_TRUE;  return; }
     
#define RETURN_BOOL(n)	{ v->gt=(n)?GT_TRUE:GT_FALSE;   return; }
#define RETURN_INT(n)	{ v->gt=GT_INT;   v->i=(GI)(n); return; }
#define RETURN_FLOAT(n)	{ v->gt=GT_FLOAT; v->f=(GF)(n); return; }

// macro to create new built-in objects
#define GURU_NIL_NEW()	((guru_obj) {.gt = GT_NIL})

#ifdef __GURU_CUDA__
__GURU__ GI   		guru_atoi(U8P s, U32 base);
__GURU__ GF			guru_atof(U8P s);

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

#define MEMCPY(d, s, sz)  guru_memcpy((U8P)(d), (U8P)(s), (U32)(sz))
#define MEMSET(d, v, sz)  guru_memset((U8P)(d), (U8)(v),  (U32)(sz))
#define MEMCMP(d, s, sz)  guru_memcmp((U8P)(d), (U8P)(s), (U32)(sz))

#define STRLEN(s)		  guru_strlen((U8P)(s))
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

// basic C string functions for GV
#define GVSTR(v) ((U8P)(v)->str->data)

#ifdef __cplusplus
}
#endif
#endif
