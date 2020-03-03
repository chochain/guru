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
#include "refcnt.h"

#ifdef __cplusplus
extern "C" {
#endif
__GURU__ S32  	guru_cmp(const GV *v0, const GV *v1);
__GURU__ GV 	NIL();
__GURU__ GV     EMPTY();

// macro for C call returns
// Note: becareful, the following macros assume a "v" pointer to top of stack
//
#define RETURN_VAL(n)	{ ref_dec(v); *v=(n); 		 			return; }
#define RETURN_NIL()	{ ref_dec(v); v->acl=0; v->gt=GT_NIL;   return; }
#define RETURN_FALSE()	{ ref_dec(v); v->acl=0; v->gt=GT_FALSE; return; }
#define RETURN_TRUE()	{ ref_dec(v); v->acl=0; v->gt=GT_TRUE;  return; }
     
#define RETURN_BOOL(n)	{ ref_dec(v); v->acl=0; v->gt=(n)?GT_TRUE:GT_FALSE;  return; }
#define RETURN_INT(n)	{ ref_dec(v); v->acl=0; v->gt=GT_INT;  v->i=(GI)(n); return; }
#define RETURN_FLOAT(n)	{ ref_dec(v); v->acl=0; v->gt=GT_FLOAT;v->f=(GF)(n); return; }

// macros to create new built-in objects
#if defined(__CUDACC__)
__GURU__ GI   			guru_atoi(const U8 *s, U32 base);
__GURU__ GF				guru_atof(const U8 *s);

#define ATOI(s)         guru_atoi(s, 10)
#define ATOF(s)			guru_atof(s)

#else
#define ATOI(s)			atol(s)
#define ATOF(s)			atof(s)

#define MEMCPY(d,s,sz)  memcpy(d, s, sz)
#define MEMSET(d,v,sz)  memset(d, v, sz)
#define MEMCMP(d,s,sz)  memcmp(d, s, sz)

#define STRLEN(s)		strlen(s)
#define STRCPY(d,s)	  	strcpy(d, s)
#define STRCMP(s1,s2)   strcmp(s1, s2)
#define STRCHR(d,c)     strchr(d, c)
#define STRCAT(d,s)     strcat(d, s)
#endif

// basic C string functions for GV
#define GVSTR(v) ((U8*)(v)->str->data)

#ifdef __cplusplus
}
#endif
#endif
