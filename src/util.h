/*! @file
  @brief
  GURU Utility functions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_UTIL_H_
#define GURU_SRC_UTIL_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void 			d_memcpy(U8 *d, const U8 *s, U32 sz);
__GURU__ void    		d_memset(U8 *d, U8  v, U32 sz);
__GURU__ S32     		d_memcmp(const U8 *d, const U8 *s, U32 sz);

__GURU__ U32  			d_strlen(const U8 *s, U32 use_byte);
__GURU__ S32     		d_strcmp(const U8 *s1, const U8 *s2);
__GURU__ void    		d_strcpy(U8 *s1, const U8 *s2);
__GURU__ U8*			d_strchr(U8 *d,  const U8 c);
__GURU__ U8*			d_strcat(U8 *d,  const U8 *s);
__GURU__ U8*     		d_strcut(const U8 *s, U32 n);			// take n utf8 chars from the string

__GURU__ U32 			d_calc_hash(const U8 *str);

#if defined(__CUDACC__)

#define MEMCPY(d,s,sz)  d_memcpy((U8*)(d), (U8*)(s), (U32)(sz))
#define MEMSET(d,v,sz)  d_memset((U8*)(d), (U8)(v),  (U32)(sz))
#define MEMCMP(d,s,sz)  d_memcmp((U8*)(d), (U8*)(s), (U32)(sz))

#define STRLEN(s)		d_strlen((U8*)(s), 0)
#define STRLENB(s)		d_strlen((U8*)(s), 1)
#define STRCPY(d,s)		d_strcpy((U8*)d, (U8*)s)
#define STRCMP(d,s)    	d_strcmp((U8*)d, (U8*)s)
#define STRCHR(d,c)     d_strchr((U8*)d, (U8)c)
#define STRCAT(d,s)     d_strcat((U8*)d, (U8*)s)
#define STRCUT(d,sz)	d_strcut((U8*)d, (U32)sz)

#define HASH(s)			d_calc_hash((U8*)s)

#else

#define MEMCPY(d,s,sz)  memcpy(d, s, sz)
#define MEMSET(d,v,sz)  memset(d, v, sz)
#define MEMCMP(d,s,sz)  memcmp(d, s, sz)

#define STRLEN(s)		strlen(s)
#define STRCPY(d,s)	  	strcpy(d, s)
#define STRCMP(s1,s2)   strcmp(s1, s2)
#define STRCHR(d,c)     strchr(d, c)
#define STRCAT(d,s)     strcat(d, s)
#define STRCUT(s,sz)	substr(s, sz)

#define HASH(s)			calc_hash((U8*)s)		// add implementation

#endif

#ifdef __cplusplus
}
#endif
#endif
