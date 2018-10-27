/*! @file
  @brief
  Guru value definitions

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef MRBC_SRC_VALUE_H_
#define MRBC_SRC_VALUE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif
__GURU__ int   		mrbc_compare(const mrbc_value *v1, const mrbc_value *v2);
__GURU__ void  		mrbc_dec_refc(mrbc_value *v);
__GURU__ void  		mrbc_inc_refc(mrbc_value *v);
__GURU__ void  		mrbc_release(mrbc_value *v);

// for C call
#define SET_RETURN(n)		do { mrbc_value nnn = (n); mrbc_dec_refc(v); v[0] = nnn; 	} while (0)
#define SET_NIL_RETURN()	do { mrbc_dec_refc(v); v[0].tt = MRBC_TT_NIL;   			} while (0)
#define SET_FALSE_RETURN()	do { mrbc_dec_refc(v); v[0].tt = MRBC_TT_FALSE; 			} while (0)
#define SET_TRUE_RETURN()	do { mrbc_dec_refc(v); v[0].tt = MRBC_TT_TRUE;  			} while (0)
#define SET_BOOL_RETURN(n)	do { mrbc_dec_refc(v); v[0].tt = (n)?MRBC_TT_TRUE:MRBC_TT_FALSE; } while (0)
#define SET_INT_RETURN(n)	do { mrbc_int nnn = (n);					\
		mrbc_dec_refc(v); v[0].tt = MRBC_TT_FIXNUM; v[0].i = nnn; } while (0)
#define SET_FLOAT_RETURN(n)	do { mrbc_float nnn = (n);                  \
        mrbc_dec_refc(v); v[0].tt = MRBC_TT_FLOAT;  v[0].f = nnn; } while (0)

#define GET_TT_ARG(n)			(v[(n)].tt)
#define GET_INT_ARG(n)			(v[(n)].i)
#define GET_ARY_ARG(n)			(v[(n)])
#define GET_ARG(n)		    	(v[(n)])
#define GET_FLOAT_ARG(n)		(v[(n)].f)
#define GET_STRING_ARG(n)		(v[(n)].string->data)

#define mrbc_fixnum_value(n)	((mrbc_value){.tt = MRBC_TT_FIXNUM, .i=(n)})
#define mrbc_float_value(n)	    ((mrbc_value){.tt = MRBC_TT_FLOAT,  .f=(n)})
#define mrbc_nil_value()	    ((mrbc_value){.tt = MRBC_TT_NIL})
#define mrbc_true_value()	    ((mrbc_value){.tt = MRBC_TT_TRUE})
#define mrbc_false_value()	    ((mrbc_value){.tt = MRBC_TT_FALSE})
#define mrbc_bool_value(n)	    ((mrbc_value){.tt = (n)?MRBC_TT_TRUE:MRBC_TT_FALSE})

#ifdef __GURU_CUDA__
__GURU__ mrbc_int   guru_atoi(const char *s, int base);
__GURU__ mrbc_float	guru_atof(const char *s);

__GURU__ void    	guru_memcpy(uint8_t *d, const uint8_t *s, size_t sz);
__GURU__ void    	guru_memset(uint8_t *d, const uint8_t v,  size_t sz);
__GURU__ int     	guru_memcmp(const uint8_t *d, const uint8_t *s, size_t sz);

__GURU__ size_t  	guru_strlen(const char *s);
__GURU__ void    	guru_strcpy(const char *s1, const char *s2);
__GURU__ int     	guru_strcmp(const char *s1, const char *s2);
__GURU__ char   	*guru_strchr(const char *s, const char c);
__GURU__ char   	*guru_strcat(char *d, const char *s);


#define ATOI(s)           guru_atoi(s, 10)
#define ATOF(s)			  guru_atof(s)

#define MEMCPY(d, s, sz)  guru_memcpy(d, s, sz)
#define MEMSET(d, v, sz)  guru_memset(d, v, sz)
#define MEMCMP(d, s, sz)  guru_memcmp(d, s, sz)

#define STRLEN(s)		  guru_strlen(s)
#define STRCPY(s1, s2)	  guru_strcpy(s1, s2)
#define STRCMP(s1, s2)    guru_strcmp(s1, s2)
#define STRCHR(s, c)      guru_strchr(s, c)
#define STRCAT(d, s)      guru_strcat(d, s)
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
#define VSTRLEN(v)		((v)->str->size)
#define VSTR(v)			((char *)(v)->str->data)
#define VSTRCMP(v1, v2) (STRCMP((v1)->str->data, (v2)->str->data))
#define VSYM(v)			((char *)symid2name((v)->i))

#ifdef __cplusplus
}
#endif
#endif
