/*! @file
  @brief
  GURU common values and constructor/destructor/comparator registry

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_BASE_H_
#define GURU_SRC_BASE_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*guru_init_func)(...);
typedef void (*guru_destroy_func)(GR *r);
typedef S32  (*guru_cmp_func)(const GR *r0, const GR *r1);

__GURU__ void 	guru_register_func(GT t, guru_init_func fi, guru_destroy_func fd, guru_cmp_func fc);
__GURU__ GR		guru_new(...);
__GURU__ GR     guru_pack(GR *r);
__GURU__ void   guru_splat(GR *r1, GR *r0, U32 n);
__GURU__ void	guru_destroy(GR *r);
__GURU__ S32  	guru_cmp(const GR *r0, const GR *r1);

__GURU__ GR 	*ref_dec(GR *r);
__GURU__ GR 	*ref_inc(GR *r);

// macro for C call returns
// Note: becareful, the following macros assume a "v" pointer to top of stack
//
extern __GURU__ GR 	NIL;
extern __GURU__ GR 	EMPTY;

#define RETURN_VAL(n)	{ ref_dec(r); *r=(n); 		 			return; }
#define RETURN_NIL()	{ ref_dec(r); r->acl=0; r->gt=GT_NIL;   return; }
#define RETURN_FALSE()	{ ref_dec(r); r->acl=0; r->gt=GT_FALSE; return; }
#define RETURN_TRUE()	{ ref_dec(r); r->acl=0; r->gt=GT_TRUE;  return; }
     
#define RETURN_BOOL(n)	{ ref_dec(r); r->acl=0; r->gt=(n)?GT_TRUE:GT_FALSE;  return; }
#define RETURN_INT(n)	{ ref_dec(r); r->acl=0; r->gt=GT_INT;  r->i=(GI)(n); return; }
#define RETURN_FLOAT(n)	{ ref_dec(r); r->acl=0; r->gt=GT_FLOAT;r->f=(GF)(n); return; }

#ifdef __cplusplus
}
#endif
#endif // GURU_SRC_BASE_H_
