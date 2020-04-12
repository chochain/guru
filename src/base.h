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

typedef void (*guru_func)(GV *v);

__GURU__ GV 	NIL();
__GURU__ GV 	EMPTY();

__GURU__ void 	guru_register_init_func(GT gt, guru_func f);
__GURU__ void 	guru_register_destroy_func(GT gt, guru_func f);
__GURU__ void 	guru_register_cmp_func(GT gt, guru_func f);

__GURU__ GV		guru_new(GT gt);
__GURU__ GV		guru_destroy(GV *gv);
__GURU__ S32  	guru_cmp(const GV *v0, const GV *v1);

__GURU__ GV 	*ref_get(GV *v);
__GURU__ GV		*ref_free(GV *v);

__GURU__ GV 	*ref_dec(GV *v);
__GURU__ GV 	*ref_inc(GV *v);

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

#ifdef __cplusplus
}
#endif
#endif
