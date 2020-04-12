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
