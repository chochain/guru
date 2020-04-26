/*! @file
  @brief
  GURU String object

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_C_STRING_H_
#define GURU_SRC_C_STRING_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  Define String handle.
*/
__GURU__ GV 	guru_str_new(const U8 *str);				// U8P will require to many casting
__GURU__ GV		guru_str_buf(U32 sz);						// a string buffer with given length
__GURU__ GV		guru_str_clr(GV *s);						// reset str->sz,bsz to zero
__GURU__ void   guru_str_del(GV *s);
__GURU__ S32    guru_str_cmp(const GV *s0, const GV *s1);

__GURU__ GV     guru_str_add(GV *s0, GV *s1);				//	return a new string
__GURU__ GV     guru_buf_add_cstr(GV *buf, const U8 *str);	//  return the same s0

__GURU__ void   guru_init_class_string(void);

#ifdef __cplusplus
}
#endif
#endif
