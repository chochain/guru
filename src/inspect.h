/*! @file
  @brief
  GURU Inspection functions for each classes (use Aspect programming instead of OO)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_INSPECT_H_
#define GURU_SRC_INSPECT_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__CFUNC__ 	gr_to_s(GR r[], U32 ri);
__CFUNC__	int_chr(GR r[], U32 ri);
__CFUNC__ 	ary_join(GR r[], U32 ri);

__CFUNC__	gr_sprintf(GR r[], U32 ri);
__CFUNC__	gr_printf(GR r[], U32 ri);

#ifdef __cplusplus
}
#endif
#endif
