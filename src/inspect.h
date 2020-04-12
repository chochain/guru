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

__CFUNC__ 	gv_to_s(GV v[], U32 vi);
__CFUNC__ 	ary_join(GV v[], U32 vi);

__CFUNC__	gv_sprintf(GV v[], U32 vi);
__CFUNC__	gv_printf(GV v[], U32 vi);

#ifdef __cplusplus
}
#endif
#endif
