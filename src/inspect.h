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
#include "puts.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void guru_na(const U8 *msg);
__GURU__ void gv_to_s(GV v[], U32 vi);
__GURU__ void gv_join(GV v[], U32 vi);

#ifdef __cplusplus
}
#endif
#endif
