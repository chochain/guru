/*! @file
  @brief
  GURU global objects.

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef GURU_SRC_GLOBAL_H_
#define GURU_SRC_GLOBAL_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void 	global_set(GS xid, GV *v);
__GURU__ void 	const_set(GS xid,  GV *v);

__GURU__ GV 	*global_get(GS xid);
__GURU__ GV 	*const_get(GS xid);
    
#ifdef __cplusplus
}
#endif
#endif
