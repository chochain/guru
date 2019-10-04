/*! @file
  @brief
  mruby/c Range object

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_C_ITER_H_
#define GURU_SRC_C_ITER_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  Define Range object (same header of other objects)
*/
typedef struct RIter {		// 48-byte
    GURU_HDR;				// 8-byte
    GV  *ivar;
    GV 	*range;				// 16-byte
    GV 	*step;				// 16-byte
} guru_iter;

__GURU__ GV 	guru_iter_new(GV *obj, GV *step);
__GURU__ U32    guru_iter_next(GV *iter);
__GURU__ void	guru_iter_del(GV *v);


#ifdef __cplusplus
}
#endif
#endif
