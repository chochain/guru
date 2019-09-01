/*! @file
  @brief
  Manage global objects.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_GLOBAL_H_
#define GURU_SRC_GLOBAL_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GPU__  void 		guru_global_init(void);

__GURU__ void 		global_object_add(GS sid, guru_obj *obj);
__GURU__ void 		const_object_add(GS sid,  guru_obj *obj);

__GURU__ guru_obj 	global_object_get(GS sid);
__GURU__ guru_obj 	const_object_get(GS sid);
    
#ifdef __cplusplus
}
#endif
#endif
