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

__GURU__ void        mrbc_init_global(void);
    
__GURU__ void        global_object_add(mrbc_sym sid, mrbc_value v);
__GURU__ void        const_object_add(mrbc_sym sid, mrbc_object *obj);

__GURU__ mrbc_value  global_object_get(mrbc_sym sid);
__GURU__ mrbc_object const_object_get(mrbc_sym sid);
    
#ifdef __cplusplus
}
#endif
#endif
