/*! @file
  @brief
  If you want to add your own extension,
  add your code in c_ext.c and c_ext.h. 

  <pre>
  Copyright (C) 2015 Kyushu Institute of Technology.
  Copyright (C) 2015 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef MRBC_SRC_C_EXT_H_
#define MRBC_SRC_C_EXT_H_

#include "vm_config.h"

#ifdef __cplusplus
extern "C" {
#endif

extern char *guru_init(size_t sz);
extern int   do_cuda(void);

#ifdef __cplusplus
}
#endif
#endif
