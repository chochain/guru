/*! @file
  @brief

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.


  </pre>
*/

#ifndef GURU_SRC_PUTS_H_
#define GURU_SRC_PUTS_H_

#include "guru.h"

#if GURU_USE_CONSOLE	// use built-in print functions
#include "console.h"
#include "sprintf.h"
#define PRINTF		guru_printf
#define VPRINTF		guru_vprintf
#else					// use CUDA printf function
#include <stdio.h>
#define PRINTF		printf
#endif

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ void guru_puts(mrbc_value *v, U32 argc);
__GURU__ void guru_p(mrbc_value *v, U32 argc);
__GURU__ void guru_na(const U8 *msg);

#ifdef __cplusplus
}
#endif
#endif
