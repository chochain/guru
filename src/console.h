/*! @file
  @brief
  console output module. (not yet input)

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_CONSOLE_H_
#define GURU_SRC_CONSOLE_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*! printf tiny (mruby/c) version data container.
*/
typedef struct print_node {
	U32			id   : 6;
    mrbc_vtype  tt 	 : 5;
    mrbc_vtype	fmt	 : 5;
    U32			size : 16;

    U8			*data;
} guru_print_node;

__GPU__  void guru_console_init(U8 *buf, U32 sz);

__GURU__ void console_char(U8 c);
__GURU__ void console_int(mrbc_int i);
__GURU__ void console_hex(mrbc_int i);
__GURU__ void console_str(const U8 *str);
__GURU__ void console_float(mrbc_float f);
__GURU__ void console_na(const U8 *msg);

__HOST__ void guru_console_flush(U8 *output_buf);
    
#ifdef __cplusplus
}
#endif
#endif
