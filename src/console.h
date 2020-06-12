/*! @file
  @brief
  GURU console output module. (not yet input)

  <pre>
  Copyright (C) 2019 GreenII

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
/*! printf internal version data container.
*/
typedef struct print_node {
	U32	id   : 6;
    GT  gt 	 : 5;
    GT	fmt	 : 5;
    U32	size : 16;
    U8	data[];          								// different from *data
} guru_print_node;

__GPU__  void guru_console_init(U8 *buf, U32 sz);

__GURU__ void console_int(GI i);
__GURU__ void console_hex(GI i);
__GURU__ void console_ptr(void *ptr);
__GURU__ void console_float(GF f);
__GURU__ void console_char(U8 c);
__GURU__ void console_str(const U8 *str);				// instead of U8P, too many static string
__GURU__ void console_na(const U8 *msg);

__HOST__ void guru_console_flush(U8 *output_buf, U32 trace);
    
#ifdef __cplusplus
}
#endif
#endif // GURU_SRC_CONSOLE_H_
