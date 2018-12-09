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
    mrbc_vtype  tt :8;
    mrbc_vtype	fmt:8;
    uint16_t	size;
	uint8_t		data[];
} guru_print_node;

__GURU__ void console_char(char c);
__GURU__ void console_int(mrbc_int i);
__GURU__ void console_hex(mrbc_int i);
__GURU__ void console_str(const char *str);
__GURU__ void console_float(mrbc_float f);
__GURU__ void console_na(const char *msg);

__GPU__ void guru_console_init(uint8_t *buf, size_t sz);
__HOST__   void guru_console_flush(uint8_t *output_buf);
    
#ifdef __cplusplus
}
#endif
#endif
