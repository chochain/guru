/*! @file
  @brief
  console output module. (not yet input)

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef MRBC_SRC_CONSOLE_H_
#define MRBC_SRC_CONSOLE_H_

#include <stdint.h>
#include <stdarg.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*! printf tiny (mruby/c) version data container.
*/
typedef struct RPrintfFormat {
    char 			type;				//!< format char. (e.g. 'd','f','x'...)
    unsigned int 	flag_plus  : 1;
    unsigned int 	flag_minus : 1;
    unsigned int 	flag_space : 1;
    unsigned int 	flag_zero  : 1;
    int 			width;				//!< display width. (e.g. %10d as 10)
    int 			precision;			//!< precision (e.g. %5.2f as 2)
} mrbc_print_fmt;
    
typedef struct RPrintf {
    char 			*buf;		//!< output buffer.
    const char 		*buf_end;	//!< output buffer end point.
    char 			*p;			//!< output buffer write point.
    const char 		*fstr;		//!< format string. (e.g. "%d %03x")
    
    mrbc_print_fmt 	fmt;
} mrbc_printf;

__GURU__ void console_printf(const char *fstr, ...);
__GURU__ void console_print(const char *str);
__GURU__ void console_nprint(const char *str, int size);
__GURU__ void console_putchar(char c);

__global__ void guru_init_console_buf(uint8_t *buf, size_t sz);
    
#ifdef __cplusplus
}
#endif
#endif
