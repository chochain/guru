/*! @file
  @brief
  console output module. (not yet input)

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef MRBC_SRC_SPRINTF_H_
#define MRBC_SRC_SPRINTF_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*! printf tiny (mruby/c) version data container.
 */
typedef struct RPrintfFormat {
        char         type;			//!< format char. (e.g. 'd','f','x'...)
        unsigned int plus  : 1;
        unsigned int minus : 1;
        unsigned int space : 1;
        unsigned int zero  : 1;
        int          width;			//!< display width. (e.g. %10d as 10)
        int          prec;		    //!< precision (e.g. %5.2f as 2)
} mrbc_print_fmt;

typedef struct RPrintf {
    mrbc_print_fmt 	fmt;
    char       		*buf;		    //!< output buffer.
    char       		*p;		        //!< output buffer write point.
    const char 		*end;	    	//!< output buffer end point.
    const char 		*fstr;	        //!< format string. (e.g. "%d %03x")
} mrbc_printf;

__GURU__ char *guru_sprintf(const char *fstr, ...);
__GURU__ char *guru_vprintf(const char *fstr, mrbc_value v[], int argc);

#ifdef __cplusplus
}
#endif
#endif
