/*! @file
  @brief
  console output module. (not yet input)

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_SPRINTF_H_
#define GURU_SRC_SPRINTF_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*! printf tiny (mruby/c) version data container.
 */
typedef struct RPrintFormat {
	U32 			type  : 8;			//!< format char. (e.g. 'd','f','x'...)
    U32 			plus  : 1;
    U32 			minus : 1;
    U32 			space : 1;
    U32 			zero  : 1;
    U32 			prec  : 4;		    //!< precision (e.g. %5.2f as 2)
    U32 			width : 16;			//!< display width. (e.g. %10d as 10)
} guru_print_fmt;

typedef struct RPrint {
    guru_print_fmt 	fmt;
    const U8 		*fstr;				//!< format string. (e.g. "%d %03x")
    const U8       	*buf;				//!< output buffer.
    const U8		*end;				//!< output buffer end point.
    U8       		*p;					//!< output buffer write point.
} guru_print;

__GURU__ void guru_printf(const U8 *fstr, ...);							// fstr is always static string (char *)
__GURU__ void guru_vprintf(const U8 *fstr, GV v[], U32 argc);

#ifdef __cplusplus
}
#endif
#endif
