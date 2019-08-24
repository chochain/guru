/*! @file
  @brief
  Guru value definitions

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_GURUX_H_
#define GURU_SRC_GURUX_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// short notation
typedef uint64_t    U64;
typedef uint32_t	U32;
typedef uint16_t    U16;
typedef uint8_t		U8;
typedef int32_t     S32;				// single precision float
typedef int16_t		S16;
typedef double		F64;
typedef float       F32;

// mrbc types
typedef S32 		mrbc_int;
typedef F32	 		mrbc_float;
typedef S16 		mrbc_sym;

typedef struct guru_ses_ {
	U8 	*in;
	U8	*out;
	U16 id;
	U16 trace;

	struct guru_ses_ *next;
} guru_ses;

int guru_system_setup(int trace);
int guru_system_run(int trace);
int guru_session_add(guru_ses *ses, const char *rite_fname, int trace);

#ifdef __cplusplus
}
#endif

#endif
