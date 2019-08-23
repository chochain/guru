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

typedef struct guru_ses_ {
	uint8_t *in ;
	uint8_t *out;
	uint32_t id    : 16;
	uint32_t trace : 16;
	struct guru_ses_ *next;
} guru_ses;

// mrbc types
typedef int32_t 	mrbc_int;
typedef float 		mrbc_float;
typedef int16_t 	mrbc_sym;

int guru_system_setup(int trace);
int guru_system_run(int trace);

int guru_session_add(guru_ses *ses, const char *rite_fname, int trace);

#ifdef __cplusplus
}
#endif

#endif
