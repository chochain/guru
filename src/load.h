/*! @file
  @brief
  GURU bytecode loader (host load IREP code, build image and copy into CUDA memory).

  alternatively, load_gpu.cu can be used for device image building
  <pre>
  Copyright (C) 2019- Greeni

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef GURU_SRC_LOAD_H_
#define GURU_SRC_LOAD_H_
#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RITE_HDR 	\
		U16 rsz;	\
		U16 psz;	\
		U16	isz;	\
		U16 ssz

typedef struct {
	RITE_HDR;
} rite_size;

typedef struct {
	RITE_HDR;
	guru_irep 	*reps;
	GV			*pool;
	U32			*iseq;
	U8			*stbl;
} GRIT;

#if GURU_HOST_IMAGE
__HOST__ U8 *parse_bytecode(GV **regs, U8 *src);	// parsed on HOST, image passed into GPU
#else
__GURU__ GRIT *parse_bytecode(U8 *src);				// parsed on HOST, image passed into GPU
#endif // GURU_HOST_IMAGE

#ifdef __cplusplus
}
#endif
#endif
