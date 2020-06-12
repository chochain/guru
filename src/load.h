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
	U16 rsz;		\
	U16 psz;		\
	U16	isz;		\
	U16 ssz

typedef struct {
	RITE_HDR;
} rite_size;

typedef struct {
	RITE_HDR;
	S16		reps;
	S16		pool;
	S16		iseq;
	S16		stbl;
} GRIT;

#if GURU_HOST_GRIT_IMAGE
#define __CODE__	__HOST__
#define __MEMCPY 	memcpy
#define __MEMCMP	memcmp
#define	__ATOI		atoi
#define __ATOF		atof
#else
#define __CODE__ 	__GURU__
#define __MEMCPY	MEMCPY
#define __MEMCMP	MEMCMP
#define __ATOI(v)	ATOI(v,10)
#define __ATOF(v)	ATOF(v)
#endif // GURU_HOST_GRIT_IMAGE

__CODE__ U8 *parse_bytecode(U8 *src);		// parsed on HOST, image passed into GPU
#ifdef __cplusplus
}
#endif
#endif // GURU_SRC_LOAD_H_
