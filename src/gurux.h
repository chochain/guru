/*! @file
  @brief
  GURU external type and interface definitions
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

typedef uintptr_t   U32P;
typedef uint8_t     *U8P;

U32 guru_setup(U32 trace);
U32 guru_load(U8 **argv, U32 n, U32 trace);
U32 guru_run(U32 trace);

#ifdef __cplusplus
}
#endif

#endif
