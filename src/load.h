/*! @file
  @brief
  Guru bytecode loader.

  <pre>
  Copyright (C) 2015 Kyushu Institute of Technology.
  Copyright (C) 2015 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_LOAD_H_
#define GURU_SRC_LOAD_H_

#include <stdint.h>
#include "guru.h"
#include "vm.h"

#ifdef __cplusplus
extern "C" {
#endif

#if GURU_HOST_IMAGE
__host__   void  guru_parse_bytecode(guru_vm *vm, const uint8_t *ptr);
#else
__global__ void  guru_parse_bytecode(mrbc_vm *vm, const uint8_t *ptr);
#endif

#ifdef __cplusplus
}
#endif
#endif
