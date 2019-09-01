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

#include <stdint.h>
#include "guru.h"
#include "vm.h"

#ifdef __cplusplus
extern "C" {
#endif

#if GURU_HOST_IMAGE
// bytecode parsed by HOST, image passed into GPU
__HOST__ void  guru_parse_bytecode(guru_vm *vm, U8P ptr);
__HOST__ void  guru_show_irep(guru_irep *irep);
#else
// bytecode parsed directly by GPU
__GPU__  void  mrbc_parse_bytecode(mrbc_vm *vm, U8P ptr);
__HOST__ void  mrbc_show_irep(mrbc_irep *irep);
#endif

#ifdef __cplusplus
}
#endif
#endif
