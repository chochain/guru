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
