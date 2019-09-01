/*! @file
  @brief
  GURU Symbol class

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_SYMBOL_H_
#define GURU_SRC_SYMBOL_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ GV 	guru_sym_new(const U8P str);
__GURU__ GS   	name2id(const U8P str);
__GURU__ U8P    id2name(GS sid);

__GURU__ void 	guru_init_class_symbol();

#ifdef __cplusplus
}
#endif
#endif
