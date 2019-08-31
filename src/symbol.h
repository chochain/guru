/*! @file
  @brief
  Guru Symbol class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_SYMBOL_H_
#define GURU_SRC_SYMBOL_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ mrbc_value guru_sym_new(const U8P str);
__GURU__ U16        calc_hash(const U8P str);
__GURU__ guru_sym   name2symid(const U8P str);
__GURU__ U8P        symid2name(guru_sym sid);

__GURU__ void mrbc_init_class_symbol();

#ifdef __cplusplus
}
#endif
#endif
