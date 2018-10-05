/*! @file
  @brief
  Guru Symbol class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef MRBC_SRC_SYMBOL_H_
#define MRBC_SRC_SYMBOL_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

__GURU__ mrbc_value mrbc_symbol_new(const char *str);
__GURU__ uint16_t   calc_hash(const char *str);
__GURU__ mrbc_sym   str_to_symid(const char *str);
__GURU__ const char *symid_to_str(mrbc_sym sym_id);

//================================================================
/*! get c-language string (char *)
*/
__GURU__ const char * mrbc_symbol_cstr(const mrbc_value *v);

#ifdef __cplusplus
}
#endif
#endif
