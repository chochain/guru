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

__GURU__ GS     new_sym(const U8 *str);			// create new symbol, returns sid
__GURU__ GS   	name2id(const U8 *str);			// sid by name
__GURU__ U8		*id2name(GS sid);				// name by sid

__GURU__ void 	guru_init_class_symbol();
__GURU__ GV 	guru_sym_new(const U8 *str);

#if GURU_DEBUG
__HOST__ void 	id2name_host(GS sid, U8 *str);	// ~= id2name, by host mode
#endif // GURU_DEBUG

#ifdef __cplusplus
}
#endif
#endif
