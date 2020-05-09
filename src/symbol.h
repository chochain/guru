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

__GURU__ GS     create_sym(const U8 *str);		// create new symbol, returns sid
__GURU__ GS   	name2id(const U8 *str);			// sid by name
__GURU__ U8		*id2name(GS sid);				// name by sid

__GURU__ void   guru_sym_rom(GR *r);			// ROMable symbol
__GURU__ GR		guru_sym_new(const U8 *str);	// create a symbol GR

#ifdef __cplusplus
}
#endif
#endif
