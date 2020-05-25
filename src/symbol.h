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

// struct RSymbol, and guru_sym are defined in guru.h

__GURU__ GS   	name2id(const U8 *str);			// sid by name
__GURU__ GP		id2name(GS sid);				// name (string) offset by sid

__GURU__ void   guru_sym_transcode(GR *r);		// ROMable symbol

#ifdef __cplusplus
}
#endif
#endif
