/*! @file
  @brief
  GURU - static data declarations

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/

#ifndef GURU_SRC_STATIC_H_
#define GURU_SRC_STATIC_H_

#include "guru.h"
#include "class.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Class Tree */
extern __GURU__ guru_class *guru_class_object;

extern __GURU__ guru_class *guru_class_false;
extern __GURU__ guru_class *guru_class_true;
extern __GURU__ guru_class *guru_class_nil;
extern __GURU__ guru_class *guru_class_int;
extern __GURU__ guru_class *guru_class_symbol;
extern __GURU__ guru_class *guru_class_proc;

#if GURU_USE_FLOAT
extern __GURU__ guru_class *guru_class_float;
extern __GURU__ guru_class *guru_class_math;
#endif // GURU_USE_FLOAT

#if GURU_USE_STRING
extern __GURU__ guru_class *guru_class_string;
#endif // GURU_USE_STRING

#if GURU_USE_ARRAY
extern __GURU__ guru_class *guru_class_array;
extern __GURU__ guru_class *guru_class_range;
extern __GURU__ guru_class *guru_class_hash;
#endif  // GURU_USE_ARRAY

#ifdef __cplusplus
}
#endif
#endif
