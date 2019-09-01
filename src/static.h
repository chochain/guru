/*! @file
  @brief
  Declare static data.

  <pre>
  Copyright (C) 2015-2016 Kyushu Institute of Technology.
  Copyright (C) 2015-2016 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/

#ifndef GURU_SRC_STATIC_H_
#define GURU_SRC_STATIC_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Class Tree */
extern __GURU__ guru_class *guru_class_object;

extern __GURU__ guru_class *guru_class_false;
extern __GURU__ guru_class *guru_class_true;
extern __GURU__ guru_class *guru_class_nil;
extern __GURU__ guru_class *guru_class_int;
#if GURU_USE_FLOAT
extern __GURU__ guru_class *guru_class_float;
extern __GURU__ guru_class *guru_class_math;
#endif
extern __GURU__ guru_class *guru_class_symbol;

extern __GURU__ guru_class *guru_class_proc;

#if GURU_USE_STRING
extern __GURU__ guru_class *guru_class_string;
#endif
#if GURU_USE_ARRAY
extern __GURU__ guru_class *guru_class_array;
extern __GURU__ guru_class *guru_class_range;
extern __GURU__ guru_class *guru_class_hash;
#endif

#ifdef __cplusplus
}
#endif
#endif
