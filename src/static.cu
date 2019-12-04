/*! @file
  @brief
  GURU - static data declarations

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include "guru.h"
#include "static.h"

/* Class Tree */
__GURU__ guru_class *guru_class_object;
__GURU__ guru_class *guru_class_sys;

/* Classes */
__GURU__ guru_class *guru_class_false;
__GURU__ guru_class *guru_class_true;
__GURU__ guru_class *guru_class_nil;
__GURU__ guru_class *guru_class_int;
__GURU__ guru_class *guru_class_symbol;

/* Proc */
__GURU__ guru_class *guru_class_proc;

#if GURU_USE_FLOAT
__GURU__ guru_class *guru_class_float;
__GURU__ guru_class *guru_class_math;
#endif // GURU_USE_FLOAT

__GURU__ guru_class *guru_class_string;
#if GURU_USE_ARRAY
__GURU__ guru_class *guru_class_array;
__GURU__ guru_class *guru_class_range;
__GURU__ guru_class *guru_class_hash;
#endif // GURU_USE_ARRAY





