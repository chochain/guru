/*! @file
  @brief
  Declare static data.

  <pre>
  Copyright (C) 2015-2016 Kyushu Institute of Technology.
  Copyright (C) 2015-2016 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/

#include "guru.h"

/* Class Tree */
__GURU__ guru_class *guru_class_object;

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
#endif
#if GURU_USE_STRING
__GURU__ guru_class *guru_class_string;
#endif
#if GURU_USE_ARRAY
__GURU__ guru_class *guru_class_array;
__GURU__ guru_class *guru_class_range;
__GURU__ guru_class *guru_class_hash;
#endif

