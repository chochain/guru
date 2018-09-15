/*! @file
  @brief
  Declare static data.

  <pre>
  Copyright (C) 2015-2016 Kyushu Institute of Technology.
  Copyright (C) 2015-2016 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/

#include "vm_config.h"
#include "guru.hu"
#include "global.hu"
#include "static.hu"

/* Class Tree */
__GURU__ mrbc_class *mrbc_class_object;

/* Classes */
__GURU__ mrbc_class *mrbc_class_false;
__GURU__ mrbc_class *mrbc_class_true;
__GURU__ mrbc_class *mrbc_class_nil;
__GURU__ mrbc_class *mrbc_class_fixnum;
__GURU__ mrbc_class *mrbc_class_symbol;

/* Proc */
__GURU__ mrbc_class *mrbc_class_proc;

#if MRBC_USE_FLOAT
__GURU__ mrbc_class *mrbc_class_float;
__GURU__ mrbc_class *mrbc_class_math;
#endif
#if MRBC_USE_STRING
__GURU__ mrbc_class *mrbc_class_string;
#endif
#if MRBC_USE_ARRAY
__GURU__ mrbc_class *mrbc_class_array;
__GURU__ mrbc_class *mrbc_class_range;
__GURU__ mrbc_class *mrbc_class_hash;
#endif

