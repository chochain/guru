/*! @file
  @brief
  mruby/c Hash class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef GURU_SRC_C_HASH_H_
#define GURU_SRC_C_HASH_H_

#include "guru.h"
#include "c_array.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  Define Hash handle.
*/
typedef struct RHash {
    // (NOTE)
    //  Needs to be same members and order as RArray.
    GURU_OBJECT_HEADER;

    uint32_t   size : 16;	//!< data buffer size.
    uint32_t   n	: 16;	//!< # of stored.
    mrbc_value *data;		//!< pointer to allocated memory.

    // TODO: and other member for search.
} mrbc_hash;

__GURU__ mrbc_value mrbc_hash_new(int size);
__GURU__ void       mrbc_hash_delete(mrbc_value *hash);
__GURU__ int        mrbc_hash_compare(const mrbc_value *v1, const mrbc_value *v2);

__GURU__ void       mrbc_init_class_hash();


#ifdef __cplusplus
}
#endif
#endif
