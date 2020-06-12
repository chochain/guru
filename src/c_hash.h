/*! @file
  @brief
  GURU Hash class

  <pre>
  Copyright (C) 2019- GreenII

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
    GURU_HDR;
    GR *data;

    // TODO: and other member for search.
} guru_hash;

__GURU__ GR 		guru_hash_new(int size);
__GURU__ void       guru_hash_del(GR *hash);
__GURU__ int        guru_hash_cmp(const GR *r0, const GR *r1);

__GURU__ void       guru_init_class_hash();


#ifdef __cplusplus
}
#endif
#endif // GURU_SRC_C_HASH_H_
