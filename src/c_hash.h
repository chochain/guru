/*! @file
  @brief
  mruby/c Hash class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef MRBC_SRC_C_HASH_H_
#define MRBC_SRC_C_HASH_H_

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
    MRBC_OBJECT_HEADER;

    uint16_t   data_size;	//!< data buffer size.
    uint16_t   n_stored;	//!< # of stored.
    mrbc_value *data;	    //!< pointer to allocated memory.

    // TODO: and other member for search.
} mrbc_hash;


//================================================================
/*!@brief
  Define Hash iterator.
*/
typedef struct RHashIterator {
    mrbc_hash *target;
    mrbc_value *point;
    mrbc_value *p_end;
} mrbc_hash_iterator;

__GURU__ mrbc_value mrbc_hash_new(int size);
__GURU__ void       mrbc_hash_delete(mrbc_value *hash);

__GURU__ mrbc_value *mrbc_hash_search(const mrbc_value *hash, const mrbc_value *key);
__GURU__ int        mrbc_hash_set(mrbc_value *hash, mrbc_value *key, mrbc_value *val);
__GURU__ mrbc_value mrbc_hash_get(mrbc_value *hash, mrbc_value *key);
__GURU__ mrbc_value mrbc_hash_remove(mrbc_value *hash, mrbc_value *key);
__GURU__ void       mrbc_hash_clear(mrbc_value *hash);

__GURU__ int        mrbc_hash_compare(const mrbc_value *v1, const mrbc_value *v2);
__GURU__ mrbc_value mrbc_hash_dup(mrbc_value *src);

__GURU__ mrbc_hash_iterator mrbc_hash_iterator_new(mrbc_value *v);
__GURU__ int        mrbc_hash_i_has_next(mrbc_hash_iterator *ite);
__GURU__ mrbc_value *mrbc_hash_i_next(mrbc_hash_iterator *ite);

__GURU__ void       mrbc_init_class_hash();


#ifdef __cplusplus
}
#endif
#endif
