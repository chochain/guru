/*! @file
  @brief
  Guru Key(Symbol) - Value store.

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include <string.h>

#include "value.h"
#include "alloc.h"
#include "keyvalue.h"
#include "errorcode.h"

//================================================================
/*! binary search

  @param  kvh		pointer to key-value handle.
  @param  sym_id	symbol ID.
  @return		result. It's not necessarily found.
*/
__GURU__
int _binary_search(mrbc_kv_handle *kvh, mrbc_sym sym_id)
{
    int left  = 0;
    int right = kvh->n_stored - 1;
    if (right < 0) return -1;

    while (left < right) {
        int mid = (left + right) / 2;
        if (kvh->data[mid].sym_id < sym_id) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  size	initial size.
  @return 	Key-Value handle.
*/
__GURU__
mrbc_kv_handle * mrbc_kv_new(int size)
{
    /*
      Allocate handle and data buffer.
    */
    mrbc_kv_handle *kvh = (mrbc_kv_handle *)mrbc_alloc(sizeof(mrbc_kv_handle));
    if (!kvh) return NULL;	// ENOMEM

    kvh->data = (mrbc_kv *)mrbc_alloc(sizeof(mrbc_kv) * size);
    if (!kvh->data) {		// ENOMEM
        mrbc_free(kvh);
        return NULL;
    }

    kvh->data_size = size;
    kvh->n_stored = 0;

    return kvh;
}

//================================================================
/*! destructor

  @param  kvh	pointer to key-value handle.
*/
__GURU__
void mrbc_kv_delete(mrbc_kv_handle *kvh)
{
    mrbc_kv_clear(kvh);

    mrbc_free(kvh->data);
    mrbc_free(kvh);
}

//================================================================
/*! resize buffer

  @param  kvh	pointer to key-value handle.
  @param  size	size.
  @return	mrbc_error_code.
*/
__GURU__
int mrbc_kv_resize(mrbc_kv_handle *kvh, int size)
{
    mrbc_kv *data2 = (mrbc_kv *) mrbc_realloc(kvh->data, sizeof(mrbc_kv) * size);
    if (!data2) return E_NOMEMORY_ERROR;		// ENOMEM

    kvh->data = data2;
    kvh->data_size = size;

    return 0;
}

//================================================================
/*! setter

  @param  kvh		pointer to key-value handle.
  @param  sym_id	symbol ID.
  @param  set_val	set value.
  @return		mrbc_error_code.
*/
__GURU__
int mrbc_kv_set(mrbc_kv_handle *kvh, mrbc_sym sym_id, mrbc_value *set_val)
{
    int idx = _binary_search(kvh, sym_id);
    if (idx < 0) {
        idx = 0;
        goto INSERT_VALUE;
    }

    // replace value ?
    if (kvh->data[idx].sym_id == sym_id) {
        mrbc_dec_refc(&kvh->data[idx].value);
        kvh->data[idx].value = *set_val;
        return 0;
    }

    if (kvh->data[idx].sym_id < sym_id) {
        idx++;
    }

INSERT_VALUE:
    // need resize?
    if (kvh->n_stored >= kvh->data_size) {
        if (mrbc_kv_resize(kvh, kvh->data_size + 5) != 0)
            return E_NOMEMORY_ERROR;		// ENOMEM
    }

    // need move data?
    if (idx < kvh->n_stored) {
        int size = sizeof(mrbc_kv) * (kvh->n_stored - idx);
        MEMCPY((uint8_t *)&kvh->data[idx+1], (const uint8_t *)&kvh->data[idx], size);
    }

    kvh->data[idx].sym_id = sym_id;
    kvh->data[idx].value = *set_val;
    kvh->n_stored++;

    return 0;
}

//================================================================
/*! getter

  @param  kvh		pointer to key-value handle.
  @param  sym_id	symbol ID.
  @return		pointer to mrbc_value or NULL.
*/
__GURU__
mrbc_value * mrbc_kv_get(mrbc_kv_handle *kvh, mrbc_sym sym_id)
{
    int idx = _binary_search(kvh, sym_id);
    if (idx < 0) return NULL;
    if (kvh->data[idx].sym_id != sym_id) return NULL;

    return &kvh->data[idx].value;
}

//================================================================
/*! setter - only append tail

  @param  kvh		pointer to key-value handle.
  @param  sym_id	symbol ID.
  @param  set_val	set value.
  @return		mrbc_error_code.
*/
__GURU__
int mrbc_kv_append(mrbc_kv_handle *kvh, mrbc_sym sym_id, mrbc_value *set_val)
{
    // need resize?
    if (kvh->n_stored >= kvh->data_size) {
        if (mrbc_kv_resize(kvh, kvh->data_size + 5) != 0)
            return E_NOMEMORY_ERROR;		// ENOMEM
    }

    kvh->data[kvh->n_stored].sym_id = sym_id;
    kvh->data[kvh->n_stored].value = *set_val;
    kvh->n_stored++;

    return 0;
}

__GURU__
int compare_key(const void *kv1, const void *kv2)
{
    return ((mrbc_kv *)kv1)->sym_id - ((mrbc_kv *)kv2)->sym_id;
}

//================================================================
/*! remove a data

  @param  kvh		pointer to key-value handle.
  @param  sym_id	symbol ID.
  @return		mrbc_error_code.
*/
__GURU__
int mrbc_kv_remove(mrbc_kv_handle *kvh, mrbc_sym sym_id)
{
    int idx = _binary_search(kvh, sym_id);
    if (idx < 0) return 0;
    if (kvh->data[idx].sym_id != sym_id) return 0;

    mrbc_dec_refc(&kvh->data[idx].value);
    kvh->n_stored--;
    MEMCPY((uint8_t *)(kvh->data + idx), (const uint8_t *)(kvh->data + idx + 1), sizeof(mrbc_kv) * (kvh->n_stored - idx));

    return 0;
}

//================================================================
/*! clear all

  @param  kvh		pointer to key-value handle.
*/
__GURU__
void mrbc_kv_clear(mrbc_kv_handle *kvh)
{
    mrbc_kv *p1 = kvh->data;
    const mrbc_kv *p2 = p1 + kvh->n_stored;
    while (p1 < p2) {
        mrbc_dec_refc(&p1->value);
        p1++;
    }
    kvh->n_stored = 0;
}

//================================================================
/*! get size
*/
__GURU__ __forceinline__
int mrbc_kv_size(const mrbc_kv_handle *kvh)
{
  return kvh->n_stored;
}



