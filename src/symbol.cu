/*! @file
  @brief
  Guru Symbol class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>

#include "value.h"
#include "alloc.h"
#include "symbol.h"

#if !defined(MRBC_SYMBOL_SEARCH_LINER) && !defined(MRBC_SYMBOL_SEARCH_BTREE)
#define MRBC_SYMBOL_SEARCH_BTREE
#endif

#ifndef MRBC_SYMBOL_TABLE_INDEX_TYPE
#define MRBC_SYMBOL_TABLE_INDEX_TYPE	uint16_t
#endif

struct SYM_INDEX {
    uint16_t hash;	//!< hash value, returned by calc_hash().
#ifdef MRBC_SYMBOL_SEARCH_BTREE
    MRBC_SYMBOL_TABLE_INDEX_TYPE left;
    MRBC_SYMBOL_TABLE_INDEX_TYPE right;
#endif
    const char *cstr;	//!< point to the symbol string.
};

__GURU__ struct SYM_INDEX sym_index[MAX_SYMBOLS_COUNT];
__GURU__ int sym_index_pos;	// point to the last(free) sym_index array.

//================================================================
/*! search index table
 */
__GURU__ int search_index(uint16_t hash, const char *str)
{
#ifdef MRBC_SYMBOL_SEARCH_LINER
    int i;
    for(i = 0; i < sym_index_pos; i++) {
        if (sym_index[i].hash==hash && strcmp(str, sym_index[i].cstr)==0) {
            return i;
        }
    }
    return -1;
#endif

#ifdef MRBC_SYMBOL_SEARCH_BTREE
    int i = 0;
    do {
        if (sym_index[i].hash==hash && guru_strcmp(str, sym_index[i].cstr)==0) {
            return i;
        }
        if (hash < sym_index[i].hash) {
            i = sym_index[i].left;
        } else {
            i = sym_index[i].right;
        }
    } while(i != 0);
    return -1;
#endif
}

//================================================================
/*! add to index table
 */
__GURU__ int add_index(uint16_t hash, const char *str)
{
    // check overflow.
    if (sym_index_pos >= MAX_SYMBOLS_COUNT) {
//        printf("Overflow %s for '%s'\n", "MAX_SYMBOLS_COUNT", str);
        assert(1==0);
        return -1;
    }
    int sym_id = sym_index_pos++;

    // append table.
    sym_index[sym_id].hash = hash;
    sym_index[sym_id].cstr = str;

#ifdef MRBC_SYMBOL_SEARCH_BTREE
    int i = 0;

    while(1) {
        if (hash < sym_index[i].hash) {
            // left side
            if (sym_index[i].left==0) {	// left is empty?
                sym_index[i].left = sym_id;
                break;
            }
            i = sym_index[i].left;
        } else {
            // right side
            if (sym_index[i].right==0) {	// right is empty?
                sym_index[i].right = sym_id;
                break;
            }
            i = sym_index[i].right;
        }
    }
#endif
    return sym_id;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  str	String
  @return 	symbol object
*/
__GURU__ mrbc_value mrbc_symbol_new(const char *str)
{
    mrbc_value ret = {.tt = MRBC_TT_SYMBOL};
    uint16_t h = calc_hash(str);
    mrbc_sym sym_id = search_index(h, str);

    if (sym_id >= 0) {
        ret.i = sym_id;
        return ret;		// already exist.
    }

    // create symbol object dynamically.
    int size = guru_strlen(str) + 1;
    char *buf = (char *)mrbc_raw_alloc(size);
    if (buf==NULL) return ret;		// ENOMEM raise?

    MEMCPY((uint8_t *)buf, (const uint8_t *)str, size);
    ret.i = add_index(h, buf);

    return ret;
}

//================================================================
/*! Calculate hash value.

  @param  str		Target string.
  @return uint16_t	Hash value.
*/
__GURU__ uint16_t calc_hash(const char *str)
{
    uint16_t h = 0;

    while(*str != '\0') {
        h = h * 37 + *str;
        str++;
    }
    return h;
}

//================================================================
/*! Convert string to symbol value.

  @param  str		Target string.
  @return mrbc_sym	Symbol value.
*/
__GURU__ mrbc_sym str_to_symid(const char *str)
{
    uint16_t h = calc_hash(str);
    mrbc_sym sym_id = search_index(h, str);
    if (sym_id >= 0) return sym_id;

    return add_index(h, str);
}


//================================================================
/*! Convert symbol value to string.

  @param  mrbc_sym	Symbol value.
  @return const char*	String.
  @retval NULL		Invalid sym_id was given.
*/
__GURU__ const char * symid_to_str(mrbc_sym sym_id)
{
    if (sym_id < 0) return NULL;
    if (sym_id >= sym_index_pos) return NULL;

    return sym_index[sym_id].cstr;
}

#if MRBC_USE_ARRAY
//================================================================
/*! (method) all_symbols
 */
__GURU__ void c_all_symbols(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_array_new(sym_index_pos);

    int i;
    for(i = 0; i < sym_index_pos; i++) {
        mrbc_value sym1 = {.tt = MRBC_TT_SYMBOL};
        sym1.i = i;
        mrbc_array_push(&ret, &sym1);
    }
    SET_RETURN(ret);
}
#endif

#if MRBC_USE_STRING
//================================================================
/*! (method) inspect
 */
static void c_inspect(mrbc_vm *vm, mrbc_value v[], int argc)
{
    const char *s = symid_to_str(v[0].i);
    v[0] = mrbc_string_new_cstr(vm, ":");
    mrbc_string_append_cstr(&v[0], s);
}


//================================================================
/*! (method) to_s
 */
static void c_to_s(mrbc_vm *vm, mrbc_value v[], int argc)
{
    v[0] = mrbc_string_new_cstr(vm, symid_to_str(v[0].i));
}
#endif

//================================================================
/*! get c-language string (char *)
*/
__GURU__ const char * mrbc_symbol_cstr(const mrbc_value *v)
{
  return symid_to_str(v->i);
}

