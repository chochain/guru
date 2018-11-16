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
#include "static.h"
#include "class.h"
#include "symbol.h"

#if GURU_USE_ARRAY
#include "c_array.h"
#endif

#if !defined(MRBC_SYMBOL_SEARCH_LINER) && !defined(MRBC_SYMBOL_SEARCH_BTREE)
#define MRBC_SYMBOL_SEARCH_BTREE
#endif

#ifndef MRBC_SYMBOL_TABLE_INDEX_TYPE
#define MRBC_SYMBOL_TABLE_INDEX_TYPE	uint16_t
#endif

struct SYM_LIST {
    uint16_t hash;	//!< hash value, returned by calc_hash().
#ifdef MRBC_SYMBOL_SEARCH_BTREE
    MRBC_SYMBOL_TABLE_INDEX_TYPE left;
    MRBC_SYMBOL_TABLE_INDEX_TYPE right;
#endif
    const char *cstr;	//!< point to the symbol string.
};

__GURU__ struct SYM_LIST sym_list[MAX_SYMBOLS_COUNT];
__GURU__ int sym_idx;	// point to the last(free) sym_list array.

//================================================================
/*! Calculate hash value.

  @param  str		Target string.
  @return uint16_t	Hash value.
*/
__GURU__ __INLINE__ uint16_t
_calc_hash(const char *str)
{
    uint16_t h = 0;

    while (*str != '\0') {
        h = h * 37 + *str;
        str++;
    }
    return h;
}

//================================================================
/*! search index table
 */
__GURU__ int
_search_index(const char *str)
{
    uint16_t   hash = _calc_hash(str);

#ifdef MRBC_SYMBOL_SEARCH_LINER
    for(int i = 0; i < sym_idx; i++) {
        if (sym_list[i].hash==hash && strcmp(str, sym_list[i].cstr)==0) {
            return i;
        }
    }
    return -1;
#endif

#ifdef MRBC_SYMBOL_SEARCH_BTREE
    int i = 0;
    do {
        if (sym_list[i].hash==hash &&
        		guru_strcmp(str, sym_list[i].cstr)==0) {
            return i;
        }
        i = (hash < sym_list[i].hash)
        		? sym_list[i].left
            	: sym_list[i].right;
    } while (i != 0);
    return -1;
#endif
}

//================================================================
/*! add to index table
 */
__GURU__ int
_add_index(const char *str)
{
    uint16_t hash = _calc_hash(str);

    // check overflow.
    if (sym_idx >= MAX_SYMBOLS_COUNT) {
    	assert(1==0);
    	return -1;
    }

    int sym_id = sym_idx++;

    // append table.
    sym_list[sym_id].hash = hash;
    sym_list[sym_id].cstr = str;

#ifdef MRBC_SYMBOL_SEARCH_BTREE
    int i = 0;

    while (1) {
        if (hash < sym_list[i].hash) {
            // left side
            if (sym_list[i].left==0) {	// left is empty?
                sym_list[i].left = sym_id;
                break;
            }
            i = sym_list[i].left;
        }
        else {
            // right side
            if (sym_list[i].right==0) {	// right is empty?
                sym_list[i].right = sym_id;
                break;
            }
            i = sym_list[i].right;
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
__GURU__ mrbc_value
mrbc_symbol_new(const char *str)
{
    mrbc_value v      = {.tt = MRBC_TT_SYMBOL};
    mrbc_sym   sym_id = _search_index(str);

    if (sym_id >= 0) {
        v.i = sym_id;
        return v;		// already exist.
    }

    // create symbol object dynamically.
    int     size = guru_strlen(str) + 1;
    uint8_t *buf = (uint8_t *)mrbc_alloc(size);
    if (buf==NULL) return v;		// ENOMEM raise?

    MEMCPY(buf, (uint8_t *)str, size);
    v.i = _add_index((const char *)buf);

    return v;
}

//================================================================
/*! Convert string to symbol value.

  @param  str		Target string.
  @return mrbc_sym	Symbol value.
*/
__GURU__ mrbc_sym
name2symid(const char *str)
{
    mrbc_sym sym_id = _search_index(str);
    if (sym_id >= 0) return sym_id;

    return _add_index(str);
}

//================================================================
/*! Convert symbol value to string.

  @param  mrbc_sym	Symbol value.
  @return const char*	String.
  @retval NULL		Invalid sym_id was given.
*/
__GURU__ const char*
symid2name(mrbc_sym sym_id)
{
    return (sym_id < 0 || sym_id >= sym_idx)
    		? NULL
    		: sym_list[sym_id].cstr;
}

#if GURU_USE_STRING
// from c_string.cu
extern "C" __GURU__ mrbc_value mrbc_string_new(const char *src);
extern "C" __GURU__ int        mrbc_string_append_cstr(mrbc_value *s1, const char *s2);

//================================================================
/*! (method) inspect
 */
__GURU__ void
c_inspect(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_new(":");

    mrbc_string_append_cstr(&ret, symid2name(v[0].i));

    SET_RETURN(ret);
}


//================================================================
/*! (method) to_s
 */
__GURU__ void
c_to_s(mrbc_value v[], int argc)
{
    v[0] = mrbc_string_new(symid2name(v[0].i));
}
#endif

#if GURU_USE_ARRAY
//================================================================
/*! (method) all_symbols
 */
__GURU__ void
c_all_symbols(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_array_new(sym_idx);

    for(int i = 0; i < sym_idx; i++) {
        mrbc_value sym1 = {.tt = MRBC_TT_SYMBOL};
        sym1.i = i;
        mrbc_array_push(&ret, &sym1);
    }
    SET_RETURN(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void mrbc_init_class_symbol()  // << from symbol.cu
{
    mrbc_class *c = mrbc_class_symbol = mrbc_define_class("Symbol", mrbc_class_object);

#if GURU_USE_ARRAY
    mrbc_define_method(c, "all_symbols", 	c_all_symbols);
#endif
#if GURU_USE_STRING
    mrbc_define_method(c, "inspect", 		c_inspect);
    mrbc_define_method(c, "to_s", 			c_to_s);
    mrbc_define_method(c, "id2name", 		c_to_s);
#endif
    mrbc_define_method(c, "to_sym", 		c_nop);
}




