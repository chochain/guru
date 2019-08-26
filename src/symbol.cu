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

#if !defined(GURU_SYMBOL_SEARCH_LINER) && !defined(GURU_SYMBOL_SEARCH_BTREE)
#define GURU_SYMBOL_SEARCH_BTREE
#endif

#ifndef GURU_SYMBOL_TABLE_INDEX_TYPE
#define GURU_SYMBOL_TABLE_INDEX_TYPE U16
#endif

struct SYM_LIST {
    U16 							hash;		//!< hash value, returned by calc_hash().
#ifdef GURU_SYMBOL_SEARCH_BTREE
    GURU_SYMBOL_TABLE_INDEX_TYPE 	left;
    GURU_SYMBOL_TABLE_INDEX_TYPE 	right;
#endif
    U8P 							cstr;		//!< point to the symbol string.
};

__GURU__ U32 _sym_idx;							// point to the last(free) sym_list array.
__GURU__ struct SYM_LIST _sym_list[MAX_SYMBOLS_COUNT];

//================================================================
/*! Calculate hash value.

  @param  str		Target string.
  @return uint16_t	Hash value.
*/
__GURU__ __INLINE__ U32
_calc_hash(U8P str)
{
    U32 h = 0;

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
_search_index(const U8P str)
{
    U32 hash = _calc_hash(str);

#ifdef GURU_SYMBOL_SEARCH_LINER
    for (U32 i=0; i < _sym_idx; i++) {
        if (_sym_list[i].hash==hash && strcmp(str, _sym_list[i].cstr)==0) {
            return i;
        }
    }
    return -1;
#endif

#ifdef GURU_SYMBOL_SEARCH_BTREE
    int i = 0;
    do {
        if (_sym_list[i].hash==hash &&
        		guru_strcmp(str, _sym_list[i].cstr)==0) {
            return i;
        }
        i = (hash < _sym_list[i].hash)
        		? _sym_list[i].left
            	: _sym_list[i].right;
    } while (i != 0);
    return -1;
#endif
}

//================================================================
/*! add to index table
 */
__GURU__ S32
_add_index(const U8P str)
{
    U32 hash = _calc_hash(str);

    // check overflow.
    if (_sym_idx >= MAX_SYMBOLS_COUNT) {
    	assert(1==0);
    	return -1;
    }

    U32 sid = _sym_idx++;

    // append table.
    _sym_list[sid].hash = hash;
    _sym_list[sid].cstr = str;

#ifdef GURU_SYMBOL_SEARCH_BTREE
    U32 i = 0;

    while (1) {
        if (hash < _sym_list[i].hash) {
            // left side
            if (_sym_list[i].left==0) {	// left is empty?
                _sym_list[i].left = sid;
                break;
            }
            i = _sym_list[i].left;
        }
        else {
            // right side
            if (_sym_list[i].right==0) {	// right is empty?
                _sym_list[i].right = sid;
                break;
            }
            i = _sym_list[i].right;
        }
    }
#endif
    return sid;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  str	String
  @return 	symbol object
*/
__GURU__ mrbc_value
mrbc_symbol_new(const U8P str)
{
    mrbc_value v   = {.tt = GURU_TT_SYMBOL};
    mrbc_sym   sid = _search_index(str);

    if (sid >= 0) {
        v.i = sid;
        return v;		// already exist.
    }

    // create symbol object dynamically.
    U32 size = guru_strlen(str) + 1;
    U8P buf  = (U8P)mrbc_alloc(size);
    if (buf==NULL) return v;		// ENOMEM raise?

    MEMCPY(buf, str, size);
    v.i = _add_index(buf);

    return v;
}

//================================================================
/*! Convert string to symbol value.

  @param  str		Target string.
  @return mrbc_sym	Symbol value.
*/
__GURU__ mrbc_sym
name2symid(const U8P str)
{
    mrbc_sym sid = _search_index(str);
    if (sid >= 0) return sid;

    return _add_index(str);
}

//================================================================
/*! Convert symbol value to string.

  @param  mrbc_sym	Symbol value.
  @return const char*	String.
  @retval NULL		Invalid sym_id was given.
*/
__GURU__ U8P
symid2name(mrbc_sym sid)
{
    return (sid < 0 || sid >= _sym_idx)
    		? NULL
    		: _sym_list[sid].cstr;
}

#if GURU_USE_STRING
// from c_string.cu
extern "C" __GURU__ mrbc_value mrbc_string_new(const U8P src);
extern "C" __GURU__ void       mrbc_string_append_cstr(mrbc_value *s1, const U8P s2);

//================================================================
/*! (method) inspect
 */
__GURU__ void
c_inspect(mrbc_value v[], U32 argc)
{
    mrbc_value ret = mrbc_string_new((U8P)":");

    mrbc_string_append_cstr(&ret, symid2name(v[0].i));

    SET_RETURN(ret);
}


//================================================================
/*! (method) to_s
 */
__GURU__ void
c_to_s(mrbc_value v[], U32 argc)
{
    v[0] = mrbc_string_new(symid2name(v[0].i));
}
#endif

#if GURU_USE_ARRAY
//================================================================
/*! (method) all_symbols
 */
__GURU__ void
c_all_symbols(mrbc_value v[], U32 argc)
{
    mrbc_value ret = mrbc_array_new(_sym_idx);

    for (U32 i = 0; i < _sym_idx; i++) {
        mrbc_value sym1 = {.tt = GURU_TT_SYMBOL};
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
    mrbc_class *c = mrbc_class_symbol = guru_add_class("Symbol", mrbc_class_object);

#if GURU_USE_ARRAY
    guru_add_proc(c, "all_symbols", c_all_symbols);
#endif
#if GURU_USE_STRING
    guru_add_proc(c, "inspect", 	c_inspect);
    guru_add_proc(c, "to_s", 		c_to_s);
    guru_add_proc(c, "id2name", 	c_to_s);
#endif
    guru_add_proc(c, "to_sym", 		c_nop);
}




