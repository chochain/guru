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
    U16								hash;		//!< hash value, returned by calc_hash().
    U8P 							cstr;		//!< point to the symbol string.
#ifdef GURU_SYMBOL_SEARCH_BTREE
    GURU_SYMBOL_TABLE_INDEX_TYPE 	left;
    GURU_SYMBOL_TABLE_INDEX_TYPE 	right;
#endif
};

__GURU__ U32 _sym_idx;							// point to the last(free) sym_list array.
__GURU__ struct SYM_LIST _sym_list[MAX_SYMBOL_COUNT];

//================================================================
/*! Calculate hash value.

  @param  str		Target string.
  @return uint16_t	Hash value.
*/
__GURU__ U16
_calc_hash(U8P str)
{
    U16 h = 0;

    while (*str != '\0') {
        h = h * 37 + *str;
        str++;
    }
    return h;
}

//================================================================
/*! search index table
 */
__GURU__ U32
_search_index(const U8P str)
{
    U16 hash = _calc_hash(str);

#ifdef GURU_SYMBOL_SEARCH_LINER
    for (U32 i=0; i < _sym_idx; i++) {
        if (_sym_list[i].hash==hash && strcmp(str, _sym_list[i].cstr)==0) {
            return i;
        }
    }
    return MAX_SYMBOL_COUNT;
#endif

#ifdef GURU_SYMBOL_SEARCH_BTREE
    U32 i=0;
    do {
        if (_sym_list[i].hash==hash &&
        	guru_strcmp(str, _sym_list[i].cstr)==0) {
            return i;
        }
        i = (hash < _sym_list[i].hash) ? _sym_list[i].left : _sym_list[i].right;
    } while (i!=0);

    return MAX_SYMBOL_COUNT;		// not found
#endif
}

//================================================================
/*! add to index table (1-based index, i.e. list[0] is not used)
 */
__GURU__ S32
_add_index(const U8P str)
{
    U32 hash = _calc_hash(str);

    // check overflow.
    assert(_sym_idx < MAX_SYMBOL_COUNT);

    // append table.
    U32 sid = _sym_idx++;		// add to next entry
    _sym_list[sid].hash = hash;
    _sym_list[sid].cstr = str;

#ifdef GURU_SYMBOL_SEARCH_BTREE
    for (U32 i=0; ;) {
        if (hash < _sym_list[i].hash) {
            // left side
            if (_sym_list[i].left==0) {		// left is empty?
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
__GURU__ GV
guru_sym_new(const U8P str)
{
    GV v = {.gt = GT_SYM};
    guru_sym   sid = _search_index(str);

    if (sid < MAX_SYMBOL_COUNT) {
        v.i = sid;
        return v;					// already exist.
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
  @return guru_sym	Symbol value.
*/
__GURU__ guru_sym
name2id(const U8P str)
{
    guru_sym sid = _search_index(str);
    if (sid < MAX_SYMBOL_COUNT) return sid;

    return _add_index(str);
}

//================================================================
/*! Convert symbol value to string.

  @param  guru_sym	Symbol value.
  @return const char*	String.
  @retval NULL		Invalid sym_id was given.
*/
__GURU__ U8P
id2name(guru_sym sid)
{
    return (sid >= _sym_idx)
    		? NULL
    		: _sym_list[sid].cstr;
}

#if GURU_USE_STRING
// from c_string.cu
extern "C" __GURU__ GV 		guru_str_new(const U8 *src);
extern "C" __GURU__ void    guru_str_append_cstr(const GV *s1, const U8 *s2);

//================================================================
/*! (method) inspect
 */
__GURU__ void
c_inspect(GV v[], U32 argc)
{
    GV ret = guru_str_new(":");

    guru_str_append_cstr(&ret, id2name(v[0].i));

    SET_RETURN(ret);
}


//================================================================
/*! (method) to_s
 */
__GURU__ void
c_to_s(GV v[], U32 argc)
{
    v[0] = guru_str_new(id2name(v[0].i));
}
#endif

#if GURU_USE_ARRAY
//================================================================
/*! (method) all_symbols
 */
__GURU__ void
c_all_symbols(GV v[], U32 argc)
{
    GV ret = guru_array_new(_sym_idx);

    for (U32 i=0; i < _sym_idx; i++) {
        GV sym1 = {.gt = GT_SYM};
        sym1.i = i;
        guru_array_push(&ret, &sym1);
    }
    SET_RETURN(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void guru_init_class_symbol()  // << from symbol.cu
{
    guru_class *c = guru_class_symbol = guru_add_class("Symbol", guru_class_object);

#if GURU_USE_ARRAY
    guru_add_proc(c, "all_symbols", c_all_symbols);
#endif
#if GURU_USE_STRING
    guru_add_proc(c, "inspect", 	c_inspect);
    guru_add_proc(c, "to_s", 		c_to_s);
    guru_add_proc(c, "id2name", 	c_to_s);
#endif
}




