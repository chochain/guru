/*! @file
  @brief
  GURU Symbol class

  <pre>
  Copyright (C) 2019- GreenII

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

#if !defined(GS_LINER) && !defined(GS_BTREE)
#define GS_BTREE
#endif

struct RTree {
    U8P cstr;						//!< point to the symbol string.
#ifdef GS_BTREE						// array-based btree
    GS 	left;
    GS 	right;
#endif
    U16	hash;						//!< hash value, returned by _calc_hash.
};

__GURU__ U32 			_sym_idx;	// point to the last(free) sym_list array.
__GURU__ struct RTree 	_sym[MAX_SYMBOL_COUNT];

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
        h = h * 37 + *str;		// a simplistic hashing algo
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

#ifdef GS_BTREE
    U32 i=0;
    do {
        if (_sym[i].hash==hash &&
        	guru_strcmp(str, _sym[i].cstr)==0) {
            return i;
        }
        i = (hash < _sym[i].hash) ? _sym[i].left : _sym[i].right;
    } while (i!=0);
#else
    for (U32 i=0; i < _sym_idx; i++) {
        if (_sym[i].hash==hash && strcmp(str, _sym[i].cstr)==0) {
            return i;
        }
    }
#endif
    return MAX_SYMBOL_COUNT;		// not found
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
    U32 sid = _sym_idx++;				// add to next entry
    _sym[sid].hash = hash;
    _sym[sid].cstr = str;

#ifdef GS_BTREE
    for (U32 i=0; ;) {					// array-based btree walker
        if (hash < _sym[i].hash) {
            // left side
            if (_sym[i].left==0) {		// left is empty?
                _sym[i].left = sid;
                break;
            }
            i = _sym[i].left;
        }
        else {
            // right side
            if (_sym[i].right==0) {		// right is empty?
                _sym[i].right = sid;
                break;
            }
            i = _sym[i].right;
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
    GV v; { v.gt = GT_SYM; }
    GS sid = _search_index(str);

    if (sid < MAX_SYMBOL_COUNT) {
        v.i = sid;
        return v;					// already exist.
    }

    // create symbol object dynamically.
    U32 size = guru_strlen(str) + 1;
    U8P buf  = (U8P)guru_alloc(size);

    MEMCPY(buf, str, size);
    v.i = _add_index(buf);

    return v;
}

//================================================================
/*! Convert string to symbol value.

  @param  str		Target string.
  @return GS	Symbol value.
*/
__GURU__ GS
name2id(const U8P str)
{
    GS sid = _search_index(str);
    if (sid < MAX_SYMBOL_COUNT) return sid;

    return _add_index(str);
}

//================================================================
/*! Convert symbol value to string.

  @param  GS	Symbol value.
  @return const char*	String.
  @retval NULL		Invalid sym_id was given.
*/
__GURU__ U8P
id2name(GS sid)
{
    return (sid < _sym_idx) ? _sym[sid].cstr : NULL;
}

#if GURU_USE_STRING
// from c_string.cu
extern "C" __GURU__ GV 		guru_str_new(const U8 *src);
extern "C" __GURU__ void    guru_str_append_cstr(const GV *s1, const U8 *s2);

//================================================================
/*! (method) inspect
 */
__GURU__ void
c_sym_inspect(GV v[], U32 argc)
{
    GV ret = guru_str_new(":");

    guru_str_append_cstr(&ret, id2name(v[0].i));

    RETURN_VAL(ret);
}


//================================================================
/*! (method) to_s
 */
__GURU__ void
c_sym_to_s(GV v[], U32 argc)
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
        GV sym1; { sym1.gt = GT_SYM; }
        sym1.i = i;
        guru_array_push(&ret, &sym1);
    }
    RETURN_VAL(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void guru_init_class_symbol()  // << from symbol.cu
{
    guru_class *c = guru_class_symbol = NEW_CLASS("Symbol", guru_class_object);

#if GURU_USE_ARRAY
    NEW_PROC("all_symbols", c_all_symbols);
#endif
#if GURU_USE_STRING
    NEW_PROC("inspect", 	c_sym_inspect);
    NEW_PROC("to_s", 		c_sym_to_s);
    NEW_PROC("id2name", 	c_sym_to_s);
#endif
}




