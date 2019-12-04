/*! @file
  @brief
  GURU Symbol class (implemented as a string hasher)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>

#include "value.h"
#include "mmu.h"
#include "static.h"
#include "symbol.h"
#include "c_string.h"
#include "c_array.h"
#include "inspect.h"

#if !defined(GS_LINER) && !defined(GS_BTREE)
#define GS_BTREE
#endif

struct RTree {
    U8  *cstr;						//!< point to the symbol string.
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
_calc_hash(const U8 *str)
{
    U16 h = 0;
    for (U32 i=0, b=STRLENB(str); i<b; i++) {
        h = h * 37 + *str++;		// a simplistic hashing algo
    }
    return h;
}

//================================================================
/*! search index table
 */
__GURU__ U32
_search_index(const U8 *str)
{
    U16 hash = _calc_hash(str);

#ifdef GS_BTREE
    U32 i=0;
    do {
        if (_sym[i].hash==hash &&
        	STRCMP(str, _sym[i].cstr)==0) {
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
_add_index(const U8 *str)
{
    U32 hash = _calc_hash(str);

    // check overflow.
    assert(_sym_idx < MAX_SYMBOL_COUNT);

    // append table.
    U32 sid = _sym_idx++;				// add to next entry
    _sym[sid].hash = hash;
    _sym[sid].cstr = (U8*)str;

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
guru_sym_new(const U8 *str)
{
    GV v; { v.gt = GT_SYM; v.acl=0; }
    GS sid = _search_index(str);

    if (sid < MAX_SYMBOL_COUNT) {
        v.i = sid;
        return v;					// already exist.
    }

    // create symbol object dynamically.
    U32 asz  = STRLENB(str) + 1;	ALIGN(asz);
    U8  *buf = (U8*)guru_alloc(asz);

    MEMCPY(buf, str, asz);
    v.i = _add_index(buf);

    return v;
}

//================================================================
/*! Convert string to symbol value.

  @param  str		Target string.
  @return GS	Symbol value.
*/
__GURU__ GS
name2id(const U8 *str)
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
__GURU__ U8*
id2name(GS sid)
{
    return (sid < _sym_idx) ? _sym[sid].cstr : NULL;
}

//================================================================
// call by symbol
#if !GURU_USE_ARRAY
__CFUNC__	sym_all(GV v[], U32 vi)	{}
#else
__CFUNC__
sym_all(GV v[], U32 vi)
{
    GV ret = guru_array_new(_sym_idx);

    for (U32 i=0; i < _sym_idx; i++) {
        GV sym1; { sym1.gt = GT_SYM; sym1.acl=0; }
        sym1.i = i;
        guru_array_push(&ret, &sym1);
    }
    RETURN_VAL(ret);
}
#endif // GURU_USE_ARRAY

__CFUNC__
sym_to_s(GV v[], U32 vi)
{
	GV ret = guru_str_new(id2name(v->i));
    RETURN_VAL(ret);
}

__CFUNC__ sym_nop(GV v[], U32 vi) {	/* do nothing */	}

//================================================================
/*! initialize
 */
__GURU__ __const__ Vfunc sym_vtbl[] = {
	{ "id2name", 	gv_to_s		},
	{ "to_sym",     sym_nop		},
	{ "to_s", 		sym_to_s	}, 	// no leading ':'
	{ "inspect", 	gv_to_s		},
	{ "all_symbols", sym_all	}
};

__GURU__ void
guru_init_class_symbol()  // << from symbol.cu
{
    guru_class_symbol = guru_add_class("Symbol", guru_class_object, sym_vtbl, VFSZ(sym_vtbl));
}

__GPU__ void
_id2str(GS sid, U8 *str)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	U8 *s = id2name(sid);
	STRCPY(str, s);
}

#if GURU_DEBUG
__HOST__ void
id2name_host(GS sid, U8 *str)
{
	_id2str<<<1,1>>>(sid, str);
	cudaDeviceSynchronize();
}
#endif // GURU_DEBUG
