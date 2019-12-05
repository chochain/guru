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
#include "class.h"
#include "symbol.h"
#include "c_string.h"
#include "c_array.h"
#include "inspect.h"

__GURU__ U32 	_sym_idx = 0;					// point to the last(free) sym_list array.
__GURU__ U8*	_sym[MAX_SYMBOL_COUNT];
__GURU__ U32	_sym_hash[MAX_SYMBOL_COUNT];

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
__GPU__ void
__scan(S32 *idx, const U32 hash)
{
	S32 i = threadIdx.x;

	if (i<MAX_SYMBOL_COUNT && _sym_hash[i]==hash) *idx = i;

	__syncthreads();
}

__GURU__ S32
_search_index(const U8 *str)
{
	U32 hash = _calc_hash(str);

	static S32 idx  = -1;
    __scan<<<1, MAX_SYMBOL_COUNT>>>(&idx, hash);

    return idx;
}

//================================================================
/*! add to index table (assume no entry exists)
 */
__GURU__ U32
_add_index(const U8 *str)
{
    // append table.
    U32 idx = _sym_idx++;
    assert(idx<MAX_SYMBOL_COUNT);

    _sym[idx]      = (U8*)str;
    _sym_hash[idx] = _calc_hash(str);

    return idx;
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

    S32 sid  = _search_index(str);
    if (sid >=0) {
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
    S32 sid = _search_index(str);
    if (sid>=0) return sid;

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
    return (sid < _sym_idx) ? _sym[sid] : NULL;
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
    guru_rom_set_class(GT_SYM, "Symbol", GT_OBJ, sym_vtbl, VFSZ(sym_vtbl));
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
