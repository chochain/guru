/*! @file
  @brief
  GURU Symbol class (implemented as a string hasher)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "value.h"
#include "mmu.h"
#include "class.h"
#include "symbol.h"
#include "c_string.h"
#include "c_array.h"
#include "inspect.h"

#define _LOCK	{ MUTEX_LOCK(_mutex_sym); }
#define _FREE	{ MUTEX_FREE(_mutex_sym); }

__GURU__ U32    _mutex_sym = 0;
__GURU__ U32 	_sym_idx = 0;					// point to the last(free) sym_list array.
__GURU__ U8*	_sym[MAX_SYMBOL_COUNT];
__GURU__ U32	_sym_hash[MAX_SYMBOL_COUNT];

//================================================================
/*! Calculate hash value.

  @param  str		Target string.
  @return uint16_t	Hash value.
*/
#define HASH_K 1000003

__GURU__ U32
_loop_hash(const U8 *str)
{
	U32 bsz = STRLENB(str);
	U32 h   = 0;
    for (U32 i=0; i<bsz/4; i++, str+=4) {
        h = h * HASH_K + *(U32*)str;			// a simplistic hashing algo
    }
    for (U32 i=0; i<bsz%4; i++) {
    	h = h * HASH_K + *str++;
    }
    return h;
}
//================================================================
/*! search index table
 */
__GURU__ S32
_loop_search(U32 hash)
{
    for (S32 i=0; i<_sym_idx; i++) {
    	if (_sym_hash[i]==hash) return i;
    }
    return -1;
}
//================================================================
/*! add to index table (assume no entry exists)
 */
__GURU__ U32
_add_index(const U8 *str, U32 hash)
{
    _LOCK;
    U32 idx = _sym_idx++;
    _FREE;

    ASSERT(_sym_idx<MAX_SYMBOL_COUNT);
/*
    // deep copy the string (can shallow work?)
    U32 asz  = STRLENB(str) + 1;	ALIGN(asz);
    U8  *buf = (U8*)guru_alloc(asz);

    MEMCPY(buf, str, asz);
    _sym[idx]      = (U8*)buf;
*/
    _sym[idx]      = (U8*)str;
    _sym_hash[idx] = hash;

	return idx;
}

__GURU__ GS
new_sym(const U8 *str)						// create new symbol
{
	U32 hash = _loop_hash(str);
	S32 sid  = _loop_search(hash);
	if (sid<0) {
		sid  = _add_index(str, hash);
	}
#if CC_DEBUG
	printf("  sym[%2d]%08x: %s==%s\n", sid, hash, str, _sym[sid]);
#endif // CC_DEBUG

	return sid;
}

//================================================================
/*! Calculate hash value

  @param  str		Target string.
  @return GS	Symbol value.
*/
__GURU__ U32 _dyna_hsh[32];				// for dynamic parallel search
__GPU__ void
_dyna_hash0(U32 *hash, U32 sz, const U8 *str)
{
	U32 i  = threadIdx.x;
	U32 *h = _dyna_hsh;
	U32 n  = sz>64 ? 32 : sz>>1;

	h[i] = (str[i]+1) * HASH_K;		// bank conflict?
	for (U32 off=n; off>1; off>>=1) {
		if (i<off) h[i] = h[i]*HASH_K + str[off+i];
	}
	*hash += h[0] + h[1];
}

__GURU__ U32
_dyna_hash(const U8 *str)
{
	U32 bsz  =STRLENB(str);	bsz+=bsz&1;	// include '\0' if odd length
	static U32 hash;
	hash = 0;
	for (U32 i=0; i<bsz; i+=64) {
		_dyna_hash0<<<1,32>>>(&hash, bsz-i, &str[i]);
	}
	cudaDeviceSynchronize();

	return hash;
}

__GPU__ void
_dyna_search0(S32 *idx, const U32 hash)
{
	U32 tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid<_sym_idx && _sym_hash[tid]==hash) {
		*idx = tid;				// capture the index
	}
}

__GURU__ S32 _dyna_idx[32];		// for dynamic parallel search
__GURU__ S32
_dyna_search(U32 hash)
{
	U32 tid  = threadIdx.x;
	S32 *idx = &_dyna_idx[tid];

	U32 tc = 32;				// Pascal:512, Volta:1024 max threads/block
	U32 bc = (_sym_idx>>5)+1;	// P104: 20, P102: 30 quad-issue SMs (i.e. 4 blocks/issue)

	cudaStream_t st;
	cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);	// wrapper overhead ~= 84us
	{
		_dyna_search0<<<bc, tc, 0, st>>>(idx, hash);		// spawn
	}
	cudaDeviceSynchronize();	// sync all child threads in the block
	cudaStreamDestroy(st);

	__syncthreads();			// sync parent threads
	return *idx;				// each parent thread gets one result back
}

#define DYNA_HASH_THRESHOLD     128
#define DYNA_SEARCH_THRESHOLD 	128			// meta-parameter

__GURU__ GS
name2id(const U8 *str)
{
//	U32 hash = _dyna_hash(str);				// loop: 0.28ms vs dyna: 0.95 ms/call
//	return 0;

	U32 hash = _loop_hash(str);				// loop: 0.28ms vs dyna: 0.95 ms/call
	S32 sid  = (_sym_idx>DYNA_SEARCH_THRESHOLD)
		? _dyna_search(hash)				// 100us/c
		: _loop_search(hash);				// 50us/c

#if CC_DEBUG
    printf("  sym[%2d]%08x=>%s\n", sid, _sym_hash[sid], _sym[sid]);
#endif // CC_DEBUG

	return sid;					// different value for each parent thread
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
/*! constructor

  @param  vm	pointer to VM.
  @param  str	String
  @return 	symbol object
*/
__GURU__ GV
guru_sym_new(const U8 *str)
{
    GV v; { v.gt = GT_SYM; v.acl=0; v.i=new_sym(str); }

    return v;
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
        GV sym1; { sym1.gt = GT_SYM; sym1.acl=0; sym1.i=1; }
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
