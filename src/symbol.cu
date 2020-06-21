/*! @file
  @brief
  GURU Symbol class (implemented as a string hasher)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "guru.h"
#include "util.h"
#include "mmu.h"
#include "static.h"
#include "symbol.h"

#define DYNA_SEARCH_THRESHOLD 	128				// meta-parameter

#define _LOCK	{ MUTEX_LOCK(_mutex_sym); }
#define _FREE	{ MUTEX_FREE(_mutex_sym); }

__GURU__ U32    _mutex_sym = 0;
__GURU__ S32 	_sym_idx = 0;					// point to the last(free) sym_list array.

__GURU__ S32	_sym[MAX_SYMBOL_COUNT];			// list of arrays
__GURU__ U32	_sym_hash[MAX_SYMBOL_COUNT];

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
    U32 asz  = ALIGN(STRLENB(str) + 1);
    U8  *buf = (U8*)guru_alloc(asz);

    MEMCPY(buf, str, asz);
    _sym[idx]      = (U8*)buf;
*/
    _sym[idx]      = MEMOFF(str);			// shallow copy, need to keep source in the memory
    _sym_hash[idx] = hash;

	return idx;
}

//================================================================
/*! search index table
 */
__GURU__ S32
_loop_search(U8 *str)
{
	U32 hash = HASH(str);
    for (int i=0; i<_sym_idx; i++) {
    	if (_sym_hash[i]==hash) return i;
    }
    return -1;
}

#if GURU_ENABLE_CDP
__GPU__ void
_dyna_search(S32 *idx, const U8 *str)
{
	U32 x = threadIdx.x + blockIdx.x*blockDim.x;

	U32 hash = HASH(str);
	if (x<_sym_idx && _sym_hash[x]==hash) {
		*idx = x;				// capture the index
	}
}
#endif // GURU_ENABLE_CDP

__GURU__ S32 _warp_i[32];
__GURU__ S32
_search(U8 *str)
{
#if CUDA_ENABLE_CDP
	if (_sym_idx<DYNA_SEARCH_THRESHOLD) return _loop_search(hash);

	S32 *idx = &_warp_i[threadIdx.x];	*idx = -1;

	U32 tc = 32;				// Pascal:512, Volta:1024 max threads/block
	U32 bc = (_sym_idx>>5)+1;	// P104: 20, P102: 30 quad-issue SMs (i.e. 4 blocks/issue)

	cudaStream_t st;
	cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);	// wrapper overhead ~= 84us
	{
		_dyna_search<<<bc, tc, 0, st>>>(idx, hash);			// spawn
		GPU_SYNC();				// sync all child threads in the block
	}
	cudaStreamDestroy(st);

	return *idx;				// each parent thread gets one result back
#else
	S32 sid = guru_rom_get_sym((char*)str);

	return (sid>0) ? sid : _loop_search(str);
#endif // CUDA_ENABLE_CDP
}

__GURU__ GS
name2id(const U8 *str)
{
	S32 sid = guru_rom_get_sym((char*)str);

#if CC_DEBUG
	guru_sym *sym = _SYM(sid);
	U8       *raw = _RAW(sid);
    printf("%2d> sym[%2x]%08x=>%s\n", threadIdx.x, sid, sym, raw);
#endif // CC_DEBUG

	return sid;						// different value for each parent thread
}

//================================================================
/*! Convert symbol value to string.

  @param  GS	Symbol value.
  @return const char*	String.
  @retval NULL		Invalid sym_id was given.
*/
__GURU__ GP
id2name(GS sid)
{
	guru_sym *sym = _SYM(sid);

    return guru_device_rom.str + sym->raw;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  str	String
  @return 	symbol object
*/
__GURU__ void
guru_sym_transcode(GR *r)
{
	GS sid = guru_rom_add_sym((char*)U8PADD(r, r->off));
	r->i = (GI)sid;
}
