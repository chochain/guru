/*! @file
  @brief
  GURU Utilities functions
    String hasher

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "util.h"

__GURU__ void
guru_memcpy(U8 *d, const U8 *s, U32 bsz)
{
    for (U32 i=0; s && d && i<bsz; i++, *d++ = *s++);
}

__GURU__ void
guru_memset(U8 *d, U8 v,  U32 bsz)
{
    for (U32 i=0; d && i<bsz; i++, *d++ = v);
}

__GURU__ int
guru_memcmp(const U8 *d, const U8 *s, U32 bsz)
{
	U32 i;
    for (i=0; i<bsz && *d==*s; i++, d++, s++);

    return i<bsz ? (*d - *s) : 0;
}

__GURU__ __INLINE__ void
_next_utf8(U8 **sp)
{
	U8  c = **sp;
	U32 b = 0;
	if      (c>0 && c<=127) 		b=1;
	else if ((c & 0xE0) == 0xC0) 	b=2;
	else if ((c & 0xF0) == 0xE0) 	b=3;
	else if ((c & 0xF8) == 0xF0) 	b=4;
	else *sp=NULL;					// invalid utf8

	*sp+=b;
}

__GURU__ U32
guru_strlen(const U8 *str, U32 use_byte)
{
	U32 n  = 0;
	U8  *s = (U8*)str;
	for (U32 i=0; s && *s!='\0'; i++, n++) {
		_next_utf8(&s);
	}
	return (s && use_byte) ? s - str : n;
}

__GURU__ U8 *
guru_strcut(const U8 *str, U32 n)
{
	U8 *s = (U8*)str;
	for (U32 i=0, c=0; n>0 && s && *s!='\0'; i++) {
		_next_utf8(&s);
		if (++c >= n) break;
	}
	return s;
}

__GURU__ void
guru_strcpy(U8 *d, const U8 *s)
{
    guru_memcpy(d, s, STRLENB(s)+1);
}

__GURU__ S32
guru_strcmp(const U8 *s1, const U8 *s2)
{
    return guru_memcmp(s1, s2, STRLENB(s1));
}

__GURU__ U8*
guru_strchr(U8 *s, const U8 c)
{
    while (s && *s!='\0' && *s!=c) s++;

    return (U8*)((*s==c) ? &s : NULL);
}

__GURU__ U8*
guru_strcat(U8 *d, const U8 *s)
{
	guru_memcpy(d+STRLENB(d), s, STRLENB(s)+1);
    return d;
}

//================================================================
/*! Calculate hash value.

  @param  str		Target string.
  @return uint16_t	Hash value.
*/
#define DYNA_HASH_THRESHOLD     1

#define _LOCK	{ MUTEX_LOCK(_mutex_util); }
#define _FREE	{ MUTEX_FREE(_mutex_util); }

__GURU__ U32    _mutex_util = 0;
__GURU__ S32 	_util_idx = 0;			// point to the last(free) sym_list array.

#define HASH_K 1000003

__GURU__ U32
_loop_hash(const U8 *str, U32 bsz)
{
	// a simple polynomial hashing algorithm
	U32 h = 0;
    for (U32 i=0; i<bsz; i++) {
        h = h * HASH_K + str[i];
    }
    return h;
}

//================================================================
/*! Calculate hash value

  @param  str		Target string.
  @return GS	Symbol value.
*/
__GPU__ void
_dyna_hash(U32 *hash, const U8 *str, U32 sz)
{
	U32 x = threadIdx.x;									// row-major
	U32 m = __ballot_sync(0xffffffff, x<sz);				// ballot_mask
	U32 h = x<sz ? str[x] : 0;								// move to register

	for (U32 n=16; x<sz && n>0; n>>=1) {
		h += HASH_K*__shfl_down_sync(m, h, n);				// shuffle down
	}
	if (x==0) *hash += h;
}

__GPU__ void
_dyna_hash2d(U32 *hash, const U8 *str, U32 *mask, U32 bsz)
{
	U32  x = threadIdx.x + threadIdx.y * blockDim.x;		// row-major
	bool c = x<bsz;
	U32  m = __ballot_sync(0xffffffff, c);
	U32  hx= c ? str[x] : 0;

	for (U32 n=blockDim.x>>1; n>0; n>>=1) {					// rollup rows
		hx += HASH_K*__shfl_down_sync(m, hx, n);			// shuffle down
	}
	if (threadIdx.x!=0) return;								// only row leaders

	c = threadIdx.y<blockDim.y;
	m = __ballot_sync(0xffffffff, c);
	U32 hy = c ? hx : 0;
	for (U32 n=blockDim.y>>1; n>0; n>>=1) {					// rollup columns
		hy += HASH_K*__shfl_down_sync(m, hy, n);			// shuffle down
	}
	if (threadIdx.y==0) *hash = hy;
}

__GURU__ U32 _warp_h[32];
__GURU__ U32
_hash(const U8 *str, U32 bsz)
{
	if (bsz < DYNA_HASH_THRESHOLD) return _loop_hash(str, bsz);

	U32 x  = threadIdx.x;
	U32 *h = &_warp_h[x];	*h=0;							// each calling thread takes a slot
	cudaStream_t st;
	cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);	// wrapper overhead ~= 84us
	for (U32 i=0; i<bsz; i+=32) {
		_dyna_hash<<<1,32,0,st>>>(h, &str[i], bsz-i);
		SYNC();												// sync all children threads
	}
	cudaStreamDestroy(st);

	return *h;
}

__GURU__ U32
guru_calc_hash(const U8 *str)
{
	U32 bsz = STRLENB(str);
	return _hash(str, bsz);
}
