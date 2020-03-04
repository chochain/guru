/*! @file
  @brief
  GURU Utilities functions
    Memory cpy/set/cmp
    String hasher

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <stdint.h>
#include "util.h"

typedef int         WORD;
#define WSIZE   	(sizeof(WORD))
#define	WMASK		(WSIZE-1)

#define DYNA_HASH_THRESHOLD     1
#define HASH_K 1000003

__device__ int _warp_h[32];			// each thread takes a slot

__device__ __inline__ void
_next_utf8(char **sp)
{
	char c = **sp;
	int  b = 0;
	if      (c>0 && c<=127) 		b=1;
	else if ((c & 0xE0) == 0xC0) 	b=2;
	else if ((c & 0xF0) == 0xE0) 	b=3;
	else if ((c & 0xF8) == 0xF0) 	b=4;
	else *sp=NULL;					// invalid utf8

	*sp+=b;
}

__device__ int
_loop_hash(const char *str, int bsz)
{
	// a simple polynomial hashing algorithm
	int h = 0;
    for (int i=0; i<bsz; i++) {
        h = h * HASH_K + str[i];
    }
    return h;
}

//================================================================
/*! Calculate hash value

  @param  str	Target string.
  @return int	Symbol value.
*/
__global__ void
_dyna_hash(int *hash, const char *str, int sz)
{
	int x = threadIdx.x;									// row-major
	int m = __ballot_sync(0xffffffff, x<sz);				// ballot_mask
	int h = x<sz ? str[x] : 0;								// move to register

	for (int n=16; x<sz && n>0; n>>=1) {
		h += HASH_K*__shfl_down_sync(m, h, n);				// shuffle down
	}
	if (x==0) *hash += h;
}

__global__ void
_dyna_hash2d(int *hash, const char *str, int *mask, int bsz)
{
	int  x = threadIdx.x + threadIdx.y * blockDim.x;		// row-major
	bool c = x<bsz;
	int  m = __ballot_sync(0xffffffff, c);
	int  hx= c ? str[x] : 0;

	for (int n=blockDim.x>>1; n>0; n>>=1) {					// rollup rows
		hx += HASH_K*__shfl_down_sync(m, hx, n);			// shuffle down
	}
	if (threadIdx.x!=0) return;								// only row leaders

	c = threadIdx.y<blockDim.y;
	m = __ballot_sync(0xffffffff, c);
	int hy = c ? hx : 0;
	for (int n=blockDim.y>>1; n>0; n>>=1) {					// rollup columns
		hy += HASH_K*__shfl_down_sync(m, hy, n);			// shuffle down
	}
	if (threadIdx.y==0) *hash = hy;
}

__device__ int
_hash(const char *str, int bsz)
{
	if (bsz < DYNA_HASH_THRESHOLD) return _loop_hash(str, bsz);

	int x  = threadIdx.x;
	int *h = &_warp_h[x];	*h=0;							// each calling thread takes a slot
	cudaStream_t st;
	cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);	// wrapper overhead ~= 84us
	for (int i=0; i<bsz; i+=32) {
		_dyna_hash<<<1,32,0,st>>>(h, &str[i], bsz-i);
		cudaDeviceSynchronize();							// sync all children threads
	}
	cudaStreamDestroy(st);

	return *h;
}

__device__ void*
d_memcpy(void *d, const void *s, size_t n)
{
	if (n==0 || d==s) return d;

	char *ds = (char*)d, *ss = (char*)s;
	size_t t = (uintptr_t)ss;						// take low bits

	if ((unsigned long)ds < (unsigned long)ss) {	// copy forward
		if ((t | (uintptr_t)ds) & WMASK) {
			int i = ((t ^ (uintptr_t)ds) & WMASK || n < WSIZE)			// align operands
				? n
				: WSIZE - (t & WMASK);
			n -= i;
			for (; i; i--) *ds++ = *ss++;							// leading bytes
		}
		for (int i=n/WSIZE; i; i--) { *(WORD*)ds=*(WORD*)ss; ds+=WSIZE; ss+=WSIZE; }
		for (int i=n&WMASK; i; i--) *ds++ = *ss++;					// trailing bytes
	}
	else {											// copy backward
		ss += n;
		ds += n;
		if ((t | (uintptr_t)ds) & WMASK) {
			int i = ((t ^ (uintptr_t)ds) & WMASK || n <= WSIZE)
				? n
				: t & WMASK;
			n -= i;
			for (; i; i--) *--ds = *--ss;							// leading bytes
		}
		for (int i=n/WSIZE; i; i--) { ss-=WSIZE; ds-=WSIZE; *(WORD*)ds=*(WORD*)ss; }
		for (int i=n&WMASK; i; i--) *--ds = *--ss;
	}
	return d;
}

__device__ void*
d_memset(void *d, int c, size_t n)
{
    char *s = (char*)d;

    /* Fill head and tail with minimal branching. Each
     * conditional ensures that all the subsequently used
     * offsets are well-defined and in the dest region. */

    if (!n) return d;
    s[0] = s[n-1] = c;
    if (n <= 2) return d;
    s[1] = s[n-2] = c;
    s[2] = s[n-3] = c;
    if (n <= 6) return d;
    s[3] = s[n-4] = c;
    if (n <= 8) return d;

    /* Advance pointer to align it at a 4-byte boundary,
     * and truncate n to a multiple of 4. The previous code
     * already took care of any head/tail that get cut off
     * by the alignment. */

    size_t k = -(uintptr_t)s & 3;
    s += k;
    n -= k;
    n &= -4;			// change of sign???
    n /= 4;

    uint32_t *ws = (uint32_t *)s;
    uint32_t  wc = c & 0xFF;
    wc |= ((wc << 8) | (wc << 16) | (wc << 24));

    /* Pure C fallback with no aliasing violations. */
    for (; n; n--) *ws++ = wc;

    return d;
}

__device__ int
d_memcmp(const void *s1, const void *s2, size_t n)
{
	char *p1=(char*)s1, *p2=(char*)s2;
	for (; n; n--, p1++, p2++) {
		if (*p1 != *p2) return *p1 - *p2;
	}
	return 0;
}

__device__ int
d_strlen(const char *str, int raw)
{
	int  n  = 0;
	char *s = (char*)str;
	for (int i=0; s && *s!='\0'; i++, n++) {
		_next_utf8(&s);
	}
	return (s && raw) ? s - str : n;
}

__device__ void
d_strcpy(char *d, const char *s)
{
    d_memcpy(d, s, STRLENB(s)+1);
}

__device__ int
d_strcmp(const char *s1, const char *s2)
{
    return d_memcmp(s1, s2, STRLENB(s1));
}

__device__ char*
d_strchr(const char *s, const char c)
{
	char *p = (char*)s;
    for (; p && *p!='\0'; p++) {
    	if (*p==c) return p;
    }
    return NULL;
}

__device__ char*
d_strcat(char *d, const char *s)
{
	d_memcpy(d+STRLENB(d), s, STRLENB(s)+1);
    return d;
}

__device__ char*
d_strcut(const char *s, int n)
{
	char *p = (char*)s;
	for (int i=0; n && i<n && p && *p!='\0'; i++) {
		_next_utf8(&p);
	}
	return p;
}

__device__ int
d_hash(const char *s)
{
	return _hash(s, STRLENB(s));
}

//================================================================
/*!@brief

  convert ASCII string to integer Guru version

  @param  s	source string.
  @param  base	n base.
  @return	result.
*/
__device__ int
d_atoi(const char *s, size_t base)
{
    int ret  = 0;
    int sign = 0;

REDO:
    switch(*s) {
    case '-': sign = 1;		// fall through.
    case '+': s++;	        break;
    case ' ': s++;          goto REDO;
    }

    char ch;
    int  n;
    while ((ch = *s++) != '\0') {
        if      ('a' <= ch) 			 n = ch - 'a' + 10;
        else if ('A' <= ch) 			 n = ch - 'A' + 10;
        else if ('0' <= ch && ch <= '9') n = ch - '0';
        else break;

        if (n >= base) break;

        ret = ret * base + n;
    }
    return (sign) ? -ret : ret;
}

__device__ double
d_atof(const char *s)
{
    int sign = 1, esign = 1, state=0;
    int r = 0, e = 0;
    long v = 0L, f = 0L;

    while ((*s<'0' || *s>'9') && *s!='+' && *s!='-') s++;

    if (*s=='+' || *s=='-') sign = *s++=='-' ? -1 : 1;

    while (*s!='\0' && *s!='\n' && *s!=' ' && *s!='\t') {
    	if      (state==0 && *s>='0' && *s<='9') {	// integer
    		v = (*s - '0') + v * 10;
    	}
    	else if (state==1 && *s>='0' && *s<='9') {	// decimal
    			f = (*s - '0') + f * 10;
    			r--;
        }
    	else if (state==2) {						// exponential
            if (*s=='-') {
                esign = -1;
                s++;
            }
            if (*s>='0' && *s<='9') e = (*s - '0') + e * 10;
        }
        state = (*s=='e' || *s=='E') ? 2 : ((*s=='.') ? 1 : state);
        s++;
    }
    return sign
    		* (v + (f==0 ? 0.0 : f * exp10((double)r)))
    		* (e==0 ? 1.0 : exp10((double)esign * e));
}
