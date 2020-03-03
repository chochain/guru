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

__device__ __inline__ void
d_memcpy(void *d, const void *s, size_t bsz)
{
	char *ds=(char*)d, *ss=(char*)s;
    for (int i=0; s && d && i<bsz; i++) *ds++ = *ss++;
}

__device__ __inline__ void
d_memset(void *d, int v, size_t bsz)
{
	char *ds = (char*)d;
	for (int i=0; d && i<bsz; i++) *ds++ = v;
}

__device__ int
d_memcmp(const void *d, const void *s, size_t bsz)
{
	char *ds=(char*)d, *ss=(char*)s;
	for (int i=0, x=0; i<bsz; i++, ds++, ss++) {
		if ((x=(*ds++ - *ss++))!=0) return x;
	}
	return 0;
}

__device__ int
d_strlen(const char *str, int raw)
{
	int n  = 0;
	char  *s = (char*)str;
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
d_strchr(char *s, const char c)
{
    while (s && *s!='\0' && *s!=c) s++;

    return (char*)((*s==c) ? &s : NULL);
}

__device__ __inline__ char*
d_strcat(char *d, const char *s)
{
	d_memcpy(d+STRLENB(d), s, STRLENB(s)+1);
    return d;
}

__device__ char *
d_strcut(const char *str, int n)
{
	char *s = (char*)str;
	for (int i=0, c=0; n>0 && s && *s!='\0'; i++) {
		_next_utf8(&s);
		if (++c >= n) break;
	}
	return s;
}

__device__ int
d_calc_hash(const char *str)
{
	int bsz = STRLENB(str);
	return _hash(str, bsz);
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

