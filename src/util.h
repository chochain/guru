/*! @file
  @brief
  GURU Utility functions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_UTIL_H_
#define GURU_SRC_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__CUDACC__)

__device__ unsigned long bin_to_u32(const void *bin);
__device__ unsigned int	 bin_to_u16(const void *bin);
__device__ void			u16_to_bin(unsigned int s, const void *bin);
__device__ void			u32_to_bin(unsigned long l, const void *bin);

__device__ void    		*d_memcpy(void *d, const void *s, size_t n);
__device__ void    		*d_memset(void *d, int c, size_t n);
__device__ int     		d_memcmp(const void *s1, const void *s2, size_t n);

__device__ int  		d_strlen(const char *s, int raw);
__device__ int     		d_strcmp(const char *s1, const char *s2);
__device__ char*		d_strchr(const char *s,  const char c);
__device__ char*		d_strcat(char *d,  const char *s);
__device__ char*     	d_strcut(const char *s, int n);			// take n utf8 chars from the string

__device__ int 			d_hash(const char *s);
__device__ int 			d_atoi(const char *s, size_t base);
__device__ double		d_atof(const char *s);

#define MEMCPY(d,s,n)   memcpy(d,s,n)
#define MEMSET(d,v,n)   memset(d,v,n)
#define MEMCMP(d,s,n)   d_memcmp(d,s,n)

#define STRLEN(s)		d_strlen((char*)(s), 0)
#define STRLENB(s)		d_strlen((char*)(s), 1)
#define STRCPY(d,s)		MEMCPY(d,s,STRLENB(s)+1)
#define STRCMP(d,s)    	MEMCMP(d,s,STRLENB(s))
#define STRCHR(d,c)     d_strchr((char*)d,c)
#define STRCAT(d,s)     d_strcat((char*)d, (char*)s)
#define STRCUT(d,n)		d_strcut((char*)d, (int)n)

#define HASH(s)			d_hash((char*)(s))
#define ATOI(s, base)   d_atoi((char*)(s), base)
#define ATOF(s)			d_atof((char*)(s))

#else

#define MEMCPY(d,s,sz)  memcpy(d, s, sz)
#define MEMCMP(d,s,sz)  memcmp(d, s, sz)
#define MEMSET(d,v,sz)  memset(d, v, sz)

#define STRLEN(s)		strlen(s)
#define STRCPY(d,s)	  	strcpy(d, s)
#define STRCMP(s1,s2)   strcmp(s1, s2)
#define STRCHR(d,c)     strchr(d, c)
#define STRCAT(d,s)     strcat(d, s)
#define STRCUT(s,sz)	substr(s, sz)

#define HASH(s)			calc_hash((s)			// add implementation

#define ATOI(s)			atol(s)
#define ATOF(s)			atof(s)

#endif	// __CUDACC__

#ifdef __cplusplus
}
#endif
#endif
