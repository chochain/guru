/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2018- Greenii
  </pre>
*/
#include "vm_config.h"
#include "guru.h"

char *guru_alloc(size_t sz)
{
    char *p;

    cudaMallocManaged(&p, sz);

    return p;
}

__GURU__ void guru_memcpy(uint8_t *d, const uint8_t *s, size_t sz)
{
    for (int i=0; i<sz; i++, *d++ = *s++);
}

__GURU__ void guru_memset(uint8_t *d, const uint8_t v,  size_t sz)
{
    for (int i=0; i<sz; i++, *d++ = v);
}

__GURU__ size_t guru_memcmp(const uint8_t *d, const uint8_t *s, size_t sz)
{
    int i;
    
    for (i=0; i<sz && *d++==*s++; i++);
    
    return (size_t)(i>=sz);
}

__GURU__ long guru_atol(const char *s)
{
    return 0L;
}

__GURU__ size_t guru_strlen(const char *str)
{
    int i=0;
    while (str[++i]!='\0');
    return (size_t)i;
}

__GURU__ void  guru_strcpy(const char *s1, const char *s2)
{
    guru_memcpy((uint8_t *)s1, (uint8_t *)s2, guru_strlen(s1));
}

__GURU__ size_t  guru_strcmy(const char *s1, const char *s2)
{
    return guru_memcmp((uint8_t *)s1, (uint8_t *)s2, guru_strlen(s1));
}

__GURU__ char   *guru_strchr(const char *s, const char c)
{
    while (*s!='\0' && *s!=c) s++;
    
    return (char *)((*s==c) ? &s : NULL);
}

__GURU__ char   *guru_strcat(char *d, const char *s)
{
    return d;
}




    
