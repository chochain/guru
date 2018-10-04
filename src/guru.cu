/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2018- Greenii
  </pre>
*/
#include "vm_config.h"
#include "guru.hu"

__GURU__ void guru_memcpy(uint8_t *d, const uint8_t *s, size_t sz)
{
    for (int i=0; i<sz; i++, *d++ = *s++);
}

__GURU__ void guru_memset(uint8_t *d, const uint8_t v,  size_t sz)
{
    for (int i=0; i<sz; i++, *d++ = v);
}

__GURU__ int guru_memcmp(const uint8_t *d, const uint8_t *s, size_t sz)
{
    int i;
    
    for (i=0; i<sz && *d++==*s++; i++);
    
    return i>=sz;
}

__GURU__ long guru_atol(const char *s)
{
    return 0L;
}
    
