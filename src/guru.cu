/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2018- Greenii
  </pre>
*/
#include "guru.h"

char *guru_alloc(size_t sz)
{
    char *p;

    cudaMallocManaged(&p, sz);

    return p;
}




    
