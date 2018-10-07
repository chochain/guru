/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2018- Greenii
  </pre>
*/
#include "guru.h"

int guru_init(guru_ses *ses, size_t req_sz, size_t res_sz)
{
    cudaMallocManaged(&(ses->req), req_sz);
    cudaMallocManaged(&(ses->res), res_sz);

    return (cudaSuccess==cudaGetLastError()) ? 0 : 1;
}

    
