/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2018- Greenii
  </pre>
*/
#include "guru.h"

__GURU__
char *guru_output_buffer;		// global output buffer for now, per session later

__global__
void guru_console_set_buf(char *buf)
{
	if (threadIdx.x!=0 || blockIdx.x !=0) return;

	guru_output_buffer = buf;
}

int guru_init(guru_ses *ses, size_t req_sz, size_t res_sz)
{
    cudaMallocManaged(&(ses->req), req_sz);
    cudaMallocManaged(&(ses->res), res_sz);

    guru_console_set_buf<<<1,1>>>(ses->res);

    return (cudaSuccess==cudaGetLastError()) ? 0 : 1;
}

    
