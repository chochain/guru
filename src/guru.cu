/*! @file
  @brief
  Guru value definitions non-optimized

  <pre>
  Copyright (C) 2018- Greenii
  </pre>
*/
#include <stdio.h>
#include "guru.h"
#include "load.h"

__GURU__
uint8_t *guru_output_buffer;		// global output buffer for now, per session later

__global__
void _guru_set_console_buf(uint8_t *buf)
{
	if (threadIdx.x!=0 || blockIdx.x !=0) return;

	guru_output_buffer = buf;
}

__host__
int _guru_alloc(guru_ses *ses, size_t req_sz, size_t res_sz)
{
    cudaMallocManaged(&(ses->req), req_sz);			// allocate bytecode storage
    cudaMallocManaged(&(ses->res), res_sz);			// allocate output buffer

    _guru_set_console_buf<<<1,1>>>(ses->res);

    return (cudaSuccess==cudaGetLastError()) ? 0 : 1;
}

__host__
int _input_bytecode(guru_ses *ses, const char *rite_fname)
{
  FILE *fp = fopen(rite_fname, "rb");

  if (fp==NULL) {
    fprintf(stderr, "File not found\n");
    return -1;
  }

  // get filesize
  fseek(fp, 0, SEEK_END);
  size_t sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  int err = _guru_alloc(ses, sz, MAX_BUFFER_SIZE);

  if (err != 0) {
	  fprintf(stderr, "CUDA memory allocate error: %d.\n", err);
	  return err;
  }
  else {
	  fread(ses->req, sizeof(char), sz, fp);
  }
  fclose(fp);

  return 0;
}

int init_session(guru_ses *ses, const char *rite_fname)
{
	int rst = _input_bytecode(ses, rite_fname);

	if (rst != 0) return rst;

	mrbc_vm vm;

//	mrbc_parse_bytecode<<<1,1>>>(&vm, ses->req);

    return 0;
}
    
