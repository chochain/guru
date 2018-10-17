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

extern "C" void *guru_malloc(size_t sz, int mem_type);
extern "C" void dump_alloc_stat(void);

extern "C" __global__ void guru_init_alloc(void *ptr, unsigned int sz);
extern "C" __global__ void guru_init_static(void);

int _alloc_session(guru_ses *ses, size_t req_sz, size_t res_sz)
{
	ses->req = (uint8_t *)guru_malloc(req_sz, 1);	// allocate bytecode storage
	ses->res = (uint8_t *)guru_malloc(res_sz, 1);	// allocate output buffer

	if (!ses->req || !ses->res) return 1;

    _guru_set_console_buf<<<1,1>>>(ses->res);

    return (cudaSuccess==cudaGetLastError()) ? 0 : 1;
}

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

  int err = _alloc_session(ses, sz, MAX_BUFFER_SIZE);

  if (err != 0) {
	  fprintf(stderr, "session buffer allocation error: %d.\n", err);
	  return err;
  }
  else {
	  fread(ses->req, sizeof(char), sz, fp);
  }
  fclose(fp);

  return 0;
}

uint8_t *init_session(guru_ses *ses, const char *rite_fname)
{
	int rst = _input_bytecode(ses, rite_fname);

	if (rst != 0) return NULL;

	void *mem = guru_malloc(BLOCK_MEMORY_SIZE, 0);

    guru_init_alloc<<<1,1>>>(mem, BLOCK_MEMORY_SIZE);
	guru_init_static<<<1,1>>>();
	dump_alloc_stat();

	mrbc_vm *vm = (mrbc_vm *)guru_malloc(sizeof(mrbc_vm), 1);			// allocate bytecode storage
    if (!vm) return NULL;

	guru_parse_bytecode<<<1,1>>>(vm, ses->req);

	uint8_t *vm_rst = (uint8_t *)malloc(sizeof(mrbc_vm));
	if (vm_rst==0) {
    	printf("allocation error: vm_rst");
    	return NULL;
	}
	memcpy(vm, &vm_rst, sizeof(mrbc_vm));

    return vm_rst;
}
    
