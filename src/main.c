/*
 * Sample Main Program
 */

#include <stdio.h>
#include <stdlib.h>
#include "mrubyc.h"
#include "c_ext.h"

#define MEMORY_SIZE (1024*32)
static uint8_t memory_pool[MEMORY_SIZE];

int guru(uint8_t *mrbbuf)
{
  struct VM *vm;

  mrbc_init_alloc(memory_pool, MEMORY_SIZE);
  init_static();
  mrbc_init_class_extension(0);

  vm = mrbc_vm_open(NULL);
  if (vm==0) {
    fprintf(stderr, "Error: Can't open VM.\n");
    return 1;
  }

  if (mrbc_upload_bytecode(vm, mrbbuf) != 0) {
    fprintf(stderr, "Error: Illegal bytecode.\n");
    return 1;
  }

  do_cuda();

  mrbc_vm_begin(vm);
  mrbc_vm_run(vm);
  mrbc_vm_end(vm);
  mrbc_vm_close(vm);

  return 0;
}

int main(int argc, char *argv[])
{
  if (argc!=2) {
    printf("Usage: %s <xxxx.mrb>\n", argv[0]);
    return 1;
  }

  uint8_t *mrbbuf = load_mrb_file(argv[1]);  // allocate from host memory
  if(mrbbuf==0) {
	  printf("%s not found\n", argv[1]);
	  return 1;
  }

  guru(mrbbuf);
  free(mrbbuf);

  return 0;
}
