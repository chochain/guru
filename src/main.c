#include <stdio.h>
#include "guru.h"

int   do_cuda(void);

char* load_mrb_file(const char *filename)
{
  FILE *fp = fopen(filename, "rb");

  if (fp==NULL) {
    fprintf(stderr, "File not found\n");
    return NULL;
  }

  // get filesize
  fseek(fp, 0, SEEK_END);
  size_t sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  // allocate from host memory
  char *p = guru_alloc(sz);

  if (p != NULL) {
    fread(p, sizeof(char), sz, fp);
  } else {
    fprintf(stderr, "Memory allocate error.\n");
  }
  fclose(fp);

  return p;
}

int main(int argc, char **argv)
{
    //do_cuda();
	char *p = load_mrb_file(argv[1]);

    return 0;
}
