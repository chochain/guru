#include <stdio.h>
#include "guru.h"

extern int do_cuda(void);

guru_ses* load_mrb_file(const char *filename)
{
  guru_ses *ses;

  FILE *fp = fopen(filename, "rb");

  if (fp==NULL) {
    fprintf(stderr, "File not found\n");
    return NULL;
  }

  // get filesize
  fseek(fp, 0, SEEK_END);
  size_t sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  int err = guru_init(ses, sz, MAX_BUFFER_SIZE);

  if (err!=0) {
	  fprintf(stderr, "Memory allocate error.\n");
  }
  else {
	  fread(ses->req, sizeof(char), sz, fp);
  }
  fclose(fp);

  return ses;
}

int main(int argc, char **argv)
{
    //do_cuda();
	guru_ses *p = load_mrb_file(argv[1]);

    return 0;
}
