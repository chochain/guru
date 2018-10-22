#include <stdio.h>
#include <stdlib.h>
#include "guru.h"
#include "c_ext.h"

int  do_cuda(void);
char *host_sprintf(const char *fstr, ...);

int main(int argc, char **argv)
{
	char *str = host_sprintf("%s:%d(0x%x)\n", "test", 10, 10);
	printf("%s", str);
	return 0;

#ifdef MRBC_DEBUG
	//do_cuda();
	mrbc_vm *vmh = (mrbc_vm *)malloc(sizeof(mrbc_vm));
	guru_init_ext(vmh, argv[1]);
	free(vmh);
#endif
	guru_ses ses;

	if (session_init(&ses, argv[1])!=0) return -1;

	session_start(&ses);

    return 0;
}
