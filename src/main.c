#include <stdio.h>
#include <stdlib.h>
#include "guru.h"

#if 0
#include "../ext/c_ext.h"
#include "../ext/c_ext_sprintf.h"

int  do_cuda(void);

int main_on_host(int argc, char **argv)
{
	char *str = host_sprintf("%s:%d(0x%x)\n", "test", 10, 10);	// from ../ext/c_ext_sprintf.c
	printf("%s", str);
	return 0;
	//do_cuda();
	mrbc_vm *vmh = (mrbc_vm *)malloc(sizeof(mrbc_vm));
	guru_init_ext(vmh, argv[1]);								// from ../ext/c_ext.c
	free(vmh);
}
#endif

int main(int argc, char **argv)
{
	guru_ses ses;

	if (session_init(&ses, argv[1])!=0) return -1;

	session_start(&ses);

    return 0;
}
