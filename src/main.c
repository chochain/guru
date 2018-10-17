#include <stdio.h>
#include <stdlib.h>
#include "guru.h"
#include "c_ext.h"

int  do_cuda(void);
void dump_vm(mrbc_vm *vm);
void run_vm(mrbc_vm *vm);

int main(int argc, char **argv)
{
    //do_cuda();
	mrbc_vm *vmh = (mrbc_vm *)malloc(sizeof(mrbc_vm));
	guru_init_ext(vmh, argv[1]);
	free(vmh);

	guru_ses ses;
	mrbc_vm *vmd = (mrbc_vm *)init_session(&ses, argv[1]);
	dump_vm(vmd);

	run_vm(vmd);

    return 0;
}
