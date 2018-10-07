#include <stdio.h>
#include "guru.h"

extern int do_cuda(void);

int main(int argc, char **argv)
{
	guru_ses ses;
    //do_cuda();
	int rst = init_session(&ses, argv[1]);

    return 0;
}
