#include <stdio.h>
#include <stdlib.h>
#include "guru.h"

#if 0
#include "../ext/c_ext.h"
#include "../ext/c_ext_sprintf.h"

int  do_cuda(void);
#endif

int main(int argc, char **argv)
{
	guru_ses ses;

	ses.debug = argc<3 ? 0 : (*argv[2]=='1' ? 1 : 2);

	if (session_init(&ses, argv[1])!=0) return -1;

	session_start(&ses);

    return 0;
}
