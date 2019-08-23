#include <stdio.h>
#include <stdlib.h>
#include "gurux.h"

#if 0
#include "../ext/c_ext.h"
#include "../ext/c_ext_sprintf.h"

int  do_cuda(void);
#endif

int main(int argc, char **argv)
{
	int trace = *argv[argc-1]=='1' ? 1 : (*argv[argc-1]=='2' ? 2 : 0);
	int n     = argc - (trace ? 2 : 1);

	if (guru_system_setup(trace)) return -1;

	guru_ses *ses = malloc(sizeof(ses) * n);
	for (int i=1; i<=n; i++, ses++) {
		guru_session_add(ses, argv[i], trace);
	}
	return guru_system_run(trace);
}
