#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "gurux.h"

#if 0
#include "../ext/c_ext.h"
#include "../ext/c_ext_sprintf.h"

int  do_cuda(void);
#endif

#define TRACE_MASK  	0x3
#define VM_STEP_FLAG  	0x8000

int _opt(int argc, char *argv[], int *opt)
{
    int o, n=0;
    *opt = 0;

    while ((o = getopt(argc, argv, ":ts")) != -1) {
    	switch(o) {
    	case 's': *opt |= VM_STEP_FLAG;						break;
    	case 't':
    		o = optarg ? atoi(optarg) : 1;
    		if (optarg) n++;
    		*opt |= o & TRACE_MASK;							break;
    	case '?': printf("unknown option %c\n", optopt); 	break;
    	}
    	n++;
    }
    return n;
}

int main(int argc, char *argv[])
{
	int opt, n = _opt(argc, argv, &opt);
	int trace = opt & TRACE_MASK;
	int step  = (opt & VM_STEP_FLAG) ? 1 : 0;

	if (guru_setup(trace)) 				return -1;
	for (int i=n+1; i<argc; i++) {
		char *fname = argv[i];
		if (guru_load(fname, step, trace)) 	return -2;
	}
	return guru_run(trace);
}
