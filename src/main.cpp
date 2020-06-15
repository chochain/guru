#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include "gurux.h"
#include "debug.h"

#define TRACE_MASK  	0x3
#define VM_EXEC_FLAG  	0x8000

int _opt(int argc, char *argv[], int *opt)
{
    int o, n=0;
    *opt = 0;

    while ((o = getopt(argc, argv, "xt:")) != -1) {
    	switch(o) {
    	case 'x': *opt |= VM_EXEC_FLAG;						break;
    	case 't':
    		o = (*optarg >= '0') ? atoi(optarg) : 1;
    		*opt |= o & TRACE_MASK;							break;
    	case '?':
    	default:
    		printf(
    			"Usage:> %s [options] fname1.mrb [[fname2.mrb] ...]\n"
    			"\toptions:\n"
    			"\t-tn : where n is trace level\n"
    			"\t      0: no tracing\n"
    			"\t      1: stack dump\n"
    			"\t      2: heap free list\n"
        		"\t      3: stack + free list\n"
    			"\t-x  : execute entirely inside VM\n",
    			argv[0]);
    		exit(-1);
    	}
    	n++;
    }
    return n;
}

int main(int argc, char *argv[])
{
	int opt, n = _opt(argc, argv, &opt);
	int trace  = opt & TRACE_MASK;
	int step   = (opt & VM_EXEC_FLAG) ? 0 : 1;

#if GURU_CXX_CODEBASE
	Guru *guru = new Guru(step, trace);

	for (int i=n+1; i<argc; i++) {					// TODO: producer
		char *fname = argv[i];
		if (guru->load(fname)) return -2;
	}
	return guru->run();								// TODO: consumer
#else
	if (signal(SIGINT, guru_teardown)==SIG_ERR) {	// register interrupt handler
		fprintf(stderr, "ERROR: SIGINT, use kill -9");
	}

	int rst = guru_setup(step, trace);
	if (rst) {										// setup error?
		debug_error(rst);
		return -1;
	}

	for (int i=n+1; i<argc; i++) {					// TODO: producer
		char *fname = argv[i];
		if (guru_load(fname)) return -2;
	}
	guru_run();										// TODO: consumer
	guru_teardown(0);

	return 0;
#endif // GURU_CXX_CODEBASE
}
