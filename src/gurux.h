/*! @file
  @brief
  GURU external type and interface definitions
*/
#ifndef GURU_SRC_GURUX_H_
#define GURU_SRC_GURUX_H_

#ifdef __cplusplus
extern "C" {
#endif

int 	guru_setup(int step, int trace);
int 	guru_load(char *rite_name);
int 	guru_run();
void    guru_teardown(int sig);

#ifdef __cplusplus
}
#endif
#define GURU_CXX_CODEBASE	1

#if GURU_CXX_CODEBASE
class Guru								// interface class
{
	class Impl;
	Impl	*_impl;
	int		_trace;

public:
	Guru(int step, int trace);
	~Guru();

	int	load(char *rite_name);
	int	run();
};
#endif // GURU_CXX_CODEBASE
#endif // GURU_SRC_GURUX_H_
