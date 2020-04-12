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

#endif
