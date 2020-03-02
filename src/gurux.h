/*! @file
  @brief
  GURU external type and interface definitions
*/
#ifndef GURU_SRC_GURUX_H_
#define GURU_SRC_GURUX_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int 	guru_setup(int step, int trace);
int     guru_teardown();
int 	guru_load(char *rite_name);
int 	guru_run();

#ifdef __cplusplus
}
#endif

#endif
