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

int guru_setup(int trace);
int guru_load(char **argv, int n, int trace);
int guru_run(int trace);

#ifdef __cplusplus
}
#endif

#endif
