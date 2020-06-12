/*! @file
  @brief
  GURU - VM public interfaces

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  Fetch mruby VM bytecodes, decode and execute.

  </pre>
*/

#ifndef GURU_SRC_VMX_H_
#define GURU_SRC_VMX_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

int  vm_pool_init(int step);
int  vm_main_start();

int	 vm_hold(int mid);
int	 vm_stop(int mid);
int	 vm_ready(int mid);

__HOST__ int vm_get(char *ibuf);

#ifdef __cplusplus
}
#endif

#if GURU_CXX_CODEBASE
class VM_Pool {
    class Impl;
    Impl  *_impl;

public:
    VM_Pool(int step, int trace);
    ~VM_Pool();

    __HOST__ int 	start();
    __HOST__ int 	get(char *ibuf);
};
#endif // GURU_CXX_CODEBASE
#endif // GURU_SRC_VMX_H_

