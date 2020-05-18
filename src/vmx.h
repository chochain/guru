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

int  vm_pool_init(U32 step);
int  vm_main_start();

int	 vm_hold(U32 mid);
int	 vm_stop(U32 mid);
int	 vm_ready(U32 mid);

__HOST__ int vm_get(U8 *ibuf);

#ifdef __cplusplus
}
#endif

class VM_Pool {
public:
    VM_Pool(int step);
    ~VM_Pool();

    __HOST__ int 	start();
    __HOST__ int 	get(char *ibuf);
private:
    class Impl;
    Impl  *_impl;
};

#endif

