/*! @file
  @brief
  GURU class building functions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#ifndef GURU_SRC_CLASS_H_
#define GURU_SRC_CLASS_H_

#include "guru.h"

#ifdef __cplusplus
extern "C" {
#endif

//================================================================
/*!@brief
  Guru class object.
*/
typedef struct RClass {			// 64-byte
	GURU_HDR;					// rc, n, sid are used
	struct RVar		*ivar;		// DO NOT change here, shared structure with RObj
	struct RClass	*cls;		// DO NOT change here, shared structure with RObj
	struct RClass 	*super;		// guru_class[super]
    guru_proc 		*vtbl;		// c-func array (in constant memory, rc is the number of functions)

    guru_proc		*plist;		// guru_proc[rprocs], linked list
#if GURU_DEBUG
    char			*name;		// for debug. TODO: remove
#endif // GURU_DEBUG
} guru_class;

#define CLASS_USER		0x8
#define IS_BUILTIN(cls)	(!(cls->meta & CLASS_USER))

__GURU__ guru_class *guru_rom_get_class(GT cidx);
__GURU__ guru_class *guru_rom_set_class(GT cidx, const char *name, GT super_cidx, const Vfunc vtbl[], int n);

__GURU__ guru_class *guru_define_class(const U8 *name, guru_class *super);
__GURU__ guru_proc  *guru_define_method(guru_class *cls, const U8 *name, guru_fptr cfunc);
__GURU__ guru_class *class_by_obj(GV *v);
__GURU__ guru_proc  *proc_by_sid(GV *v, GS sid);

#ifdef __cplusplus
}
#endif
#endif
