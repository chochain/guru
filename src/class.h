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
	GURU_HDR;
	GR				*var;		// class variables
//	U32				meta;
//	U32				super;
	struct RClass	*meta;
	struct RClass 	*super;		// guru_class[super]
    guru_proc 		*vtbl;		// c-func array (in constant memory, rc is the number of functions)

    guru_proc		*flist;		// guru_proc[rprocs], linked list
#if GURU_DEBUG
    U32				name;		// for debug. TODO: remove
#endif // GURU_DEBUG
} guru_class;

#define USER_DEF_CLASS	0x1
#define IS_BUILTIN(cls)	(!(cls->kt & USER_DEF_CLASS))

__GURU__ guru_class *guru_rom_get_class(GT cidx);
__GURU__ guru_class *guru_rom_set_class(GT cidx, const char *name, GT super_cidx, const Vfunc vtbl[], int n);

__GURU__ guru_class *guru_define_class(const U8 *name, guru_class *super);
__GURU__ guru_proc  *guru_define_method(guru_class *cls, const U8 *name, guru_fptr cfunc);
__GURU__ guru_class *guru_class_add_meta(GR *r);			// lazy add metaclass to a class

// common class functions
__GURU__ GR 		inspect(GR *v, GR *obj);				// inspect obj using v[] as stack
__GURU__ GR 		kind_of(GR *v);							// whether v1 is a kind of v0
__GURU__ guru_class *class_by_obj(GR *v);
__GURU__ guru_proc  *proc_by_sid(GR *v, GS sid);

#ifdef __cplusplus
}
#endif
#endif
