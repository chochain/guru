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
typedef struct RClass {			// 52-byte
	GURU_HDR;					// n and sid is used
    struct RClass 	*super;		// guru_class[super]
    struct RProc  	*vtbl;		// guru_proc[rprocs], linked list
    struct RObj  	*cvar;		// class var
    struct RClass   *meta;		// guru meta class
#if GURU_DEBUG
    char			*name;		// for debug. TODO: remove
#endif // GURU_DEBUG
} guru_class;

#define META_FLAG		0x1
#define IS_META(v)		((v)->gt==GT_CLASS && ((v)->cls->n & META_FLAG))
#define SET_META(v)		((v)->cls->n |= META_FLAG)

// external methods uses static string (const char *) 												// in class.cu
__GURU__ guru_class *guru_add_class(const char *name, guru_class *super, Vfunc vtbl[], int n);		// use (char *) for static string

// internal methods (used by ucode)
__GURU__ guru_class *guru_define_class(const U8 *name, guru_class *super);
__GURU__ guru_proc  *guru_define_method(guru_class *cls, const U8 *name, guru_fptr cfunc);
__GURU__ guru_class *class_by_obj(GV *v);
__GURU__ guru_proc  *proc_by_sid(GV *v, GS sid);

#ifdef __cplusplus
}
#endif
#endif
