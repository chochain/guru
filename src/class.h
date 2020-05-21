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
	GP				var;		// (GR*) class variables
	GP				meta;		// (RClass*) offset to guru_class*
	GP				super;		// (RClass*) offset to guru_class*
    GP				vtbl;		// (RProc*) c-func array (in constant memory, rc is the number of functions)
    GP				flist;		// (RProc*) head of guru_proc linked list
#if GURU_DEBUG
    GP				name;		// (U8*) for debug. TODO: remove
#endif // GURU_DEBUG
} guru_class;

#define USER_DEF_CLASS	0x1
#define IS_BUILTIN(clsx)	(!(clsx->kt & USER_DEF_CLASS))

__GURU__ GP 	guru_rom_get_class(GT cidx);
__GURU__ GP 	guru_rom_set_class(GT cidx, const char *name, GT super_cidx, const Vfunc vtbl[], int n);
__GURU__ GP 	guru_define_class(const U8 *name, GP super);
__GURU__ GP 	guru_class_add_meta(GR *r);				// lazy add metaclass to a class

__GURU__ GP     guru_define_method(GP cls, const U8 *name, GP cfunc);

// common class functions
__GURU__ GR 	inspect(GR *v, GR *obj);				// inspect obj using v[] as stack
__GURU__ GR 	kind_of(GR *v);							// whether v1 is a kind of v0
__GURU__ GP		class_by_obj(GR *v);
__GURU__ GP  	proc_by_sid(GR *v, GS sid);

#ifdef __cplusplus
};
#endif

#if GURU_CXX_CODEBASE
class ClassMgr
{
	class  Impl;
	Impl        *_impl;
	guru_class 	_class_list[GT_MAX];

	__GURU__ ClassMgr();										// private constructor for singleon class
	__GURU__ ~ClassMgr();
public:

	static __GURU__ ClassMgr *getInstance();

	__GURU__ GP	rom_get_class(GT cidx);
	__GURU__ GP	rom_set_class(GT cidx, const char *name, GT super_cidx, const Vfunc vtbl[], int n);

	__GURU__ GP	define_class(const U8 *name, GP super);
	__GURU__ GP	class_add_meta(GR *r);				// lazy add metaclass to a class
	__GURU__ GP	define_method(GP cls, const U8 *name, GP cfunc);

	// common class functions
	__GURU__ GR	inspect(GR *v, GR *obj);				// inspect obj using v[] as stack
	__GURU__ GR	kind_of(GR *v);							// whether v1 is a kind of v0
	__GURU__ GP	class_by_obj(GR *v);
	__GURU__ GP	proc_by_sid(GR *v, GS sid);
	__GURU__ GR send(GR r[], GR *rcv, const U8 *method, U32 argc, ...);
};
#define CLS_MGR		(ClassMgr::getInstance())
#endif // GURU_CXX_CODEBASE

#endif
