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
typedef struct RClass {			// 32-byte
	GURU_HDR;
	GP				ivar;		// (GR*) class-level instance variables
	GP				klass;		// (RClass*) offset to *metaclass
	GP				csrc;		// (RClass*) offset to lexical scope (i.e. *guru_class itself or module source)
	GP				super;		// (RClass*) offset to *guru_class
    GP				mtbl;		// (RProc*) c-func array (in constant memory, rc is the number of functions)
    GP				flist;		// (RProc*) head of guru_proc linked list
} guru_class;

#define CLASS_BUILTIN	0x1
#define CLASS_SINGLETON 0x2
#define CLASS_EXTENDED	0x4
#define IS_BUILTIN(cx)		(cx->kt & CLASS_BUILTIN)
#define IS_SINGLETON(cx)	(cx->kt & CLASS_SINGLETON)
#define IS_EXTENDED(cx)		(cx->kt & CLASS_EXTENDED)

//================================================================
/*! Define instance data handle.
*/
typedef struct RProc {		// 32-byte
	GURU_HDR;				// n, sid, kt are used
    union {
		struct {
			GP	irep;		// (RIrep*) an IREP (Ruby code), defined in vm.h
	    	GP 	regs;		// (GR*) pointer to register file for lambda
		};
    	struct {
			GP 	func;		// (guru_fptr) for a raw C function
	    	GP	next;		// (RProc*) next function in linked list
		};
    };
} guru_proc;

#define PROC_IREP		0x1
#define PROC_LAMBDA		0x2
#define AS_IREP(px)		((px)->kt & PROC_IREP)
#define AS_LAMBDA(px)	((px)->kt & PROC_LAMBDA)

__GURU__ GP 	guru_define_class(guru_class *cx, GS cid, GP super);
__GURU__ GP		guru_class_include(GP super, GP mod);
__GURU__ GP 	guru_create_metaclass(GR *r);			// add metaclass to a class or an object

// common class functions
__GURU__ GR 	inspect(GR *v, GR *obj);				// inspect obj using v[] as stack
__GURU__ GR 	kind_of(GR *v);							// whether v1 is a kind of v0
__GURU__ GP		find_class_by_obj(GR *v);
__GURU__ GP		find_class_by_id(GS cid);
__GURU__ GP  	find_proc(GR *v, GS pid);

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
	__GURU__ static ClassMgr *getInstance();

	__GURU__ GP	define_class(const U8 *name, GP super);
	__GURU__ GP	class_add_meta(GR *r);				// lazy add metaclass to a class
	__GURU__ GP	define_method(GP cls, const U8 *name, GP cfunc);

	// common class functions
	__GURU__ GR	inspect(GR *v, GR *obj);				// inspect obj using v[] as stack
	__GURU__ GR	kind_of(GR *v);							// whether v1 is a kind of v0
	__GURU__ GP	find_class_by_obj(GR *v);
	__GURU__ GP find_class_by_id(GS cid);
	__GURU__ GP	find_proc(GR *v, GS pid);
	__GURU__ GR send(GR r[], GR *rcv, const U8 *method, U32 argc, ...);
};
#define CLS_MGR		(ClassMgr::getInstance())
#endif // GURU_CXX_CODEBASE

#endif // GURU_SRC_CLASS_H_
