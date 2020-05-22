/*! @file
  @brief
  Guru macros and internal class definitions
*/
#ifndef GURU_SRC_GURU_H_
#define GURU_SRC_GURU_H_
#include <stdint.h>
#include "guru_config.h"

#if GURU_USE_CONSOLE		// use guru local implemented print functions (in puts, sprintf.cu)
#define PRINTF				guru_printf
#define VPRINTF				guru_vprintf
#else						// use CUDA printf function
#include <stdio.h>
#define PRINTF				printf
#endif // GURU_USE_CONSOLE
#define NA(msg)				({ PRINTF("method not supported: %s\n", msg); })

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__CUDACC__)

#define __GURU__ 			__device__
#define __HOST__			__host__
#define __GPU__				__global__
//#define __INLINE__
#define __INLINE__			__forceinline__
#define __UCODE__ 			__GURU__ void
#define __CFUNC__			__GURU__ void
#define MUTEX_LOCK(p)  		while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)  		atomicExch((int *)&p, 0)
#define ALIGN4(sz)			((sz) + (-(sz) & 0x3))
#define ALIGN8(sz) 			((sz) + (-(sz) & 0x7))
#define ALIGN16(sz)  		((sz) + (-(sz) & 0xf))

#define ASSERT(X) \
	if (!(X)) PRINTF("ASSERT tid %d: line %d in %s\n", threadIdx.x, __LINE__, __FILE__);
#define GPU_SYNC()			{ cudaDeviceSynchronize(); }
#define GPU_CHK()			{ cudaDeviceSynchronize(); ASSERT(cudaGetLastError()==cudaSuccess); }

#else

#define __GURU__
#define __INLINE__ 			inline
#define __HOST__
#define __GPU__
#define ALIGN(sz) 			((sz) + (-(sz) & 3))
#define ASSERT(X) 			assert(x)

#endif

#define OUTPUT_BUF_SIZE 	4096			// 4K

//================================================================
/*!@brief
  define the value type.
*/
typedef enum {
    /* primitive */
    GT_EMPTY   = 0x0,						// aka MRB_TT_UNDEF
    GT_NIL,									// aka MRB_TT_FREE
    GT_FALSE,								// (note) true/false threshold. see op_jmpif
    GT_TRUE,
    GT_INT,
    GT_FLOAT,
    GT_SYM,

    GT_CLASS 	= 0x8,
    GT_PROC,
    GT_SYS,

    /* non-primitive */
    GT_OBJ 		= 0x10,
    GT_ARRAY,
    GT_STR,
    GT_RANGE,
    GT_HASH,
    GT_ITER,

    GT_MAX      = 0x18
} GT;

#define GT_BOOL(v)		((v) ? GT_TRUE : GT_FALSE)

// short notation
typedef uint64_t    U64;
typedef uint32_t	U32;
typedef uint16_t    U16;
typedef uint8_t		U8;

typedef int32_t     S32;					// signed integer
typedef int16_t		S16;
typedef uintptr_t   U32A;					// pointer address

typedef double		F64;					// double precision float
typedef float       F32;					// single precision float

typedef uint32_t    U32;
typedef uint8_t     U8;

//===============================================================================
// guru simple types (non struct)
typedef S32			GI;						// signed integer
typedef F32	 		GF;						// float
typedef U16			GS;						// symbol index
typedef S32			GP;						// offset, i.e. object pointer

// pointer arithmetic, this will not work in multiple segment implementation
#define U8PADD(p, n)	((U8*)(p) + (n))					// add
#define U8PSUB(p, n)	((U8*)(p) - (n))					// sub
#define U8POFF(p1, p0)	((S32)((U8*)(p1) - (U8*)(p0)))	    // offset (downshift from 64-bit)

#define ACL_HAS_REF		0x1
#define ACL_SCLASS		0x2
#define ACL_SELF		0x4

#define HAS_REF(v)		((v)->acl & ACL_HAS_REF)
#define HAS_NO_REF(v)	(!HAS_REF(v))
#define IS_SCLASS(v)	((v)->acl & ACL_SCLASS)
#define IS_SELF(v)		((v)->acl & ACL_SELF)

//===============================================================================
/*!@brief
  Guru VALUE and objects
	gt  : object type (GURU_TYPE i.e. GT_*)
	acl : access control (HAS_REF, READ_ONLY, ...)
	oid : object store id for user defined objects (i.e. OBJ)
	len : used by transcoding length
	xxx : reserved

  TODO: move gt into object themselves, keep fix+acl as lower 3-bits (CUDA is 32-bit ops)
*/
typedef struct {					// 16-bytes (128 bits) for ease of debugging
	GT  	gt   : 8;
	U32     acl  : 8;
	GS		oid  : 16;
	union {
		GI  	i;					// INT, SYM				32-bit
		GF 	 	f;					// FLOAT			 	32-bit
		GP	 	off;				// offset to object		32-bit
/*
		S32     str;				// STR		->RString
		S32     rng;				// RANGE 	->RRange
		S32		ary;				// ARRAY	->RArray
		S32		hsh;				// HASH		->RHash
		S32		itr;				// ITER		->RIter
		S32		obj;				// OBJ		->RObj
		S32		prc;				// PROC		->RProc
		S32		cls;				// CLASS	->RClass
*/
	};
} GR;

#define GR_OFF(r)	(MEMPTR((r)->off))
#define GR_STR(r)	((struct RString*)GR_OFF(r))
#define GR_RNG(r)	((struct RRange*) GR_OFF(r))
#define GR_ARY(r)	((struct RArray*) GR_OFF(r))
#define GR_HSH(r)	((struct RHash*)  GR_OFF(r))
#define GR_ITR(r)	((struct RIter*)  GR_OFF(r))
#define GR_OBJ(r)	((struct RObj*)   GR_OFF(r))
#define GR_PRC(r)	((struct RProc*)  GR_OFF(r))
#define GR_CLS(r)	((struct RClass*) GR_OFF(r))

#define _CLS(off)	((struct RClass*)MEMPTR(off))
#define _PRC(off)   ((struct RProc*)(off ? MEMPTR(off) : NULL))
#define _STATE(off) ((struct RState*)(off ? MEMPTR(off) : NULL))
#define _STR(off)	((U8*)MEMPTR(off))
#define _RAW(r)		((U8*)MEMPTR(GR_STR(r)->raw))
#define _REGS(r)	((GR*)MEMPTR((r)->regs))
#define _VAR(r)		((GR*)((r)->var ? MEMPTR((r)->var) : NULL))

#define _CALL(prc, r, ri)	(((guru_fptr)MEMPTR(_PRC(prc)->func))(r, ri));

/* forward declarations */
typedef void (*guru_fptr)(GR v[], U32 vi);
struct Irep;
struct Vfunc {
	const char  *name;			// raw string usually
	guru_fptr 	func;			// C-function pointer
};
#define VFSZ(vtbl)		(sizeof(vtbl)/sizeof(Vfunc))

//================================================================
/*!@brief
  Guru object header. (i.e. Ruby's RBasic)
    rc  : reference counter, sizeof class->vtbl[]
    kt  : [class,function] type for Class, Proc, lambda, iterator object type
        : proc=[0=Built-in C-func|PROC_IREP|PROC_LAMBDA]
        : cls =[0=Built-in class|CLASS_BY_USER]
    n   : Array, Hash actual number of elements in built-in object, or
        : Proc parameter count
        : Iterator range object type (i.e. GT_*)
    sz  : storage space for built-in complex objects (i.e. Array, Hash, String)
    bsz : byte count for string
    pid : proc id for Proc
    cid : class id for Class, Proc
    i   : 32-bit value used by Iterator
*/
#define GURU_HDR  		\
	U16		rc : 12;	\
	U16		kt : 4;		\
	U16     n;			\
	union {				\
		struct {		\
			U16 sz;		\
			U16	bsz;	\
		};				\
		struct {		\
			U16 pid;	\
			U16 cid;	\
		};				\
		GI i;			\
    }

//================================================================
/*!@brief
  physical store for Guru object instance.
*/
typedef struct RObj {			// 24-byte
	GURU_HDR;
	GP				var;		// (GR*) instance variables
	GP				cls;		// (RClass*) class that this object belongs to (RClass*)
} guru_obj;

typedef struct RString {	// 16-byte
	GURU_HDR;
	GP				raw;	// (U8*) pointer to allocated buffer.
	S32				xxx;	// reserved
} guru_str;

typedef struct RSes {			// 16-byte
	U8				*stdin;		// input stream
	U8	 			*stdout;	// output stream
	U16 			id;
	U16 			trace;
	struct RSes 	*next;
} guru_ses;

#ifdef __cplusplus
}
#endif

#endif
