/*! @file
  @brief
  Guru macros and internal class definitions
*/
#ifndef GURU_SRC_GURU_H_
#define GURU_SRC_GURU_H_
#include "vm_config.h"
#include "gurux.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GURU_CUDA__

#define __GURU__ 			__device__
#define __HOST__			__host__
#define __GPU__				__global__
//#define __INLINE__
#define __INLINE__			__forceinline__
#define __UCODE__ 			__GURU__ __INLINE__ void
#define __CFUNC__			__GURU__ void
#define MUTEX_LOCK(p)  		while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)  		atomicExch((int *)&p, 0)
#define ALIGN(sz) 			((sz) += -(sz) & 0x7)
#define ALIGN64(sz)			((sz) += -(sz) & 0xf)

#define ASSERT(X) \
	if (!(X)) printf("tid %d: %s, %d\n", threadIdx.x, __FILE__, __LINE__);
#define DEVSYNC()			{ cudaDeviceSynchronize(); ASSERT(cudaGetLastError()==cudaSuccess); }

#else

#define __GURU__
#define __INLINE__ 			inline
#define __HOST__
#define __GPU__
#define ALIGN(sz) 			((sz) += -(sz) & 3)

#endif

#define MAX_BUFFER_SIZE 4096		// 4K

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
typedef S32			GI;
typedef F32	 		GF;
typedef U16 		GS;

// pointer arithmetic, this will not work in multiple segment implementation
#define U8PADD(p, n)	((U8*)(p) + (n))					// add
#define U8PSUB(p, n)	((U8*)(p) - (n))					// sub
#define U8POFF(p1, p0)	((S32)((U8*)(p1) - (U8*)(p0)))	    // offset (downshift from 64-bit)

#define ACL_HAS_REF		0x1
#define ACL_READ_ONLY	0x2
#define ACL_SCLASS		0x4
#define ACL_SELF		0x8

#define HAS_REF(v)		((v)->acl & ACL_HAS_REF)
#define HAS_NO_REF(v)	(!HAS_REF(v))
#define IS_READ_ONLY(v)	((v)->acl & ACL_READ_ONLY)
#define IS_SCLASS(v)	((v)->acl & ACL_SCLASS)
#define IS_SELF(v)		((v)->acl & ACL_SELF)

//===============================================================================
/*!@brief
  Guru VALUE and objects
	gt  : object type (GURU_TYPE i.e. GT_*)
	acl : access control (HAS_REF, READ_ONLY, ...)
	sid : object store id for user defined objects (i.e. OBJ)

  TODO: move gt into object themselves, keep fix+acl as lower 3-bits (CUDA is 32-bit ops)
*/
typedef struct {					// 16-bytes (128 bits) for eaze of debugging
	GT  	gt 	: 16;				// guru object type
	U32 	acl : 16;				// object access control (i.e. ROM able
	GS 		vid;					// variable idx (used by ostore)
	U16  	temp;					// reserved
    union {							// 64-bit
		GI  	 		 i;			// GT_INT, GT_SYM
		GF 	 	 		 f;			// GT_FLOAT
        struct RObj      *self;		// GT_OBJ
        struct RClass    *cls;		// GT_CLASS (shares same header with GT_OBJ)
        struct RProc     *proc;		// GT_PROC
        struct RIter	 *iter;		// GT_ITER
        struct RString   *str;		// GT_STR
        struct RArray    *array;	// GT_ARRAY
        struct RRange    *range;	// GT_RANGE
        struct RHash     *hash;		// GT_HASH
        U8       		 *sym;		// C-string (only loader use.)
    };
} GV;

/* forward declaration */
typedef void (*guru_fptr)(GV v[], U32 vi);
struct Irep;
struct Vfunc {
	const char  *name;			// raw string usually
	guru_fptr 	func;			// C-function pointer
};
#define VFSZ(vtbl)		(sizeof(vtbl)/sizeof(Vfunc))
//================================================================
/*!@brief
  Guru object header. (i.e. Ruby's RBasic)
    rc  : reference counter
    size: storage space for built-in complex objects (i.e. ARRAY, HASH, STRING)
    meta: meta flag for class, iterator object type
    n   : actual number of elements in built-in object
    b   : byte count for string
    sid : symbol id for class, proc
    xxx : reserved for class
*/
#define GURU_HDR  		\
	U16		rc;			\
	union {				\
		U16  	sz;		\
		U16		meta;	\
	};					\
	union {				\
		struct {		\
			U16	b;		\
			U16 n;		\
    	};				\
		struct {		\
			U16 sid;	\
			U16 xxx;	\
    	};				\
    }

//================================================================
/*! Define instance data handle.
*/
typedef struct RVar {
	GURU_HDR;					// size: allocated size, n: byte count, sid: utf8 char count
    GV *attr;					// attributes
} guru_var;

typedef struct RProc {			// 48-byte
	GURU_HDR;					// sid, meta, n are used
    union {
	    struct RIrep 	*irep;	// an IREP (Ruby code), defined in vm.h
    	guru_fptr 		func;	// or a raw C function
    };
    union {
    	struct RProc 	*next;	// next function in linked list
    	GV 				*regs;	// register file for lambda
    };
#if GURU_DEBUG
    char			*cname;		// classname
    char  			*name;		// function name
#endif
} guru_proc;

#define PROC_IREP		0x1
#define PROC_LAMBDA		0x8
#define AS_IREP(p)		((p)->meta & PROC_IREP)
#define AS_LAMBDA(p)	((p)->meta & PROC_LAMBDA)

typedef struct RString {			// 16-byte
	GURU_HDR;
	char 				*raw;		//!< pointer to allocated buffer.
} guru_str;

//================================================================
/*!@brief
  physical store for Guru object instance.
*/
typedef struct RObj {				// 32-byte
	GURU_HDR;
    struct RVar 		*ivar;		// DO NOT change here, shared structure with RClass
    struct RClass 		*cls;		// DO NOT change here, shared structure with RClass
} guru_obj;

typedef struct RSes {				// 16-byte
	U8					*stdin;		// input stream
	U8	 				*stdout;	// output stream
	U16 				id;
	U16 				trace;
	struct RSes 		*next;
} guru_ses;

#ifdef __cplusplus
}
#endif

#endif
