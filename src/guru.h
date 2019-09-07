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
#define __INLINE__			__forceinline__
#define __HOST__			__host__
#define __GPU__				__global__
#define MUTEX_LOCK(p)  		while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)  		atomicExch((int *)&p, 0)
#define CHECK_ALIGN(sz) 	assert((-(sz)&7)==0)
#define CHECK_NULL(p)		assert((p))
#else

#define __GURU__
#define __INLINE__ 			inline
#define __HOST__
#define __GPU__
#define CHECK_ALIGN(sz) 	assert((-(sz)&3)==0)
#define CHECK_NULL(p)		assert((p))
#endif

#define MAX_BUFFER_SIZE 4096		// 4K

//================================================================
/*!@brief
  define the value type.
*/
#if 0 // mruby vtypes, x marks implemented by guru
  MRB_TT_FALSE = 0,   /*   0 x */
  MRB_TT_FREE,        /*   1 x */
  MRB_TT_TRUE,        /*   2 x */
  MRB_TT_FIXNUM,      /*   3 x */
  MRB_TT_SYMBOL,      /*   4 x */
  MRB_TT_UNDEF,       /*   5 x */
  MRB_TT_FLOAT,       /*   6 x */

  MRB_TT_CPTR,        /*   7   */
  MRB_TT_OBJECT,      /*   8 x */
  MRB_TT_CLASS,       /*   9 x */
  MRB_TT_MODULE,      /*  10   */
  MRB_TT_ICLASS,      /*  11   */
  MRB_TT_SCLASS,      /*  12   */
  MRB_TT_PROC,        /*  13 x */
  MRB_TT_ARRAY,       /*  14 x */
  MRB_TT_HASH,        /*  15 x */
  MRB_TT_STRING,      /*  16 x */
  MRB_TT_RANGE,       /*  17 x */

  MRB_TT_EXCEPTION,   /*  18   */
  MRB_TT_FILE,        /*  19   */
  MRB_TT_ENV,         /*  20   */
  MRB_TT_DATA,        /*  21   */
  MRB_TT_FIBER,       /*  22   */
  MRB_TT_ISTRUCT,     /*  23   */
  MRB_TT_BREAK,       /*  24   */
  MRB_TT_MAXDEFINE    /*  25   */
#endif

typedef enum {
    /* primitive */
    GT_EMPTY = 0,							// aka MRB_TT_UNDEF
    GT_NIL,									// aka MRB_TT_FREE
    GT_FALSE,								// (note) true/false threshold. see op_jmpif
    GT_TRUE,
    GT_INT,									// 0x4
    GT_FLOAT,								// 0x5
    GT_SYM,									// 0x6
    GT_CLASS,								// 0x7

    GT_HAS_REF = 16,						// 0x10

    /* non-primitive */
    GT_OBJ = GT_HAS_REF,					// 0x10
    GT_PROC,								// 0x11
    GT_ARRAY,								// 0x12
    GT_STR,									// 0x13
    GT_RANGE,								// 0x14
    GT_HASH,								// 0x15
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

typedef uint32_t    *U32P;
typedef uint8_t     *U8P;

//===============================================================================
// guru simple types (non struct)
typedef S32			GI;
typedef F32	 		GF;
typedef U32 		GS;

#define U8PADD(p, n)	((U8 *)(p) + (n))					// U8 pointer arithmetic
#define U8PSUB(p, n)	((U8 *)(p) - (n))
#define U8POFF(p1, p0)	((U32)((U8 *)(p1) - (U8 *)(p0)))	// pointer offset

//================================================================
/*!@brief
  Guru class object.
*/
typedef struct RClass {			// 16-byte
    GS       		sid;		// class name (symbol) id
    struct RClass 	*super;		// guru_class[super]
    struct RProc  	*vtbl;		// guru_proc[rprocs], linked list
#if GURU_DEBUG
    char			*name;		// for debug. TODO: remove
#endif
} guru_class;

typedef struct RString {		// 8-byte
	U32 			len;		//!< string length.
	char 			*data;		//!< pointer to allocated buffer.
} guru_str;

//================================================================
/*!@brief
  physical store for Guru object instance.
*/
typedef struct RVar {			// 12-byte
    struct RClass 	*cls;
    struct RStore 	*ivar;
    U8 			   	data[];		// here pointer, instead of *data pointer to somewhere else
} guru_var;

//================================================================
/*!@brief
  Guru proc object.
*/
#define GURU_HDR  		\
	GT  	gt   : 8; 	\
	U32		flag : 8;	\
	U32		rc   : 16

//===============================================================================
/*!@brief
  Guru objects
*/
typedef struct {					// 8-bytes
	GT  	gt   : 8;				// guru object type
	U32		flag : 8;				// special attribute flags
	U32		rc   : 16;				// reference counter
    union {
        GI         		 i;			// TT_FIXNUM, SYMBOL
        GF       		 f;			// TT_FLOAT
        U8P       		 sym;		// C-string (only loader use.)
        struct RClass    *cls;		// TT_CLASS
        struct RVar      *self;		// TT_OBJECT
        struct RProc     *proc;		// TT_PROC
        struct RString   *str;		// TT_STRING
#if GURU_USE_ARRAY
        struct RArray    *array;	// TT_ARRAY
        struct RRange    *range;	// TT_RANGE
        struct RHash     *hash;		// TT_HASH
#endif
    };
} GV, guru_obj;		// TODO: guru_val and guru_object shared the same structure for now

typedef void (*guru_fptr)(guru_obj *obj, U32 argc);

/* forward declaration */
struct Irep;
typedef struct RProc {				// 16-byte
    GS 	 				sid;		// u32
    struct RProc 		*next;		// next function in linked list
    union {
        struct RIrep 	*irep;		// an IREP (Ruby code), defined in vm.h
        guru_fptr  	 	func;		// or a raw C function
    };
#if GURU_DEBUG
    char				*cname;		// classname
    char  				*name;		// function name
#endif
} guru_proc;

#define IS_CFUNC(p)		(p)

typedef struct RSes {				// 16-byte
	U8P 				in;
	U8P 				out;
	U16 				id;
	U16 				trace;
	struct RSes 		*next;
} guru_ses;

// internal methods which uses (const char *) for static string									// in class.cu
__GURU__ guru_class *guru_add_class(const char *name, guru_class *super);						// use (char *) for static string
__GURU__ guru_proc  *guru_add_proc(guru_class *cls, const char *name, guru_fptr cfunc);

#ifdef __cplusplus
}
#endif

#endif
