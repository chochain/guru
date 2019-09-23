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
//#define __INLINE__			__forceinline__
#define __INLINE__
#define __HOST__			__host__
#define __GPU__				__global__
#define MUTEX_LOCK(p)  		while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)  		atomicExch((int *)&p, 0)
#define CHECK_ALIGN(sz) 	assert((-(sz)&7)==0)
#else
#define __GURU__
#define __INLINE__ 			inline
#define __HOST__
#define __GPU__
#define CHECK_ALIGN(sz) 	assert((-(sz)&3)==0)
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
    GT_EMPTY   = 0x0,						// aka MRB_TT_UNDEF
    GT_NIL,									// aka MRB_TT_FREE
    GT_FALSE,								// (note) true/false threshold. see op_jmpif
    GT_TRUE,
    GT_INT,
    GT_FLOAT,
    GT_SYM,
    GT_CLASS,
    GT_PROC,								// 0x08

    GT_HAS_REF = 0x10,						// 0x10

    /* non-primitive */
    GT_OBJ = GT_HAS_REF,
    GT_ARRAY,
    GT_STR,
    GT_RANGE,
    GT_HASH
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

// pointer arithmetic, this will not work in multiple segment implementation
#define U8PADD(p, n)	((U8*)(p) + (n))					// add
#define U8PSUB(p, n)	((U8*)(p) - (n))					// sub
#define U8POFF(p1, p0)	((S32)((U8*)(p1) - (U8*)(p0)))	    // offset (downshift from 64-bit)

//===============================================================================
/*!@brief
  Guru objects
*/
typedef struct {					// 16-bytes
	GT  gt 		: 32;				// guru object type
	U32 fil;						// cached ref count
    union {							// 64-bit
		GI  	 		 i;			// TT_FIXNUM, SYMBOL
		GF 	 	 		 f;			// TT_FLOAT
        struct RClass    *cls;		// TT_CLASS
        struct RProc     *proc;		// TT_PROC
        struct RVar      *self;		// TT_OBJECT
        struct RString   *str;		// TT_STRING
#if GURU_USE_ARRAY
        struct RArray    *array;	// TT_ARRAY
        struct RRange    *range;	// TT_RANGE
        struct RHash     *hash;		// TT_HASH
#endif
        U8       		 *sym;		// C-string (only loader use.)
    };
} GV, guru_obj;		// TODO: guru_val and guru_object shared the same structure for now

//================================================================
/*!@brief
  Guru class object.
*/
typedef struct RClass {			// 32-byte
    GS       		sid;		// class name (symbol) id u32
    U32				fil;		// reserved
    struct RClass 	*super;		// guru_class[super]
    struct RProc  	*vtbl;		// guru_proc[rprocs], linked list
#if GURU_DEBUG
    char			*name;		// for debug. TODO: remove
#endif
} guru_class;

/* forward declaration */
typedef void (*guru_fptr)(guru_obj *obj, U32 argc);
struct Irep;

typedef struct RProc {			// 48-byte
    GS 	 			sid;		// u32
    U32				fil;		// reserved
    struct RIrep 	*irep;		// an IREP (Ruby code), defined in vm.h
    guru_fptr  	 	func;		// or a raw C function
    struct RProc 	*next;		// next function in linked list
#if GURU_DEBUG
    char			*cname;		// classname
    char  			*name;		// function name
#endif
} guru_proc;

#define HAS_IREP(p)		(p->func==NULL)	//

//================================================================
/*!@brief
  Guru proc object.
*/
#define GURU_HDR  		\
	U32		rc;			\
    U16  	size;		\
    U16  	n

typedef struct RString {			// 16-byte
	GURU_HDR;
	char 	*data;					//!< pointer to allocated buffer.
} guru_str;

//================================================================
/*!@brief
  physical store for Guru object instance.
*/
typedef struct RVar {				// 32-byte
	GURU_HDR;
    struct RClass 		*cls;
    struct RStore 		*ivar;
} guru_var;

typedef struct RSes {				// 16-byte
	U8P 				stdin;		// input stream
	U8P 				stdout;		// output stream
	U16 				id;
	U16 				trace;
	struct RSes 		*next;
} guru_ses;

// internal methods which uses (const char *) for static string									// in class.cu
__GURU__ guru_class *guru_add_class(const char *name, guru_class *super);						// use (char *) for static string
__GURU__ guru_proc  *guru_add_proc(guru_class *cls, const char *name, guru_fptr cfunc);

// macro for class and proc creation (assumption: guru_class *c is defined)
#define NEW_CLASS(name, super)   	guru_add_class(name, super)
#define NEW_PROC(name, cfunc)		guru_add_proc(c, name, cfunc)

#ifdef __cplusplus
}
#endif

#endif
