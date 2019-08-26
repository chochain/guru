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
#define __GURU__ 		__device__
#define __INLINE__		__forceinline__
#define __HOST__		__host__
#define __GPU__			__global__
#define MUTEX_LOCK(p)  	while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)  	atomicExch((int *)&p, 0)
#else
#define __GURU__
#define __INLINE__ 	inline
#define __HOST__
#define __GPU__
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
    GURU_TT_EMPTY = 0,							// aka MRB_TT_UNDEF
    GURU_TT_NIL,								// aka MRB_TT_FREE

    GURU_TT_FALSE,								// (note) true/false threshold. see op_jmpif
    GURU_TT_TRUE,

    GURU_TT_FIXNUM,								// 0x4
    GURU_TT_FLOAT,
    GURU_TT_SYMBOL,
    GURU_TT_CLASS,

    GURU_TT_HAS_REF = 16,						// 0x10

    /* non-primitive */
    GURU_TT_OBJECT = GURU_TT_HAS_REF + 4,		// 0x14 or 20
    GURU_TT_PROC,
    GURU_TT_ARRAY,
    GURU_TT_STRING,
    GURU_TT_RANGE,
    GURU_TT_HASH,
} mrbc_vtype;

#define TT_BOOL(v)		((v) ? GURU_TT_TRUE : GURU_TT_FALSE)

// short notation
typedef uint64_t    U64;
typedef uint32_t	U32;
typedef uint16_t    U16;
typedef uint8_t		U8;

typedef int32_t     S32;				// signed integer
typedef int16_t		S16;

typedef double		F64;				// double precision float
typedef float       F32;				// single precision float

typedef uint32_t    *U32P;
typedef uint8_t     *U8P;
typedef uintptr_t   U32A;				// pointer address

#define U8PADD(p, n)	((U8 *)p + n)	// U8 pointer arithmetic

// guru internal types
typedef S32 		mrbc_int;
typedef F32	 		mrbc_float;
typedef U32 		mrbc_sym;

//================================================================
/*!@brief
  Guru value object.
*/
typedef struct RObject {					// 16-bytes
    mrbc_vtype           tt   : 8;			// 8-bit
    U32				 	 flag : 8;			// reserved
    U32					 size : 16;			// reserved, 32-bit aligned
    union {
        mrbc_int         i;					// GURU_TT_FIXNUM, SYMBOL
#if GURU_USE_FLOAT
        mrbc_float       f;					// GURU_TT_FLOAT
#endif
        struct RClass    *cls;				// GURU_TT_CLASS
        struct RVar      *self;				// GURU_TT_OBJECT
        struct RProc     *proc;				// GURU_TT_PROC
        struct RString   *str;				// GURU_TT_STRING
#if GURU_USE_ARRAY
        struct RArray    *array;			// GURU_TT_ARRAY
        struct RRange    *range;			// GURU_TT_RANGE
        struct RHash     *hash;				// GURU_TT_HASH
#endif
        U8P       		 sym;				// C-string (only loader use.)
    };
} mrbc_object, mrbc_value;

//================================================================
/*!@brief
  Guru class object.
*/
typedef struct RClass {						// 16-byte
    mrbc_sym       	sym_id;					// class name
    struct RClass 	*super;					// mrbc_class[super]
    struct RProc  	*vtbl;					// mrbc_proc[rprocs], linked list
#ifdef GURU_DEBUG
    U8P       		name;					// for debug. TODO: remove
#endif
} mrbc_class;

#define GURU_CFUNC 	0x80
#define IS_CFUNC(m)	((m)->flag & GURU_CFUNC)

#define GURU_OBJECT_HEADER      \
	mrbc_vtype  	tt   : 8; 	\
	U32				flag : 8;	\
	U32				refc : 16;

typedef struct RString {		// 12-byte
	GURU_OBJECT_HEADER;			// 4-byte

	U32 size;					//!< string length.
	U8P data;					//!< pointer to allocated buffer.
} mrbc_string;

//================================================================
/*!@brief
  physical store for Guru object instance.
*/
typedef struct RVar {			// 16-byte
    GURU_OBJECT_HEADER;

    struct RClass *cls;
    struct RStore *ivar;
    U8 			   data[];		// here pointer, instead of *data pointer to somewhere else
} mrbc_var;

//================================================================
/*!@brief
  Guru proc object.
*/
/* forward declaration */
struct Irep;
typedef void (*mrbc_func_t)(mrbc_object *v, U32 argc);

typedef struct RProc {			// 40-byte
    GURU_OBJECT_HEADER;

    mrbc_sym 	 sym_id;		// u32
    struct RProc *next;
    union {
        struct RIrep *irep;
        mrbc_func_t  func;
    };
#ifdef GURU_DEBUG
    U8P     	 name;			// for debug. TODO: remove
#endif
} mrbc_proc;

typedef struct guru_ses_ {
	U8P in;
	U8P out;
	U16 id;
	U16 trace;

	struct guru_ses_ *next;
} guru_ses;

// internal methods which uses (const char *) for static string									// in class.cu
__GURU__ mrbc_class *guru_add_class(const char *name, mrbc_class *super);						// use (char *) for static string
__GURU__ void       guru_add_proc(mrbc_class *cls, const char *name, mrbc_func_t cfunc);

#ifdef __cplusplus
}
#endif

#endif
