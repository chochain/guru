/*! @file
  @brief
  Guru value definitions

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef GURU_SRC_GURU_H_
#define GURU_SRC_GURU_H_
#include "vm_config.h"

#include <stdint.h>					// uint8_t, int32_t, ...

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GURU_CUDA__
#define __GURU__ 	__device__
#define __INLINE__	__forceinline__
#else
#define __GURU__
#define __INLINE__ 	inline
#endif

#define MAX_BUFFER_SIZE 4096		// 4K

typedef struct guru_ses_ {
	uint8_t  *req;
	uint8_t  *res;
	uint8_t  *vm;
	uint32_t debug;
} guru_ses;

// mrbc types
typedef int32_t 	mrbc_int;
typedef float 		mrbc_float;
typedef int16_t 	mrbc_sym;

//================================================================
/*!@brief
  define the value type.
*/
#if 0                 // mruby vtypes
  MRB_TT_FALSE = 0,   /*   0 */
  MRB_TT_FREE,        /*   1 */
  MRB_TT_TRUE,        /*   2 */
  MRB_TT_FIXNUM,      /*   3 */
  MRB_TT_SYMBOL,      /*   4 */
  MRB_TT_UNDEF,       /*   5 */
  MRB_TT_FLOAT,       /*   6 */
  MRB_TT_CPTR,        /*   7 */
  MRB_TT_OBJECT,      /*   8 */
  MRB_TT_CLASS,       /*   9 */
  MRB_TT_MODULE,      /*  10 */
  MRB_TT_ICLASS,      /*  11 */
  MRB_TT_SCLASS,      /*  12 */
  MRB_TT_PROC,        /*  13 */
  MRB_TT_ARRAY,       /*  14 */
  MRB_TT_HASH,        /*  15 */
  MRB_TT_STRING,      /*  16 */
  MRB_TT_RANGE,       /*  17 */
  MRB_TT_EXCEPTION,   /*  18 */
  MRB_TT_FILE,        /*  19 */
  MRB_TT_ENV,         /*  20 */
  MRB_TT_DATA,        /*  21 */
  MRB_TT_FIBER,       /*  22 */
  MRB_TT_ISTRUCT,     /*  23 */
  MRB_TT_BREAK,       /*  24 */
  MRB_TT_MAXDEFINE    /*  25 */
#endif

typedef enum {
    /* primitive */
    GURU_TT_EMPTY = 0,
    GURU_TT_NIL,

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

//================================================================
/*!@brief
  Guru value object.
*/
typedef struct RObject {			// 16-bytes
    mrbc_vtype           tt:8;
    unsigned int		 flag:8;	// reserved
    unsigned int	 	 size:16;	// reserved, 32-bit aligned
    union {
        mrbc_int         i;			// GURU_TT_FIXNUM, SYMBOL
#if GURU_USE_FLOAT
        mrbc_float       f;			// GURU_TT_FLOAT
#endif
        struct RClass    *cls;		// GURU_TT_CLASS
        struct RInstance *self;		// GURU_TT_OBJECT
        struct RProc     *proc;		// GURU_TT_PROC
        struct RString   *str;		// GURU_TT_STRING
#if GURU_USE_ARRAY
        struct RArray    *array;	// GURU_TT_ARRAY
        struct RRange    *range;	// GURU_TT_RANGE
        struct RHash     *hash;		// GURU_TT_HASH
#endif
        const char       *sym;		// C-string (only loader use.)
    };
} mrbc_object, mrbc_value;

//================================================================
/*!@brief
  Guru class object.
*/
typedef struct RClass {			// 32-byte
    struct RClass 	*super;		// mrbc_class[super]
    struct RProc  	*procs;		// mrbc_proc[rprocs], linked list
#ifdef GURU_DEBUG
    const char    	*name;		// for debug. TODO: remove
#endif
    mrbc_sym       	sym_id;		// class name
} mrbc_class;

#define GURU_PROC_C_FUNC 	0x80
#define IS_C_FUNC(m)		((m)->flag & GURU_PROC_C_FUNC)

#define GURU_OBJECT_HEADER      \
	unsigned int	refc:16;	\
	mrbc_vtype  	tt:8; 		\
	unsigned int	flag:8

typedef struct RString {		// 16-byte
	GURU_OBJECT_HEADER;			// 4-byte

	uint32_t 		size;		//!< string length.
	uint8_t  		*data;		//!< pointer to allocated buffer.
} mrbc_string;

//================================================================
/*!@brief
  Guru instance object.
*/
typedef struct RInstance {		// 24-byte
    GURU_OBJECT_HEADER;

    struct RClass    *cls;
    struct RKeyValue *ivar;
    uint8_t 		 data[];
} mrbc_instance;

//================================================================
/*!@brief
  Guru proc object.
*/
/* forward declaration */
struct Irep;
typedef void (*mrbc_func_t)(mrbc_object *v, int argc);

typedef struct RProc {		// 40-byte
    GURU_OBJECT_HEADER;

    struct RProc *next;
    union {
        struct RIrep *irep;
        mrbc_func_t  func;
    };
#ifdef GURU_DEBUG
    const char 	 *name;		// for debug; delete soon
#endif
    mrbc_sym 	 sym_id;
} mrbc_proc;

int session_init(guru_ses *ses, const char *rite_fname);
int session_start(guru_ses *ses);

#ifdef __cplusplus
}
#endif

#endif
