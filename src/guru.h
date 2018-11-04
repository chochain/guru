/*! @file
  @brief
  Guru value definitions

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef MRBC_SRC_GURU_H_
#define MRBC_SRC_GURU_H_
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
	uint8_t *req;
	uint8_t *res;
	uint8_t *vm;
} guru_ses;

// mrbc types
typedef int32_t 	mrbc_int;
typedef float 		mrbc_float;
typedef int16_t 	mrbc_sym;

//================================================================
/*!@brief
  define the value type.
*/
typedef enum {
    /* internal use */
    MRBC_TT_HANDLE = -1,
    /* primitive */
    MRBC_TT_EMPTY = 0,
    MRBC_TT_NIL,
    MRBC_TT_FALSE,		// (note) true/false threshold. see op_jmpif

    MRBC_TT_TRUE,
    MRBC_TT_FIXNUM,
    MRBC_TT_FLOAT,
    MRBC_TT_SYMBOL,
    MRBC_TT_CLASS,

    /* non-primitive */
    MRBC_TT_OBJECT = 20,
    MRBC_TT_PROC,
    MRBC_TT_ARRAY,
    MRBC_TT_STRING,
    MRBC_TT_RANGE,
    MRBC_TT_HASH,

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
        mrbc_int         i;			// MRBC_TT_FIXNUM, SYMBOL
#if MRBC_USE_FLOAT
        mrbc_float       f;			// MRBC_TT_FLOAT
#endif
        struct RClass    *cls;		// MRBC_TT_CLASS
        struct RInstance *self;		// MRBC_TT_OBJECT
        struct RProc     *proc;		// MRBC_TT_PROC
        struct RString   *str;		// MRBC_TT_STRING
#if MRBC_USE_ARRAY
        struct RArray    *array;	// MRBC_TT_ARRAY
        struct RRange    *range;	// MRBC_TT_RANGE
        struct RHash     *hash;		// MRBC_TT_HASH
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
#ifdef MRBC_DEBUG
    const char    	*name;		// for debug. TODO: remove
#endif
    mrbc_sym       	sym_id;		// class name
} mrbc_class;

#define GURU_PROC_C_FUNC 	0x80
#define IS_C_FUNC(m)		((m)->flag & GURU_PROC_C_FUNC)

#define MRBC_OBJECT_HEADER      \
	unsigned int	refc:16;	\
	mrbc_vtype  	tt:8; 		\
	unsigned int	flag:8

typedef struct RString {		// 16-byte
	MRBC_OBJECT_HEADER;			// 4-byte

	uint32_t 		size;		//!< string length.
	uint8_t  		*data;		//!< pointer to allocated buffer.
} mrbc_string;

//================================================================
/*!@brief
  Guru instance object.
*/
typedef struct RInstance {		// 24-byte
    MRBC_OBJECT_HEADER;

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
    MRBC_OBJECT_HEADER;

    struct RProc *next;
    union {
        struct Irep *irep;
        mrbc_func_t func;
    };
#ifdef MRBC_DEBUG
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
