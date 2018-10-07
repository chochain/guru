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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GURU_CUDA__
#define __GURU__ __device__
#else
#define __GURU__
#endif

#define MAX_BUFFER_SIZE 1024

typedef struct guru_ses_ {
	char *req;
	char *res;
} guru_ses;

// mrbc types
typedef int32_t mrbc_int;
typedef float 	mrbc_float;
typedef int16_t mrbc_sym;

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
typedef struct RObject {
    mrbc_vtype            tt:8;
    union {
        mrbc_int          i;		// MRBC_TT_FIXNUM, SYMBOL
#if MRBC_USE_FLOAT
        mrbc_float        d;		// MRBC_TT_FLOAT
#endif
        struct RClass    *cls;		// MRBC_TT_CLASS
        struct RObject   *handle;	// handle to objects
        struct RInstance *instance;	// MRBC_TT_OBJECT
        struct RProc     *proc;		// MRBC_TT_PROC
#if MRBC_USE_ARRAY
        struct RArray    *array;	// MRBC_TT_ARRAY
        struct RRange    *range;	// MRBC_TT_RANGE
        struct RHash     *hash;		// MRBC_TT_HASH
#endif
#if MRBC_USE_STRING
        struct RString   *string;	// MRBC_TT_STRING
#endif
        const char       *str;		// C-string (only loader use.)
    };
} mrbc_object, mrbc_value;

//================================================================
/*!@brief
  Guru class object.
*/
typedef struct RClass {
    mrbc_sym       sym_id;	// class name
#ifdef MRBC_DEBUG
    const char    *names;	// for debug. delete soon.
#endif
    struct RClass *super;	// mrbc_class[super]
    struct RProc  *procs;	// mrbc_proc[rprocs], linked list
} mrbc_class;

#define MRBC_OBJECT_HEADER                          \
    uint16_t 	ref_count;                          \
    mrbc_vtype 	tt : 8  	// TODO: for debug use only.

//================================================================
/*!@brief
  Guru instance object.
*/
typedef struct RInstance {
    MRBC_OBJECT_HEADER;

    struct RClass          *cls;
    struct RKeyValueHandle *ivar;
    uint8_t 				data[];
} mrbc_instance;

//================================================================
/*!@brief
  Guru proc object.
*/
/* forward declaration */
struct IREP;
typedef void (*mrbc_func_t)(mrbc_object *v, int argc);

typedef struct RProc {
    MRBC_OBJECT_HEADER;

    unsigned int c_func : 1;	// 0:IREP, 1:C Func
    mrbc_sym sym_id;
#ifdef MRBC_DEBUG
    const char *names;		// for debug; delete soon
#endif
    struct RProc *next;
    union {
        struct IREP *irep;
        mrbc_func_t func;
    };
} mrbc_proc;

int init_session(guru_ses *ses, const char *rite_fname);

#ifdef __cplusplus
}
#endif

#endif
