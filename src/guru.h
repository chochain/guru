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

char* guru_alloc(size_t sz);

// mrbc types
typedef int32_t mrbc_int;
typedef float 	mrbc_float;
typedef int16_t mrbc_sym;

/* aspec access ? */
#define MRB_ASPEC_REQ(a)          (((a) >> 18) & 0x1f)
#define MRB_ASPEC_OPT(a)          (((a) >> 13) & 0x1f)
#define MRB_ASPEC_REST(a)         (((a) >> 12) & 0x1)
#define MRB_ASPEC_POST(a)         (((a) >> 7) & 0x1f)

#define MRBC_OBJECT_HEADER                          \
    uint16_t ref_count;                             \
    mrbc_vtype tt : 8  // TODO: for debug use only.

/* forward declaration */
struct IREP;
struct RObject;
typedef void (*mrbc_func_t)(struct RObject *v, int argc);
    
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
  define the error code. (BETA TEST)
*/
typedef enum {
    E_NOMEMORY_ERROR = 1,
    E_RUNTIME_ERROR,
    E_TYPE_ERROR,
    E_ARGUMENT_ERROR,
    E_INDEX_ERROR,
    E_RANGE_ERROR,
    E_NAME_ERROR,
    E_NOMETHOD_ERROR,
    E_SCRIPT_ERROR,
    E_SYNTAX_ERROR,
    E_LOCALJUMP_ERROR,
    E_REGEXP_ERROR,
    E_NOTIMP_ERROR,
    E_FLOATDOMAIN_ERROR,
    E_KEY_ERROR,
} mrbc_error_code;



//================================================================
/*!@brief
  Guru value object.
*/
struct RObject {
    mrbc_vtype tt : 8;
    union {
        mrbc_int i;			// MRBC_TT_FIXNUM, SYMBOL
#if MRBC_USE_FLOAT
        mrbc_float d;		// MRBC_TT_FLOAT
#endif
        struct RClass *cls;		// MRBC_TT_CLASS
        struct RObject *handle;	// handle to objects
        struct RInstance *instance;	// MRBC_TT_OBJECT
        struct RProc *proc;		// MRBC_TT_PROC
        struct RArray *array;	// MRBC_TT_ARRAY
        struct RString *string;	// MRBC_TT_STRING
        const char *str;		// C-string (only loader use.)
        struct RRange *range;	// MRBC_TT_RANGE
        struct RHash *hash;		// MRBC_TT_HASH
    };
};
typedef struct RObject mrb_object;	// not recommended.
typedef struct RObject mrb_value;	// not recommended.
typedef struct RObject mrbc_object;
typedef struct RObject mrbc_value;

//================================================================
/*!@brief
  Guru class object.
*/
typedef struct RClass {
    mrbc_sym sym_id;	// class name
#ifdef MRBC_DEBUG
    const char *names;	// for debug. delete soon.
#endif
    struct RClass *super;	// mrbc_class[super]
    struct RProc *procs;	// mrbc_proc[rprocs], linked list
} mrbc_class;
typedef struct RClass mrb_class;

//================================================================
/*!@brief
  Guru instance object.
*/
typedef struct RInstance {
    MRBC_OBJECT_HEADER;

    struct RClass *cls;
    struct RKeyValueHandle *ivar;
    uint8_t data[];
} mrbc_instance;
typedef struct RInstance mrb_instance;


//================================================================
/*!@brief
  Guru proc object.
*/
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
typedef struct RProc mrb_proc;

#ifdef __cplusplus
}
#endif

#endif
