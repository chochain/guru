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

// mrbc types
typedef int32_t mrbc_int;
typedef double  mrbc_float;
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

// for C call
#define SET_RETURN(n)		do { mrbc_value nnn = (n);  \
        mrbc_dec_ref_counter(v); v[0] = nnn; } while(0)
#define SET_NIL_RETURN()	do {                                    \
        mrbc_dec_ref_counter(v); v[0].tt = MRBC_TT_NIL; } while(0)
#define SET_FALSE_RETURN()	do {                                        \
        mrbc_dec_ref_counter(v); v[0].tt = MRBC_TT_FALSE; } while(0)
#define SET_TRUE_RETURN()	do {                                    \
        mrbc_dec_ref_counter(v); v[0].tt = MRBC_TT_TRUE; } while(0)
#define SET_BOOL_RETURN(n)	do {                                        \
        mrbc_dec_ref_counter(v); v[0].tt = (n)?MRBC_TT_TRUE:MRBC_TT_FALSE; } while(0)
#define SET_INT_RETURN(n)	do { mrbc_int nnn = (n);                    \
        mrbc_dec_ref_counter(v); v[0].tt = MRBC_TT_FIXNUM; v[0].i = nnn; } while(0)
#define SET_FLOAT_RETURN(n)	do { mrbc_float nnn = (n);                  \
        mrbc_dec_ref_counter(v); v[0].tt = MRBC_TT_FLOAT; v[0].d = nnn; } while(0)

#define GET_TT_ARG(n)		(v[(n)].tt)
#define GET_INT_ARG(n)		(v[(n)].i)
#define GET_ARY_ARG(n)		(v[(n)])
#define GET_ARG(n)		    (v[(n)])
#define GET_FLOAT_ARG(n)	(v[(n)].d)
#define GET_STRING_ARG(n)	(v[(n)].string->data)

#define mrbc_fixnum_value(n)	((mrbc_value){.tt = MRBC_TT_FIXNUM, .i=(n)})
#define mrbc_float_value(n)	    ((mrbc_value){.tt = MRBC_TT_FLOAT, .d=(n)})
#define mrbc_nil_value()	    ((mrbc_value){.tt = MRBC_TT_NIL})
#define mrbc_true_value()	    ((mrbc_value){.tt = MRBC_TT_TRUE})
#define mrbc_false_value()	    ((mrbc_value){.tt = MRBC_TT_FALSE})
#define mrbc_bool_value(n)	    ((mrbc_value){.tt = (n)?MRBC_TT_TRUE:MRBC_TT_FALSE})

// CUDA on-device specific implementation, to be optimized later
char* guru_alloc(size_t sz);

#ifdef __GURU_CUDA__
#define __GURU__ __device__
__GURU__ void    guru_memcpy(uint8_t *d, const uint8_t *s, size_t sz);
__GURU__ void    guru_memset(uint8_t *d, const uint8_t v,  size_t sz);
__GURU__ size_t  guru_memcmp(const uint8_t *d, const uint8_t *s, size_t sz);
__GURU__ long    guru_atol(const char *s);

__GURU__ size_t  guru_strlen(const char *s);
__GURU__ void    guru_strcpy(const char *s1, const char *s2);
__GURU__ size_t  guru_strcmp(const char *s1, const char *s2);
__GURU__ char   *guru_strchr(const char *s, const char c);
__GURU__ char   *guru_strcat(char *d, const char *s);
    
#define MEMCPY(d, s, sz)  guru_memcpy(d, s, sz)
#define MEMSET(d, v, sz)  guru_memset(d, v, sz)
#define MEMCMP(d, s, sz)  guru_memcmp(d, s, sz)
#define ATOL(s)           guru_atol(s)
#define STRLEN(s)		  guru_strlen(s)
#define STRCPY(s1, s2)	  guru_strcpy(s1, s2)
#define STRCMP(s1, s2)    guru_strcmp(s1, s2)
#define STRCHR(s, c)      guru_strchr(s, c)
#define STRCAT(d, s)      guru_strcat(d, s)
#else
#define __GURU__
#define MEMCPY(d, s, sz)  memcpy(d, s, sz)
#define MEMSET(d, v, sz)  memset(d, v, sz)
#define MEMCMP(d, s, sz)  memcmp(d, s, sz)
#define STRLEN(s)		  strlen(s)
#define STRCPY(s1, s2)	  strcpy(s1, s2)
#define STRCMP(s1, s2)    strcmp(s1, s2)
#define STRCHR(s, c)      strchr(s, c)
#define STRCAT(d, s)      strcat(d, s)
#endif

#ifdef __cplusplus
}
#endif
#endif
