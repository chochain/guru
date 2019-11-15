/*! @file
  @brief
  GURU Object Inspect (to_s) Factory

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include <assert.h>

#include "guru.h"
#include "class.h"
#include "value.h"
#include "symbol.h"
#include "object.h"
#include "inspect.h"

#include "c_fixnum.h"
#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

#include "puts.h"

#if !GURU_USE_STRING
__GURU__ void 	guru_na(const U8 *msg)		{}
__GURU__ void 	gv_to_s(GV v[], U32 vi)		{}
__GURU__ void	gv_join(GV v[], U32 vi)		{}

#else

__GURU__ void _to_s(GV *s, GV *v, U32 n);			// forward declaration

__GURU__ void
guru_na(const U8 *msg)
{
	PRINTF("method not supported: %s\n", msg);
}

//================================================================
//! Nil class
__GURU__ void
_nil(GV *s)
{
	// do nothing
}

//================================================================
//! False class
__GURU__ void
_false(GV *s)
{
    guru_str_add_cstr(s, "false");
}

//================================================================
//! True class
__GURU__ void
_true(GV *s)
{
    guru_str_add_cstr(s, "true");
}

//================================================================
//! Integer class
__GURU__ void
_int(GV *s, GV v[], U32 vi)
{
    U32 aoff = 'a' - 10;
    U32 base = vi ? v[1].i : 10;				// if base given

    assert(base >=2 && base <= 36);

    U8 buf[64+2];									// int64 + terminate + 1
    U8 *p = buf + sizeof(buf) - 1;		*p='\0';	// fill from the tail of the buffer
    GI i  = v[0].i;
	U8 sng = i<0 ? (i=-i, '-') : 0;
    do {
        U32 x = i % base;
        *--p = (x < 10)? x + '0' : x + aoff;
        i /= base;
    } while (i>0);
    if (sng) *--p = sng;

    guru_str_add_cstr(s, p);
}

//================================================================
/*! append c string (s0 += (U32A)ptr)

  @param  s0	pointer to target value 1
  @param  ptr	pointer
*/
__GURU__ void
_phex(GV *s, void *ptr)
{
	GV iv[2]; { iv[0].gt=GT_INT; iv[1].gt=GT_INT; }
	iv[1].i = 16;
	iv[0].i = (U32A)ptr;

	_int(s, iv, 1);
}

//================================================================
// Float class not implemented yet
__GURU__ void
_flt(GV *s, GV *v)
{
	guru_na("flt");
}

//================================================================
// Symbol class
__GURU__ void
_sym(GV *s, GV *v)
{
	guru_str_add_cstr(s, ":");
    guru_str_add_cstr(s, id2name(v->i));
}

//================================================================
//! Proc class
__GURU__ void
_prc(GV *s, GV *v)
{
	guru_str_add_cstr(s, "<#Proc:");
	_phex(s, v->proc);
	guru_str_add_cstr(s, ">");
}

//================================================================
//! String class
__GURU__ void
_str(GV *s, GV *v)
{
	guru_str_add_cstr(s, "\"");
	guru_str_add_cstr(s, (U8 *)v->str->raw);
	guru_str_add_cstr(s, "\"");
}


__GURU__ void
_cls(GV *s, GV *v)
{
	U8 *name = id2name(v->cls->sid);
	guru_str_add_cstr(s, name);
}

__GURU__ void
_obj(GV *s, GV *v)
{
	assert(v->gt==GT_OBJ);
	U8 *name = id2name(class_by_obj(v)->sid);
	guru_str_add_cstr(s, "#<");
	guru_str_add_cstr(s, name);
	guru_str_add_cstr(s, ":");
	_phex(s, v->self);
	guru_str_add_cstr(s, ">");
}

#if GURU_USE_ARRAY
//================================================================
//! Array class
__GURU__ void
_ary(GV *s, GV *v)
{
    U32 n  = v->array->n;
	GV  *o = v->array->data;
    for (U32 i=0; i < n; i++) {
    	guru_str_add_cstr(s, (const U8 *)(i==0 ? "[" : ", "));
    	_to_s(s, o++, 0);				// array element
	}
	guru_str_add_cstr(s, (const U8 *)(n==0 ? "[]" : "]"));
}

//================================================================
//! Hash class
__GURU__ void
_hsh(GV *s, GV *v)
{
    assert(v->gt==GT_HASH);

    int n  = v->array->n;
    GV  *o = v->array->data;
    for (U32 i=0; i<n; i+=2) {
    	guru_str_add_cstr(s, (const U8 *)(i==0 ? "{" : ", "));

    	_to_s(s, o++, 0);				// key
        guru_str_add_cstr(s, "=>");
        _to_s(s, o++, 0);				// value
    }
    guru_str_add_cstr(s, (const U8 *)(n==0 ? "{}" : "}"));
}

//================================================================
//! Range class
__GURU__ void
_rng(GV *s, GV *v)
{
    assert(v->gt==GT_RANGE);

    for (U32 i=0; i<2; i++) {
        guru_str_add_cstr(s, (const U8 *)(i==0 ? "" : ".."));
        GV o = (i==0) ? v->range->first : v->range->last;
        _to_s(s, &o, 0);
    }
}
#endif  // GURU_USE_ARRAY

__GURU__ void
_to_s(GV *s, GV v[], U32 n)
{
	switch (v->gt) {
    case GT_NIL:
    case GT_EMPTY:	_nil(s);		break;
    case GT_FALSE:	_false(s);		break;
    case GT_TRUE:	_true(s);		break;
    case GT_INT: 	_int(s, v, n);	break;
    case GT_FLOAT: 	_flt(s, v);		break;
    case GT_SYM:    _sym(s, v);		break;
    case GT_PROC:	_prc(s, v);		break;
    case GT_CLASS:	_cls(s, v);		break;
    case GT_OBJ:	_obj(s, v);		break;
    case GT_ARRAY:	_ary(s, v);		break;
    case GT_STR: 	_str(s, v);		break;
    case GT_HASH:  	_hsh(s, v);		break;
    case GT_RANGE:  _rng(s, v);		break;
    default: assert(1==0);		// unknown type
    }
}
//================================================================
//! Object#to_s factory function
#define BUF_SIZE	512
__GURU__ void
gv_to_s(GV v[], U32 vi)
{
	GV ret = guru_str_buf(BUF_SIZE);

	_to_s(&ret, v, vi);

	RETURN_VAL(ret);
}

__GURU__ void
gv_join(GV v[], U32 vi)
{
	assert(v->gt==GT_ARRAY);
	guru_array *a = v->array;

	GV ret = guru_str_buf(BUF_SIZE);
	GV *r  = a->data;
	for (U32 i=0; i<a->n; i++, r++) {
		if (r->gt!=GT_STR)	_to_s(&ret, r, 0);
		else guru_str_add_cstr(&ret, (U8 *)r->str->raw);
		if (vi==0 || (i+1)>=a->n) continue;
		guru_str_add_cstr(&ret, (U8 *)(v+1)->str->raw);
	}
	RETURN_VAL(ret);
}
#endif	// GURU_USE_STRING
