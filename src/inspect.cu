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
__GURU__ void guru_na(const U8 *msg)		{}
__GURU__ void gv_to_s(GV v[], U32 argc)		{}

__GURU__ void int_chr(GV v[], U32 argc)		{}
__GURU__ void nil_inspect(GV v[], U32 argc)	{}
__GURU__ void sym_all(GV v[], U32 argc)     {}
__GURU__ void sym_inspect(GV v[], U32 argc)	{}

#else

__GURU__ void
guru_na(const U8 *msg)
{
	PRINTF("method not supported: %s\n", msg);
}

//================================================================
//! Nil class
__GURU__ void
nil_inspect(GV v[], U32 argc)
{
    RETURN_VAL(guru_str_new("nil"));
}

__GURU__ GV
_nil(GV v[], U32 argc)
{
    return guru_str_new(NULL);
}

//================================================================
//! False class
__GURU__ GV
_false(GV v[], U32 argc)
{
    return guru_str_new("false");
}

//================================================================
//! True class
__GURU__ GV
_true(GV v[], U32 argc)
{
    return guru_str_new("true");
}

//================================================================
//! Integer class
__GURU__ void
int_chr(GV v[], U32 argc)
{
    U8 buf[2] = { (U8)v->i, '\0' };

    RETURN_VAL(guru_str_new(buf));
}

__GURU__ GV
_int(GV v[], U32 argc)
{
    U32 aoff = 'a' - 10;
    U32 base = argc ? v[1].i : 10;				// if base given

    assert(base >=2 && base <= 36);

    U8 buf[64+2];									// int64 + terminate + 1
    U8 *p = buf + sizeof(buf) - 1;		*p='\0';	// fill from the tail of the buffer
	S32 i = v[0].i;
    do {
        U32 x = i % base;
        *--p = (x < 10)? x + '0' : x + aoff;
        i /= base;
    } while (i>0);

    return guru_str_new(p);
}

//================================================================
// Float class not implemented yet
__GURU__ GV
_flt(GV v[], U32 argc)
{
	GV ret { .gt=GT_FLOAT };

	return ret;
}

//================================================================
// Symbol class
__GURU__ void
sym_inspect(GV v[], U32 argc)
{
    GV ret = guru_str_new(":");

    guru_str_add_cstr(&ret, id2name(v[0].i));

    RETURN_VAL(ret);
}

__GURU__ GV
_sym(GV v[], U32 argc)
{
    return guru_str_new(id2name(v[0].i));
}

//================================================================
//! Proc class
__GURU__ GV
prc_inspect(GV v[], U32 argc)
{
	GV  ret = guru_str_new("<#Proc:");
	guru_str_add_cstr(&ret, guru_i2s((U64)v->proc, 16));

    return ret;
}

#if GURU_USE_ARRAY
//================================================================
//! Array class
__GURU__ void
_join(GV v[], U32 argc, GV *src, GV *ret, GV *sep)
{
	guru_array *h = src->array;
    if (h->n==0) return;

    U32 i = 0;
    GV  s1;
    while (1) {
        if (h->data[i].gt==GT_ARRAY) {		// recursive
            _join(v, argc, &h->data[i], ret, sep);
        }
        else {
            s1 = guru_inspect(v+argc, &h->data[i]);
            guru_str_add(ret, &s1);
        }
        if (++i >= h->n) break;				// normal return.
        guru_str_add(ret, sep);
    }
}

__GURU__ GV
_ary(GV v[], U32 argc)
{
	GV ret = guru_str_new("[");
    GV vi, s1;
    for (U32 i=0, n=v->array->n; i < n; i++) {
        if (i != 0) guru_str_add_cstr(&ret, ", ");
        vi = v->array->data[i];
        s1 = guru_inspect(v+argc, &vi);
        guru_str_add(&ret, &s1);
    }
    guru_str_add_cstr(&ret, "]");

    return ret;
}

//================================================================
//! Hash class
__GURU__ GV
_hsh(GV v[], U32 argc)
{
    GV blank = guru_str_new("");
    GV comma = guru_str_new(", ");
    GV ret   = guru_str_new("{");

    assert(ret.str);

    GV  s[3];
    GV  *p = v->array->data;
    int n  = v->array->n;
    for (U32 i=0; i<n; i++, p+=2) {
    	s[0] = (i==0) ? blank : comma;
        s[1] = guru_inspect(v+argc, p);			// key
        s[2] = guru_inspect(v+argc, p+1);		// value

        guru_str_add(&ret, &s[0]);
        guru_str_add(&ret, &s[1]);
        guru_str_add_cstr(&ret, "=>");
        guru_str_add(&ret, &s[2]);

        ref_clr(&s[1]);							// free locally allocated memory
        ref_clr(&s[2]);
    }
    guru_str_add_cstr(&ret, "}");

    return ret;
}

//================================================================
//! Range class
__GURU__ GV
_rng(GV v[], U32 argc)
{
    GV ret = guru_str_new(NULL);

    assert(ret.str);

    GV v1, s1;
    for (U32 i=0; i<2; i++) {
        if (i != 0) guru_str_add_cstr(&ret, (U8P)"..");
        v1 = (i == 0) ? v->range->first : v->range->last;
        s1 = guru_inspect(v+argc, &v1);

        guru_str_add(&ret, &s1);
        ref_clr(&s1);					// free locally allocated memory
    }
    return ret;
}
#endif  // GURU_USE_ARRAY

__GURU__ GV
_class(GV v[], U32 argc)
{
	U8 *name = id2name(v->cls->sid);
	return guru_str_new(name);
}

__GURU__ GV
_obj(GV v[], U32 argc)
{
	GV  iv[2] = { { .gt=GT_INT }, { .gt=GT_INT } };

	iv[1].i = 16;
	iv[0].i = (U32A)v->self;
	GV s = _int(iv, 1);

	U8 *name = id2name(v->self->cls->sid);
	GV ret   = guru_str_new("#<");
	guru_str_add_cstr(&ret, name);
	guru_str_add_cstr(&ret, ":0x");
	guru_str_add_cstr(&ret, (U8*)s.str->data);
	guru_str_add_cstr(&ret, ">");

	ref_clr(&iv[0]);

	return ret;
}

//================================================================
//! Object#to_s factory function
__GURU__ void
gv_to_s(GV v[], U32 argc)
{
	GV  ret;

	switch (v->gt) {
    case GT_NIL:
    case GT_EMPTY:	ret = _nil(v, argc);	break;
    case GT_FALSE:	ret = _false(v, argc);	break;
    case GT_TRUE:	ret = _true(v, argc);	break;
    case GT_INT: 	ret = _int(v, argc);	break;
    case GT_FLOAT: 	ret = _flt(v, argc);	break;
    case GT_CLASS:	ret = _class(v, argc);	break;
    case GT_OBJ:	ret = _obj(v, argc);	break;
    case GT_ARRAY:	ret = _ary(v, argc);	break;
    case GT_HASH:  	ret = _hsh(v, argc);	break;
    case GT_STR: 	assert(1==0);			break;	// in c_string itself
    default: 		ret = guru_str_new("");	break;
    }
    RETURN_VAL(ret);
}
#endif	// GURU_USE_STRING
