/*! @file
  @brief
  GURU Object classes i.e. Proc, Nil, False and True class and class specific functions.

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

__GURU__ void
nil_to_s(GV v[], U32 argc)
{
    RETURN_VAL(guru_str_new(NULL));
}

//================================================================
//! False class
__GURU__ void
false_to_s(GV v[], U32 argc)
{
    RETURN_VAL(guru_str_new("false"));
}

//================================================================
//! True class
__GURU__ void
true_to_s(GV v[], U32 argc)
{
    RETURN_VAL(guru_str_new("true"));
}

//================================================================
//! Integer class
__GURU__ void
int_chr(GV v[], U32 argc)
{
    U8 buf[2] = { (U8)v->i, '\0' };

    RETURN_VAL(guru_str_new(buf));
}

__GURU__ void
int_to_s(GV v[], U32 argc)
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

    RETURN_VAL(guru_str_new(p));
}

//================================================================
//! Proc class
__GURU__ void
prc_inspect(GV v[], U32 argc)
{
	GV ret = guru_str_new("<#Proc:");
	guru_str_add_cstr(&ret, guru_i2s((U64)v->proc, 16));

    RETURN_VAL(ret);
}

//================================================================
//! String class
#define BUF_SIZE 80

__GURU__ void
str_inspect(GV v[], U32 argc)
{
	const char *hex = "0123456789ABCDEF";
    GV ret  = guru_str_new("\"");

    U8 buf[BUF_SIZE];
    U8 *p = buf;
    U8 *s = (U8*)v->str->data;

    for (U32 i=0; i < v->str->n; i++, s++) {
        if (*s >= ' ' && *s < 0x80) {
        	*p++ = *s;
        }
        else {								// tiny isprint()
        	*p++ = '\\';
        	*p++ = 'x';
            *p++ = hex[*s >> 4];
            *p++ = hex[*s & 0x0f];
        }
    	if ((p-buf) > BUF_SIZE-5) {			// flush buffer
    		*p = '\0';
    		guru_str_add_cstr(&ret, buf);
    		p = buf;
    	}
    }
    *p++ = '\"';
    *p   = '\0';
    guru_str_add_cstr(&ret, buf);

    RETURN_VAL(ret);
}

//================================================================
//! Array class
__GURU__ void
ary_inspect(GV v[], U32 argc)
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

    RETURN_VAL(ret);
}

// internal function
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

__GURU__ void
ary_join(GV v[], U32 argc)
{
    GV ret = guru_str_new(NULL);
    GV sep = (argc==0)						// separator
    		? guru_str_new("")
    		: guru_inspect(v+argc, v+1);
    _join(v, argc, v, &ret, &sep);

    RETURN_VAL(ret);
}

//================================================================
//! Hash class
__GURU__ void
hsh_inspect(GV v[], U32 argc)
{
    GV blank = guru_str_new("");
    GV comma = guru_str_new(", ");
    GV ret   = guru_str_new("{");
    if (!ret.str) {
    	RETURN_NIL();
    }

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

    RETURN_VAL(ret);
}

//================================================================
//! Range class
__GURU__ void
rng_inspect(GV v[], U32 argc)
{
    GV ret = guru_str_new(NULL);
    if (!ret.str) {
        RETURN_NIL();
    }
    GV v1, s1;
    for (U32 i=0; i<2; i++) {
        if (i != 0) guru_str_add_cstr(&ret, (U8P)"..");
        v1 = (i == 0) ? v->range->first : v->range->last;
        s1 = guru_inspect(v+argc, &v1);

        guru_str_add(&ret, &s1);
        ref_clr(&s1);					// free locally allocated memory
    }
    RETURN_VAL(ret);
}


//================================================================
//! Object class
__GURU__ void
obj_to_s(GV v[], U32 argc)
{
	GV  ret;
	U8  *name;
	GV  iv[2] = { { .gt=GT_INT }, { .gt=GT_INT } };

    switch (v->gt) {
    case GT_CLASS:
    	name = id2name(v->cls->sid);
    	ret = guru_str_new(name);
    	break;
    case GT_OBJ:
    	iv[1].i = 16;
    	iv[0].i = (U32A)v->self;
    	int_to_s(iv, 1);

    	name = id2name(v->self->cls->sid);
    	ret  = guru_str_new("#<");
    	guru_str_add_cstr(&ret, name);
    	guru_str_add_cstr(&ret, ":0x");
    	guru_str_add_cstr(&ret, (U8*)iv[0].str->data);
    	guru_str_add_cstr(&ret, ">");

    	ref_clr(&iv[0]);
    	break;
    default:
    	ret = guru_str_new("");
    	break;
    }
    RETURN_VAL(ret);
}

