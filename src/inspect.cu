/*! @file
  @brief
  GURU Object Inspect (to_s) Factory

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.


  </pre>
*/
#include "guru.h"
#include "symbol.h"
#include "mmu.h"
#include "base.h"

#include "class.h"
#include "inspect.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

#if !GURU_USE_STRING
__CFUNC__	gr_to_s(GR r[], U32 ri)			{}
__CFUNC__ 	ary_join(GR r[], U32 ri)		{}
__CFUNC__	str_sprintf(GR r[], U32 ri)		{}
__CFUNC__	str_printf(GR r[], U32 ri)		{}

#else

__GURU__ void _to_s(GR *s, GR *r, U32 n);			// forward declaration

//================================================================
//! Nil class
__GURU__ void
_nil(GR *s)
{
	// do nothing
}

//================================================================
//! False class
__GURU__ void
_false(GR *buf)
{
    guru_buf_add_cstr(buf, "false");
}

//================================================================
//! True class
__GURU__ void
_true(GR *buf)
{
    guru_buf_add_cstr(buf, "true");
}

//================================================================
//! Integer class
__GURU__ void
_int(GR *buf, GR r[], U32 ri)
{
    U32 aoff = 'a' - 10;
    U32 base = ri ? r[1].i : 10;				// if base given

    ASSERT(base >=2 && base <= 36);

    U8 tmp[64+2];								// int64 + terminate + 1
    U8 *p = tmp + sizeof(tmp) - 1;	*p='\0';	// fill from the tail of the buffer
    GI i  = r[0].i;
	U8 sng = i<0 ? (i=-i, '-') : 0;
    do {
        U32 x = i % base;
        *--p = (x < 10)? x + '0' : x + aoff;
        i /= base;
    } while (i>0);
    if (sng) *--p = sng;

    guru_buf_add_cstr(buf, p);
}

//================================================================
/*! append c string (s0 += (U32A)ptr)

  @param  s0	pointer to target value 1
  @param  ptr	pointer
*/
__GURU__ void
_phex(GR *s, void *ptr)
{
	GR iv[2]; { iv[0].gt=GT_INT; iv[1].gt=GT_INT; }
	iv[1].i = 16;
	iv[0].i = (U32A)ptr;

	_int(s, iv, 1);
}

//================================================================
// Float class not implemented yet
__GURU__ void
_flt(GR *s, GR *r)
{
	NA("flt");
}

//================================================================
// Symbol class
__GURU__ void
_sym(GR *buf, GR *r)
{
	guru_buf_add_cstr(buf, ":");
    guru_buf_add_cstr(buf, id2name(r->i));
}

//================================================================
//! Proc class
__GURU__ void
_prc(GR *buf, GR *r)
{
	guru_buf_add_cstr(buf, "<#Proc:");
	_phex(buf, GR_PRC(r));
	guru_buf_add_cstr(buf, ">");
}

//================================================================
//! String class
__GURU__ void
_str(GR *buf, GR *r)
{
	guru_buf_add_cstr(buf, "\"");
	guru_buf_add_cstr(buf, (U8 *)MEMPTR(GR_STR(r)->raw));
	guru_buf_add_cstr(buf, "\"");
}


__GURU__ void
_cls(GR *buf, GR *r)
{
	U8 *name = id2name(GR_CLS(r)->sid);
	guru_buf_add_cstr(buf, name);
}

__GURU__ void
_obj(GR *buf, GR *r)
{
	ASSERT(r->gt==GT_OBJ);
	U8 *name = id2name(class_by_obj(r)->sid);
	guru_buf_add_cstr(buf, "#<");
	guru_buf_add_cstr(buf, name);
	guru_buf_add_cstr(buf, ":");
	_phex(buf, GR_OBJ(r));
	guru_buf_add_cstr(buf, ">");
}

#if GURU_USE_ARRAY
//================================================================
//! Array class
__GURU__ void
_ary(GR *buf, GR *r)
{
	guru_array *ary = GR_ARY(r);
    U32 n  = ary->n;
	GR  *o = ary->data;
    for (U32 i=0; i < n; i++) {
    	guru_buf_add_cstr(buf, (const U8 *)(i==0 ? "[" : ", "));
    	_to_s(buf, o++, 0);				// array element
	}
	guru_buf_add_cstr(buf, (const U8 *)(n==0 ? "[]" : "]"));
}

//================================================================
//! Hash class
__GURU__ void
_hsh(GR *buf, GR *r)
{
    ASSERT(r->gt==GT_HASH);

    guru_array *ary = GR_ARY(r);
    int n  = ary->n;
    GR  *o = ary->data;
    for (U32 i=0; i<n; i+=2) {
    	guru_buf_add_cstr(buf, (const U8 *)(i==0 ? "{" : ", "));

    	_to_s(buf, o++, 0);				// key
        guru_buf_add_cstr(buf, "=>");
        _to_s(buf, o++, 0);				// value
    }
    guru_buf_add_cstr(buf, (const U8 *)(n==0 ? "{}" : "}"));
}

//================================================================
//! Range class
__GURU__ void
_rng(GR *buf, GR *r)
{
    ASSERT(r->gt==GT_RANGE);

    guru_range *rng = GR_RNG(r);
    for (U32 i=0; i<2; i++) {
        guru_buf_add_cstr(buf, (const U8 *)(i==0 ? "" : ".."));
        GR o = (i==0) ? rng->first : rng->last;
        _to_s(buf, &o, 0);
    }
}
#endif  // GURU_USE_ARRAY

__GURU__ void
_to_s(GR *buf, GR r[], U32 n)
{
	switch (r->gt) {
    case GT_NIL:
    case GT_EMPTY:	_nil(buf);			break;
    case GT_FALSE:	_false(buf);		break;
    case GT_TRUE:	_true(buf);			break;
    case GT_INT: 	_int(buf, r, n);	break;
    case GT_FLOAT: 	_flt(buf, r);		break;
    case GT_SYM:    _sym(buf, r);		break;
    case GT_PROC:	_prc(buf, r);		break;
    case GT_CLASS:	_cls(buf, r);		break;
    case GT_OBJ:	_obj(buf, r);		break;
    case GT_ARRAY:	_ary(buf, r);		break;
    case GT_STR: 	_str(buf, r);		break;
    case GT_HASH:  	_hsh(buf, r);		break;
    case GT_RANGE:  _rng(buf, r);		break;
    default: ASSERT(1==0);		// unknown type
    }
}
//================================================================
//! Object#to_s factory function
#define BUF_SIZE	512
__CFUNC__
gr_to_s(GR r[], U32 ri)
{
	GR buf = guru_str_buf(BUF_SIZE);

	_to_s(&buf, r, ri);

	RETURN_VAL(buf);
}

__CFUNC__
int_chr(GR r[], U32 ri)
{
    U8 buf[2] = { (U8)r->i, '\0' };

    RETURN_VAL(guru_str_new(buf));
}

__CFUNC__
ary_join(GR r[], U32 ri)
{
	ASSERT(r->gt==GT_ARRAY);
	guru_array *a = GR_ARY(r);

	GR ret = guru_str_buf(BUF_SIZE);
	GR *d  = a->data;
	for (U32 i=0; i<a->n; i++, d++) {
		if (d->gt!=GT_STR)	_to_s(&ret, d, 0);
		else                guru_buf_add_cstr(&ret, (U8 *)MEMPTR(GR_STR(d)->raw));
		if (ri==0 || (i+1)>=a->n) continue;
		guru_buf_add_cstr(&ret, (U8 *)MEMPTR(GR_STR(r+1)->raw));
	}
	RETURN_VAL(ret);
}

//================================================================
/*! (method) sprintf
 */
__CFUNC__
gr_sprintf(GR r[], U32 ri)
{
	NA("string#sprintf");
}

//================================================================
/*! (method) printf
 */
__CFUNC__
gr_printf(GR r[], U32 ri)
{
	NA("string#printf");
}
#endif	// GURU_USE_STRING
