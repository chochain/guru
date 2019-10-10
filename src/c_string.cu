/*! @file
  @brief
  GURU String object

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <stdio.h>
#include <assert.h>

#include "mmu.h"		// includes guru.h
#include "static.h"
#include "value.h"
#include "symbol.h"

#include "c_string.h"
#include "c_range.h"
#include "inspect.h"

#if     GURU_USE_STRING
//================================================================
/*! white space character test

  @param  ch	character code.
  @return		result.
*/
__GURU__ bool
_is_space(U8 ch)
{
    static const char ws[] = " \t\r\n\f\v";	// '\0' on tail

    for (U32 i=0; i < sizeof(ws); i++) {
        if (ch==ws[i]) return true;
    }
    return false;
}

//================================================================
/*! get size
 */
__GURU__ __INLINE__ U32
_len(const GV *v)
{
    return v->str->n;
}

//================================================================
/*! get c-language string (U8*)
 */
__GURU__ __INLINE__ U8*
_raw(const GV *v)
{
    return (U8*)v->str->raw;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @param  len	source length
  @return 	string object
*/
__GURU__ GV
_blank(U32 len)
{
    GV  v; { v.gt=GT_STR; v.acl=ACL_HAS_REF; }		// assuming some one acquires it
    U32 asz = len+1;	ALIGN(asz);			// 8-byte aligned
    /*
      Allocate handle and string buffer.
    */
    guru_str *h = v.str = (guru_str *)guru_alloc(sizeof(guru_str));
    U8       *s = (U8*)guru_alloc(asz);		// 8-byte aligned

    assert(((U32A)h & 7)==0);
    assert(((U32A)s & 7)==0);

    s[0] = '\0';							// empty new string
    h->rc   = 1;
    h->size = asz;
    h->n    = len;
    h->raw  = (char *)s;					// TODO: for DEBUG, change back to (U8*)

    return v;
}

__GURU__ GV
_new(const U8 *src)
{
	U32 len = STRLENB(src);
	GV  ret = _blank(len);

    // deep copy source string
    if (src) MEMCPY(ret.str->raw, src, len+1);		// plus '\0'

    return ret;
}

//================================================================
/*! duplicate string

  @param  vm	pointer to VM.
  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @return	new string as s1 + s2
*/
__GURU__ GV
_dup(const GV *v0)
{
    guru_str *h0 = v0->str;

    GV v1 = _blank(h0->n);					// refc already set to 1

    MEMCPY(v1.str->raw, h0->raw, h0->n + 1);

    return v1;
}

//================================================================
/*! locate a substring in a string

  @param  src		pointer to target string
  @param  pattern	pointer to substring
  @param  offset	search offset
  @return		position index. or minus value if not found.
*/
__GURU__ S32
_index(const GV *v, const GV *pattern, U32 offset)
{
    U8  *p0 = _raw(v) + offset;
    U8  *p1 = _raw(pattern);
    U32 sz  = _len(pattern);
    U32 nz  = _len(v) - sz - offset;

    for (U32 i=0; nz>0 && i <= nz; i++, p0++) {
        if (MEMCMP(p0, p1, sz)==0) {
            return p1 - _raw(v);	// matched.
        }
    }
    return -1;
}

//================================================================
/*! remove the whitespace in myself

  @param  src	pointer to target value
  @param  mode	1:left-side, 2:right-side, 3:each
  @return	0 when not removed.
*/
__GURU__ U32
_strip(GV *v, U32 mode)
{
    U8  *p0 = _raw(v);
    U8  *p1 = p0 + _len(v) - 1;

    // left-side
    if (mode & 0x01) {
    	for (; p0 <= p1; p0++) {
            if (*p0=='\0') 		 break;
            if (!_is_space(*p0)) break;
        }
    }
    // right-side
    if (mode & 0x02) {
    	for (; p0 <= p1; p1--) {
            if (!_is_space(*p1)) break;
        }
    }
    U32 new_len = p1 - p0 + 1;
    if (_len(v)==new_len) return 0;

    U8 *buf = _raw(v);
    if (p0 != buf) {
    	MEMCPY(buf, p0, new_len);
    }
    buf[new_len] = '\0';
    U32 asz = new_len + 1; 	ALIGN(asz);				// 8-byte aligned

    v->str->size = asz;
    v->str->n    = new_len;
    v->str->raw  = (char *)guru_realloc(buf, asz);	// shrink suitable size.

    return 1;
}

//================================================================
/*! remove the CR,LF in myself

  @param  src	pointer to target value
  @return	0 when not removed.
*/
__GURU__ int
_chomp(GV *v)
{
    U8 *p0 = _raw(v);
    U8 *p1 = p0 + _len(v) - 1;

    if (*p1=='\n') p1--;
    if (*p1=='\r') p1--;

    U32 new_len = p1 - p0 + 1;
    if (_len(v)==new_len) return 0;

    U8 *buf = _raw(v);
    buf[new_len] = '\0';

    v->str->n = new_len;

    return 1;
}

//================================================================
/*! constructor by c string

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @return 	string object
*/
__GURU__ GV
guru_str_new(const U8 *src)			// cannot use U8P, need lots of casting
{
    return _new((U8*)src);
}

__GURU__ GV
guru_str_buf(U32 sz)				// a string buffer
{
	GV ret = _blank(sz);
	ret.str->n = 0;
	return ret;
}

__GURU__ GV
guru_str_clr(GV *s)
{
	assert(s->gt==GT_STR);
	s->str->n = 0;
	return *s;
}

//================================================================
/*! destructor

  @param  str	pointer to target value
*/
__GURU__ void
guru_str_del(GV *v)
{
    guru_free(v->str->raw);
    guru_free(v->str);
}

//================================================================
/*! add string (s2 = s0 + s1)
z
  @param  s0	pointer to target value 0
  @param  s1	pointer to target value 1
*/
__GURU__ GV
guru_str_add(GV *s0, GV *s1)
{
	assert(s1->gt==GT_STR);

    U32 len0 = s0->str->n;
    U32 len1 = s1->str->n;
    U32 asz  = len0 + len1 + 1;		ALIGN(asz);			// +'\0', 8-byte aligned

    GV  ret  = _blank(asz);
    U8  *buf = (U8*)ret.str->raw;
    MEMCPY(buf, 	 s0->str->raw, len0);
    MEMCPY(buf+len0, s1->str->raw, len1+1);

    ret.str->n = len0 + len1;

    return ret;
}

//================================================================
/*! append c string (s0 += s1)

  @param  s0	pointer to target value 1
  @param  s1	pointer to char (c_str)
*/
__GURU__ GV
guru_str_add_cstr(GV *s0, const U8 *str)
{
    U32 len0 = s0->str->n;
    U32 len1 = STRLENB(str);
    U32 asz  = len0 + len1 + 1;		asz += -asz & 7;	// 8-byte aligned
    U8  *buf = (U8*)guru_realloc(s0->str->raw, asz);

    MEMCPY(buf + len0, str, len1 + 1);

    s0->str->size = asz;
    s0->str->n 	  = len0 + len1;
    s0->str->raw  = (char *)buf;

    return *s0;
}

//================================================================
/*! (method) +
 */
__CFUNC__
str_add(GV v[], U32 vi)
{
    assert(v[1].gt == GT_STR);

    GV ret = guru_str_add(v, v+1);

    RETURN_VAL(ret);
}

//================================================================
/*! (method) *
 */
__CFUNC__
str_mul(GV v[], U32 vi)
{
	U32 sz = _len(v);

    if (v[1].gt != GT_INT) {
        PRINTF("TypeError\n");	// raise?
        return;
    }
    GV ret = _blank(sz * v[1].i);

    U8 *p = (U8*)ret.str->raw;
    for (U32 i = 0; i < v[1].i; i++) {
        MEMCPY(p, _raw(v), sz);
        p += sz;
    }
    *p = '\0';

    RETURN_VAL(ret);
}

//================================================================
/*! (method) size, length
 */
__CFUNC__
str_len(GV v[], U32 vi)
{
    GI len = _len(v);

    RETURN_INT(len);
}

//================================================================
/*! (method) to_i
 */
__CFUNC__
str_to_i(GV v[], U32 vi)
{
    U32 base = 10;
    if (vi) {
        base = v[1].i;
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    GI i = guru_atoi(_raw(v), base);

    RETURN_INT(i);
}

//================================================================
__CFUNC__
str_to_s(GV v[], U32 vi)
{
    GV ret = _dup(v);
    RETURN_VAL(ret);
}

#if     GURU_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__CFUNC__
str_to_f(GV v[], U32 vi)
{
    GF d = ATOF(_raw(v));

    RETURN_FLOAT(d);
}
#endif // GURU_USE_FLOAT

__GURU__ GV
_slice(GV *v, U32 i, U32 sz)
{
    GV  ret = _blank(sz);									//	'\0'
    U8  *d  = (U8*)ret.str->raw;

    MEMCPY(d, v->str->raw + i, sz);
    *(d+sz) = '\0';

    return ret;
}

//================================================================
/*! (method) []
 */
__CFUNC__
str_slice(GV v[], U32 vi)
{
    U32 n  = v->str->n;
    GV *v1 = &v[1];
    GV *v2 = &v[2];

    if (vi==1 && v1->gt==GT_INT) {							// slice(n) -> String | nil
        S32 i = v1->i;	i += (i < 0) ? n : 0;	i = (i< n) ? i : n;
        RETURN_VAL(i<n ? _slice(v, i, 1) : NIL());
    }
    else if (vi==2 && v1->gt==GT_INT && v2->gt==GT_INT) { 	// slice(n, len) -> String | nil
    	S32 i = v1->i; 	i += (i < 0) ? n : 0;
    	S32 sz = v2->i;	sz = (sz+i) < n ? sz : (n-i);
    	RETURN_VAL(sz > 1 ? _slice(v, i, sz) : NIL());
    }
    else if (vi==1 && v1->gt==GT_RANGE) {
    	guru_range *r = v1->range;
    	assert(r->first.gt==GT_INT && r->last.gt==GT_INT);
    	S32 i  = r->first.i;
    	S32 sz = r->last.i-i + (IS_INCLUDE(r) ? 1 : 0);
    	sz = (sz+i) < n ? sz : (n-i);
    	RETURN_VAL(sz > 1 ? _slice(v, i, sz) : NIL());
    }
    else {
    	PRINTF("Not support such case in String#[].\n");
    }
}

//================================================================
/*! (method) []=
 */
__CFUNC__
str_insert(GV v[], U32 vi)
{
    S32 nth;
    S32 len;
    GV *val;

    if (vi==2 &&								// self[n] = val
        v[1].gt==GT_INT &&
        v[2].gt==GT_STR) {
        nth = v[1].i;
        len = 1;
        val = &v[2];
    }
    else if (vi==3 &&							// self[n, len] = val
             v[1].gt==GT_INT &&
             v[2].gt==GT_INT &&
             v[3].gt==GT_STR) {
        nth = v[1].i;
        len = v[2].i;
        val = &v[3];
    }
    else {
        guru_na("case of str_insert");
        return;
    }

    U32 len1 = v->str->n;
    U32 len2 = val->str->n;
    if (nth < 0) nth = len1 + nth;              // adjust to positive number.
    if (len > len1 - nth) len = len1 - nth;
    if (nth < 0 || nth > len1 || len < 0) {
        PRINTF("IndexError\n");  // raise?
        return;
    }
    U32 asz  = len1 + len2 - len + 1;	asz += -asz & 7;			// 8-byte aligned
    U8  *str = (U8*)guru_realloc(_raw(v), asz);

    MEMCPY(str + nth + len2, str + nth + len, len1 - nth - len + 1);
    MEMCPY(str + nth, (U8*)_raw(val), len2);

    v->str->size = asz;
    v->str->n    = len1 + len2 - len;
    v->str->raw  = (char *)str;
}

//================================================================
/*! (method) chomp
 */
__CFUNC__
str_chomp(GV v[], U32 vi)
{
    GV ret = _dup(v);
    _chomp(&ret);
    RETURN_VAL(ret);
}

//================================================================
/*! (method) chomp!
 */
__CFUNC__
str_chomp_self(GV v[], U32 vi)
{
    if (_chomp(v)==0) {
        RETURN_NIL();
    }
}

//================================================================
/*! (method) dup
 */
__CFUNC__
str_dup(GV v[], U32 vi)
{
    RETURN_VAL(_dup(v));
}

//================================================================
/*! (method) index
 */
__CFUNC__
str_index(GV v[], U32 vi)
{
    S32 index;
    S32 offset;

    if (vi==1) {
        offset = 0;
    }
    else if (vi==2 && v[2].gt==GT_INT) {
        offset = v[2].i;
        if (offset < 0) offset += _len(v);
        if (offset < 0) RETURN_NIL();
    }
    else {
        RETURN_NIL();					// raise? ArgumentError
    }

    index = _index(v, v+1, offset);
    if (index < 0) RETURN_NIL();

    RETURN_INT(index);
}

//================================================================
/*! (method) ord
 */
__CFUNC__
str_ord(GV v[], U32 vi)
{
    RETURN_INT(_raw(v)[0]);
}

//================================================================
/*! (method) split
 */
__CFUNC__
str_split(GV v[], U32 vi)
{
    guru_na("string#split");
}

//================================================================
/*! (method) sprintf
 */
__CFUNC__
str_sprintf(GV v[], U32 vi)
{
	guru_na("string#sprintf");
}

//================================================================
/*! (method) printf
 */
__CFUNC__
str_printf(GV v[], U32 vi)
{
	guru_na("string#printf");
}

//================================================================
/*! (method) lstrip
 */
__CFUNC__
str_lstrip(GV v[], U32 vi)
{
    GV ret = _dup(v);

    _strip(&ret, 0x01);	// 1: left side only

    RETURN_VAL(ret);
}

//================================================================
/*! (method) lstrip!
 */
__CFUNC__
str_lstrip_self(GV v[], U32 vi)
{
    if (_strip(v, 0x01)==0) {	// 1: left side only
        RETURN_VAL(NIL());
    }
}

//================================================================
/*! (method) rstrip
 */
__CFUNC__
str_rstrip(GV v[], U32 vi)
{
    GV ret = _dup(v);

    _strip(&ret, 0x02);							// 2: right side only

    RETURN_VAL(ret);
}

//================================================================
/*! (method) rstrip!
 */
__CFUNC__
str_rstrip_self(GV v[], U32 vi)
{
    if (_strip(v, 0x02)==0) {				// 2: right side only
        RETURN_VAL(NIL());			// keep refc
    }
}

//================================================================
/*! (method) strip
 */
__CFUNC__
str_strip(GV v[], U32 vi)
{
    GV ret = _dup(v);
    _strip(&ret, 0x03);	// 3: left and right
    RETURN_VAL(ret);
}

//================================================================
/*! (method) strip!
 */
__CFUNC__
str_strip_self(GV v[], U32 vi)
{
    if (_strip(v, 0x03)==0) {		// 3: left and right
        RETURN_VAL(NIL());	// keep refc
    }
}

//================================================================
/*! (method) to_sym
 */
__CFUNC__
str_to_sym(GV v[], U32 vi)
{
    RETURN_VAL(guru_sym_new(_raw(v)));
}

//================================================================
//! Inspect
#define BUF_SIZE 80

__CFUNC__
str_inspect(GV v[], U32 vi)
{
	const char *hex = "0123456789ABCDEF";
    GV ret = guru_str_buf(BUF_SIZE*2);

    guru_str_add_cstr(&ret, "\"");

    U8 buf[BUF_SIZE];
    U8 *p = buf;
    U8 *s = (U8*)v->str->raw;

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
/*! initialize
 */
__GURU__ void
guru_init_class_string()
{
	static Vfunc vtbl[] = {
		{ "+",		str_add			},
		{ "*",		str_mul			},
		{ "size",	str_len			},
		{ "length",	str_len			},
		{ "<<",		str_add			},
		{ "[]",		str_slice		},
		{ "[]=",	str_insert		},
		// op
		{ "chomp",	str_chomp		},
		{ "chomp!",	str_chomp_self	},
		{ "dup",	str_dup			},
		{ "index",	str_index		},
		{ "ord",	str_ord			},
		{ "split",	str_split		},
		{ "lstrip",	str_lstrip		},
		{ "lstrip!",str_lstrip_self	},
		{ "rstrip",	str_rstrip		},
		{ "rstrip!",str_rstrip_self	},
		{ "strip",	str_strip		},
		{ "strip!",	str_strip_self	},
		{ "intern",	str_to_sym		},
		// conversion methods
		{ "to_i",	str_to_i		},
		{ "to_s",   str_to_s		},
		{ "to_sym",	str_to_sym		},
		{ "to_f",	str_to_f		},
		{ "inspect",str_inspect		}
	};
    guru_class_string = guru_add_class(
    	"String", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}

#endif // GURU_USE_STRING
