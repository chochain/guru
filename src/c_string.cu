/*! @file
  @brief
  GURU String object

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "guru.h"
#include "util.h"
#include "symbol.h"
#include "mmu.h"

#include "base.h"
#include "static.h"

#include "c_range.h"
#include "c_string.h"

#define STRBUF_SIZE		255

#if !GURU_USE_STRING
__GURU__ GR		guru_str_new(const U8 *src) { return NIL(); }			// cannot use U8P, need lots of casting
__GURU__ void	guru_str_del(GR *r)			{}
__GURU__ S32	guru_str_cmp(GR *s0, GR *s1){ return NIL(); }
__GURU__ GR		guru_str_buf(U32 sz)		{ return NIL(); }			// a string buffer
__GURU__ GR		guru_str_clr(GR *s)			{ return NIL(); }
__GURU__ GR     guru_str_add(GR *s0, GR *s1){ return NIL(); }
__GURU__ GR		guru_buf_add_cstr(GR *buf, const U8 *str) { return NIL(); }

__GURU__ void	guru_init_class_string() { guru_class_string = NULL; }

#else
//================================================================
/*! white space character test

  @param  ch	character code.
  @return		result.
*/
__GURU__ bool
_is_space(U8 ch)
{
    static const char ws[] = " \t\r\n\f\v";	// '\0' on tail

    for (int i=0; i < sizeof(ws); i++) {
        if (ch==ws[i]) return true;
    }
    return false;
}

//================================================================
/*! get size
 */
__GURU__ __INLINE__ U32
_sz(const GR *r)
{
    return GR_STR(r)->sz;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @param  len	source length
  @return 	string object
*/
__GURU__ GR
_blank(U32 sz)
{
    U32 bsz = ALIGN8(sz+1);						// add '\0' and 8-byte aligned
    /*
      Allocate handle and string buffer.
    */
    guru_str *h = (guru_str *)guru_alloc(sizeof(guru_str));
    U8       *s = (U8*)guru_alloc(bsz);			// 8-byte aligned

#if GURU_DEBUG
    ASSERT(((U32A)h & 7)==0);					// check alignment
    ASSERT(((U32A)s & 7)==0);
#endif // GURU_DEBUG

    s[0]   = '\0';								// empty new string
    h->rc  = 1;
    h->sz  = sz;
    h->bsz = bsz;
    h->raw = MEMOFF(s);							// TODO: for DEBUG, change back to (U8*)

    GR r { GT_STR, ACL_HAS_REF, 0, MEMOFF(h) };	// assuming some one acquires it

    return r;
}

__GURU__ GR
_new(const U8 *src)
{
	U32 sz  = STRLENB(src);
	GR  ret = _blank(sz);

    // deep copy source string
	MEMCPY(GR_RAW(&ret), src, sz+1);		// plus '\0'

	return ret;
}

//================================================================
/*! duplicate string

  @param  vm	pointer to VM.
  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @return	new string as s1 + s2
*/
__GURU__ GR
_dup(const GR *r0)
{
	U32 sz = _sz(r0);
    GR r1  = _blank(sz);					// refc already set to 1

    MEMCPY(GR_RAW(&r1), GR_RAW(r0), sz+1);	// plus '\0'

    return r1;
}

//================================================================
/*! locate a substring in a string

  @param  src		pointer to target string
  @param  pattern	pointer to substring
  @param  offset	search offset
  @return		position index. or minus value if not found.
*/
__GURU__ S32
_index(const GR *r, const GR *pattern, U32 offset)
{
    U8  *p0 = GR_RAW(r) + offset;
    U8  *p1 = GR_RAW(pattern);
    U32 sz  = _sz(pattern);
    U32 nz  = _sz(r) - sz - offset;

    for (int i=0; nz>0 && i <= nz; i++, p0++) {
        if (MEMCMP(p0, p1, sz)==0) {
            return p1 - GR_RAW(r);			// matched.
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
_strip(GR *r, U32 mode)
{
	U8  sz0 = _sz(r);
	U8  *p0 = GR_RAW(r), *raw = p0;					// head of string
    U8  *p1 = p0 + sz0 - 1;							// tail of string

    for (; mode&0x01 && p0 <= p1; p0++) {			// left-side
    	if (*p0=='\0') 		 break;
    	if (!_is_space(*p0)) break;
    }
    for (; mode&0x02 && p0 <= p1; p1--) {			// right-side
    	if (!_is_space(*p1)) break;
    }
    U32 new_sz = p1 - p0 + 1;
    if (new_sz==sz0) return 0;

    if (p0 != raw) {
    	MEMCPY(raw, p0, new_sz);					// hopefully no overstepping
    }
    raw[new_sz] = '\0';

    GR_STR(r)->sz = new_sz;

    return 1;
}

//================================================================
/*! remove the CR,LF in myself

  @param  src	pointer to target value
  @return	0 when not removed.
*/
__GURU__ int
_chomp(GR *r)
{
	U32 sz0 = _sz(r);
    U8  *p0 = GR_RAW(r);
    U8  *p1 = p0 + sz0 - 1;

    if (*p1=='\n') p1--;
    if (*p1=='\r') p1--;

    U32 new_sz = p1 - p0 + 1;
    if (new_sz==sz0) return 0;

    U8 *raw = GR_RAW(r);
    raw[new_sz] = '\0';

    GR_STR(r)->sz = new_sz;

    return 1;
}

//================================================================
/*! constructor by c string

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @return 	string object
*/
__GURU__ void
guru_str_transcode(GR *r, guru_str *h)			// cannot use U8P, need lots of casting
{
    U8 *raw = U8PADD(r, r->off);

/*
	//  Allocate handle and to point to ROM string
    guru_str *h = (guru_str *)guru_alloc(sizeof(guru_str));
*/
    ASSERT(((U32A)h & 7)==0);		// check alignment

    h->rc  = 0;
    h->sz  = STRLENB(raw);
    h->bsz = ALIGN4(h->sz+1);
    h->raw = MEMOFF(raw);

    r->off = MEMOFF(h);				// overwrite GR
}

__GURU__ GR
guru_str_new(const U8 *src)			// cannot use U8P, need lots of casting
{
    return _new((U8*)src);
}

__GURU__ GR
guru_str_buf(U32 sz)				// a string buffer
{
	GR ret = _blank(sz);
	GR_STR(&ret)->sz = 0;

	return ret;
}

__GURU__ GR
guru_str_clr(GR *s)
{
	ASSERT(s->gt==GT_STR);
	GR_STR(s)->sz = 0;

	return *s;
}

//================================================================
/*! destructor

  @param  str	pointer to target value
*/
__GURU__ void
guru_str_del(GR *r)
{
    guru_free(GR_RAW(r));
    guru_free(GR_STR(r));
}

//================================================================
/*! compare
 */
__GURU__ S32
guru_str_cmp(const GR *s0, const GR *s1)
{
	S32 x  = (U32)_sz(s0) - (U32)_sz(s1);
	if (x) return x;

	return STRCMP(GR_RAW(s0), GR_RAW(s1));
}

//================================================================
/*! add string (s2 = s0 + s1)
z
  @param  s0	pointer to target value 0
  @param  s1	pointer to target value 1
*/
__GURU__ GR
guru_str_add(GR *s0, GR *s1)
{
	ASSERT(s1->gt==GT_STR);

	GR buf = *s0;
	if (STRBUF_SIZE > GR_STR(s0)->bsz) {
		buf = guru_str_buf(STRBUF_SIZE);		// auto plus 1 for '\0'
		guru_buf_add_cstr(&buf, GR_RAW(s0));
	}
	else {
		ref_inc(s0);
	}
	guru_buf_add_cstr(&buf, GR_RAW(s1));		// ref counts increased as _dup updated

    return buf;
}

//================================================================
/*! append c string (s0 += s1)

  @param  s0	pointer to target value 1
  @param  s1	pointer to char (c_str)
*/
__GURU__ GR
guru_buf_add_cstr(GR *buf, const U8 *str)
{
    guru_str *sb  = GR_STR(buf);
    U8       *tmp = GR_RAW(buf);

    U32 sz0    = _sz(buf);
    U32 sz1    = STRLENB(str);
    U32 new_sz = sz0 + sz1;
    U32 bsz    = ALIGN8(new_sz + 1);			// '\0' and 8-byte aligned

    if (bsz > sb->bsz) {
    	tmp = (U8*)guru_realloc(tmp, bsz);
    	sb->bsz = bsz;
    	sb->raw = MEMOFF(tmp);
    }
    MEMCPY(tmp + sz0, str, sz1+1);
    sb->sz  = new_sz;

    return *buf;
}

//================================================================
/*! (method) +
 */
__CFUNC__
str_add(GR r[], U32 ri)
{
    GR ret = guru_str_add(r, r+1);				// will check (r+1)->gt==GT_STR

    RETURN_VAL(ret);
}

//================================================================
/*! (method) *
 */
__CFUNC__
str_mul(GR r[], U32 ri)
{
	U32 sz = _sz(r);

    if (r[1].gt != GT_INT) {
        PRINTF("TypeError\n");	// raise?
        return;
    }
    GR ret = _blank(sz * r[1].i);

    U8 *p = GR_RAW(&ret);
    for (int i = 0; i < r[1].i; i++) {
        MEMCPY(p, GR_RAW(r), sz);
        p += sz;
    }
    *p = '\0';

    RETURN_VAL(ret);
}

//================================================================
/*! (method) size, length
 */
__CFUNC__
str_len(GR r[], U32 ri)
{
    GI len = STRLEN(GR_RAW(r));

    RETURN_INT(len);
}

//================================================================
/*! (method) to_i
 */
__CFUNC__
str_to_i(GR r[], U32 ri)
{
    U32 base = 10;
    if (ri) {
        base = r[1].i;
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    GI i = ATOI(GR_RAW(r), base);

    RETURN_INT(i);
}

//================================================================
__CFUNC__
str_to_s(GR r[], U32 ri)
{
    GR ret = _dup(r);
    RETURN_VAL(ret);
}

#if     GURU_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__CFUNC__
str_to_f(GR r[], U32 ri)
{
    GF d = ATOF(GR_RAW(r));

    RETURN_FLOAT(d);
}
#endif // GURU_USE_FLOAT

__GURU__ GR
_slice(GR *r, U32 i, U32 n)
{
	U8  *s0  = (U8*)STRCUT(GR_RAW(r), i);		// start
	U8  *s1	 = (U8*)STRCUT(s0, n);				// end
	U32 asz  = U8POFF(s1, s0);
    GR  ret  = _blank(asz);						//	pad '\0' automatically
    U8  *raw = GR_RAW(&ret);

    MEMCPY(raw, s0, asz);
    *(raw+asz) = '\0';

    return ret;
}

//================================================================
/*! (method) []
 */
__CFUNC__
str_slice(GR r[], U32 ri)
{
    U32 n  = _sz(r);
    GR *r1 = &r[1];
    GR *r2 = &r[2];

    if (ri==1 && r1->gt==GT_INT) {							// slice(n) -> String | nil
        S32 i = r1->i;	i += (i < 0) ? n : 0;	i = (i< n) ? i : n;
        RETURN_VAL(i<n ? _slice(r, i, 1) : NIL);
    }
    else if (ri==2 && r1->gt==GT_INT && r2->gt==GT_INT) { 	// slice(n, len) -> String | nil
    	S32 i = r1->i; 	i += (i < 0) ? n : 0;
    	S32 sz = r2->i;	sz = (sz+i) < n ? sz : (n-i);
    	RETURN_VAL(sz > 1 ? _slice(r, i, sz) : NIL);
    }
    else if (ri==1 && r1->gt==GT_RANGE) {
    	guru_range *g = GR_RNG(r1);
    	ASSERT(g->first.gt==GT_INT && g->last.gt==GT_INT);
    	S32 i  = g->first.i;
    	S32 sz = g->last.i-i + (IS_INCLUDE(g) ? 1 : 0);
    	sz = (sz+i) < n ? sz : (n-i);
    	RETURN_VAL(sz > 1 ? _slice(r, i, sz) : NIL);
    }
    else {
    	PRINTF("Not support such case in String#[].\n");
    }
}

//================================================================
/*! (method) []=
 */
__CFUNC__
str_insert(GR r[], U32 ri)
{
    S32 nth;
    S32 len;
    GR *val;

    if (ri==2 &&								// self[n] = val
        r[1].gt==GT_INT &&
        r[2].gt==GT_STR) {
        nth = r[1].i;
        len = 1;
        val = &r[2];
    }
    else if (ri==3 &&							// self[n, len] = val
             r[1].gt==GT_INT &&
             r[2].gt==GT_INT &&
             r[3].gt==GT_STR) {
        nth = r[1].i;
        len = r[2].i;
        val = &r[3];
    }
    else {
        NA("case of str_insert");
        return;
    }

    U32 sz0 = _sz(r);
    U32 sz1 = _sz(val);
    if (nth < 0) nth = sz0 + nth;          		    // adjust to positive number.
    if (len > sz0 - nth) len = sz0 - nth;
    if (nth < 0 || nth > sz0 || len < 0) {
        PRINTF("IndexError\n");  // raise?
        return;
    }
    U32 new_sz = sz0 + sz1 - len;
    U32 bsz    = ALIGN8(new_sz + 1);				// '\0' and 8-byte aligned
    U8  *tmp = (U8*)guru_realloc(GR_RAW(r), bsz);

    MEMCPY(tmp + nth + sz1, tmp + nth + len, sz0 - nth - len + 1);
    MEMCPY(tmp + nth, GR_RAW(val), sz1);

    guru_str *s0 = GR_STR(r);
    s0->sz  = new_sz;
    s0->bsz = bsz;
    s0->raw = MEMOFF(tmp);
}

//================================================================
/*! (method) chomp
 */
__CFUNC__
str_chomp(GR r[], U32 ri)
{
    GR ret = _dup(r);
    _chomp(&ret);
    RETURN_VAL(ret);
}

//================================================================
/*! (method) chomp!
 */
__CFUNC__
str_chomp_self(GR r[], U32 ri)
{
    if (_chomp(r)==0) {
        RETURN_NIL();
    }
}

//================================================================
/*! (method) dup
 */
__CFUNC__
str_dup(GR r[], U32 ri)
{
    RETURN_VAL(_dup(r));
}

//================================================================
/*! (method) index
 */
__CFUNC__
str_index(GR r[], U32 ri)
{
    S32 index;
    S32 offset;

    if (ri==1) {
        offset = 0;
    }
    else if (ri==2 && r[2].gt==GT_INT) {
        offset = r[2].i;
        if (offset < 0) offset += _sz(r);
        if (offset < 0) RETURN_NIL();
    }
    else {
        RETURN_NIL();						// raise? ArgumentError
    }

    index = _index(r, r+1, offset);
    if (index < 0) RETURN_NIL();

    RETURN_INT(index);
}

//================================================================
/*! (method) include?
 */
__CFUNC__
str_include(GR r[], U32 ri)
{
    if (_index(r, r+1, 0)<0) RETURN_FALSE()
    else 					 RETURN_TRUE();
}

//================================================================
/*! (method) ord
 */
__CFUNC__
str_ord(GR r[], U32 ri)
{
    RETURN_INT(GR_RAW(r)[0]);
}

//================================================================
/*! (method) split
 */
__CFUNC__
str_split(GR r[], U32 ri)
{
    NA("string#split");
}

//================================================================
/*! (method) lstrip
 */
__CFUNC__
str_lstrip(GR r[], U32 ri)
{
    GR ret = _dup(r);

    _strip(&ret, 0x01);					// 1: left side only

    RETURN_VAL(ret);
}

//================================================================
/*! (method) lstrip!
 */
__CFUNC__
str_lstrip_self(GR r[], U32 ri)
{
    if (_strip(r, 0x01)==0) {			// 1: left side only
        RETURN_VAL(NIL);
    }
}

//================================================================
/*! (method) rstrip
 */
__CFUNC__
str_rstrip(GR r[], U32 ri)
{
    GR ret = _dup(r);

    _strip(&ret, 0x02);					// 2: right side only

    RETURN_VAL(ret);
}

//================================================================
/*! (method) rstrip!
 */
__CFUNC__
str_rstrip_self(GR r[], U32 ri)
{
    if (_strip(r, 0x02)==0) {			// 2: right side only
        RETURN_VAL(NIL);				// keep refc
    }
}

//================================================================
/*! (method) strip
 */
__CFUNC__
str_strip(GR r[], U32 ri)
{
    GR ret = _dup(r);
    _strip(&ret, 0x03);					// 3: left and right
    RETURN_VAL(ret);
}

//================================================================
/*! (method) strip!
 */
__CFUNC__
str_strip_self(GR r[], U32 ri)
{
    if (_strip(r, 0x03)==0) {			// 3: left and right
        RETURN_VAL(NIL);				// keep refc
    }
}

//================================================================
/*! (method) to_sym
 */
__CFUNC__
str_to_sym(GR r[], U32 ri)
{
	U8 *str = GR_RAW(r);
    GR ret { GT_SYM, 0, 0, { guru_rom_add_sym((char*)str) } };

    RETURN_VAL(ret);
}

//================================================================
//! Inspect
#define BUF_SIZE 79

__CFUNC__
str_inspect(GR r[], U32 ri)
{
	const char *hex = "0123456789ABCDEF";
    GR buf = guru_str_buf(BUF_SIZE*2);

    guru_buf_add_cstr(&buf, "\"");

    U8 tmp[BUF_SIZE];
    U8 *p = tmp;
    U8 *s = GR_RAW(r);

    for (int i=0; i < _sz(r); i++, s++) {
        if (*s >= ' ' && *s < 0x80) {
        	*p++ = *s;
        }
        else {								// tiny isprint()
        	*p++ = '\\';
        	*p++ = 'x';
            *p++ = hex[*s >> 4];
            *p++ = hex[*s & 0x0f];
        }
    	if ((p-tmp) > BUF_SIZE-5) {			// flush buffer
    		*p = '\0';
    		guru_buf_add_cstr(&buf, tmp);
    		p = tmp;
    	}
    }
    *p++ = '\"';
    *p   = '\0';
    guru_buf_add_cstr(&buf, tmp);

    RETURN_VAL(buf);
}

//================================================================
/*! initialize
 */
__GURU__ __const__ Vfunc str_vtbl[] = {
	{ "+",			str_add			},
	{ "*",			str_mul			},
	{ "size",		str_len			},
	{ "length",		str_len			},
	{ "<<",			str_add			},
	{ "[]",			str_slice		},
	{ "[]=",		str_insert		},
	// op
	{ "chomp",		str_chomp		},
	{ "chomp!",		str_chomp_self	},
	{ "dup",		str_dup			},
	{ "index",		str_index		},
	{ "include?", 	str_include   	},
	{ "ord",		str_ord			},
	{ "split",		str_split		},
	{ "lstrip",		str_lstrip		},
	{ "lstrip!",	str_lstrip_self	},
	{ "rstrip",		str_rstrip		},
	{ "rstrip!",	str_rstrip_self	},
	{ "strip",		str_strip		},
	{ "strip!",		str_strip_self	},
	{ "intern",		str_to_sym		},
	// conversion methods
	{ "to_i",		str_to_i		},
	{ "to_s",   	str_to_s		},
	{ "to_sym",		str_to_sym		},
	{ "to_f",		str_to_f		},
	{ "inspect",	str_inspect		}
};

__GURU__ void
guru_init_class_string()
{
    guru_rom_add_class(GT_STR, "String", GT_OBJ, str_vtbl, VFSZ(str_vtbl));
    guru_register_func(GT_STR, (guru_init_func)guru_str_new, guru_str_del, guru_str_cmp);
}

#endif // GURU_USE_STRING
