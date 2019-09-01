/*! @file
  @brief
  mruby/c String object

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <stdio.h>
#include <assert.h>

#include "value.h"
#include "alloc.h"
#include "static.h"
#include "symbol.h"
#include "c_string.h"

#include "puts.h"

#if GURU_USE_STRING
//================================================================
/*! white space character test

  @param  ch	character code.
  @return	result.
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
_size(const GV *v)
{
    return v->str->len;
}

//================================================================
/*! get c-language string (U8P)
 */
__GURU__ __INLINE__ U8P
_data(const GV *v)
{
    return (U8P)v->str->data;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @param  len	source length
  @return 	string object
*/
__GURU__ GV
_new(const U8P src, U32 len)
{
    GV ret = {.gt = GT_STR};
    /*
      Allocate handle and string buffer.
    */
    guru_str *h = (guru_str *)guru_alloc(sizeof(guru_str));

    assert(h!=NULL);			// out of memory
#if GURU_64BIT_ALIGN_REQUIERD
    assert(((U32A)h & 7)==0);
#endif

    U8P s = (U8P)guru_alloc(len);
    if (s==NULL) {					// ENOMEM
        guru_free(h);
        return ret;
    }
#if GURU_64BIT_ALIGN_REQUIRED
    assert(((U32A)s & 7)==0);
#endif

    // deep copy source string
    if (src==NULL) 	s[0] = '\0';
    else 			MEMCPY(s, src, len+1);		// plus '\0'

    h->refc = 1;
    h->gt   = GT_STR;	// TODO: for DEBUG
    h->len  = len;
    h->data = s;

    ret.str = h;

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

    GV v1 = _new(NULL, h0->len);		// refc already set to 1
    if (v1.str==NULL) return v1;				// ENOMEM

    MEMCPY(v1.str->data, h0->data, h0->len + 1);

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
    U8P p0 = _data(v) + offset;
    U8P p1 = _data(pattern);
    S32 try_cnt = _size(v) - _size(pattern) - offset;

    while (try_cnt >= 0) {
        if (MEMCMP(p0, p1, _size(pattern))==0) {
            return p1 - _data(v);	// matched.
        }
        try_cnt--;
        p0++;
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
    U8P p0 = _data(v);
    U8P p1 = p0 + _size(v) - 1;

    // left-side
    if (mode & 0x01) {
        while (p0 <= p1) {
            if (*p0=='\0') break;
            if (!_is_space(*p0)) break;
            p0++;
        }
    }
    // right-side
    if (mode & 0x02) {
        while (p0 <= p1) {
            if (!_is_space(*p1)) break;
            p1--;
        }
    }
    U32 new_len = p1 - p0 + 1;
    if (_size(v)==new_len) return 0;

    U8P buf = _data(v);
    if (p0 != buf) MEMCPY(buf, p0, new_len);
    buf[new_len] = '\0';

    v->str->data = (U8P)guru_realloc(buf, new_len+1);	// shrink suitable size.
    v->str->len = new_len;

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
    U8P p0 = _data(v);
    U8P p1 = p0 + _size(v) - 1;

    if (*p1=='\n') p1--;
    if (*p1=='\r') p1--;

    U32 new_len = p1 - p0 + 1;
    if (_size(v)==new_len) return 0;

    U8P buf = _data(v);
    buf[new_len] = '\0';
    v->str->len = new_len;

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
    return _new((U8P)src, STRLEN((U8P)src));
}

//================================================================
/*! destructor

  @param  str	pointer to target value
*/
__GURU__ void
guru_str_delete(GV *v)
{
    guru_free(v->str->data);
    guru_free(v->str);
}

//================================================================
/*! append string (s1 += s2)

  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @param	mrbc_error_code
*/
__GURU__ void
guru_str_append(const GV *v0, const GV *v1)
{
    U32 len0 = v0->str->len;
    U32 len1 = (v1->gt==GT_STR) ? v1->str->len : 1;

    U8P s = (U8P)guru_realloc(v0->str->data, len0+len1+1);		// +'\0'

    assert(s!=NULL);						// out of memory
#if GURU_64BIT_ALIGN_REQUIRED
    assert(((U32A)s & 7)==0);
#endif
    if (v1->gt==GT_STR) {			// append str2
        MEMCPY(s + len0, v1->str->data, len1 + 1);
    }
    else if (v1->gt==GT_INT) {
        s[len0]   = v1->i;
        s[len0+1] = '\0';
    }
    v0->str->len  = len0 + len1;
    v0->str->data = s;
}

//================================================================
/*! append c string (s1 += s2)

  @param  s1	pointer to target value 1
  @param  s2	pointer to char (c_str)
  @param	mrbc_error_code
*/
__GURU__ void
guru_str_append_cstr(const GV *v0, const U8 *str)
{
    U32 len0 = v0->str->len;
    U32 len1 = STRLEN(str);

    U8P buf  = (U8P)guru_realloc(v0->str->data, len0+len1+1);

    assert(buf!=NULL);						// out of memory
#if GURU_64BIT_ALIGN_REQUIRED
    assert(((U32A)buf & 7)==0);
#endif
    MEMCPY(buf + len0, v0, len1 + 1);

    v0->str->len  = len0 + len1;
    v0->str->data = buf;
}

//================================================================
/*! add string (s1 + s2)

  @param  vm	pointer to VM.
  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @return	new string as s1 + s2
*/
__GURU__ GV
guru_str_add(const GV *v0, const GV *v1)
{
    guru_str *h0 = v0->str;
    guru_str *h1 = v1->str;

    GV  v  = _new(NULL, h0->len + h1->len);
    guru_str *s = v.str;

    MEMCPY(s->data,           h0->data, h0->len);
    MEMCPY(s->data + h0->len, h1->data, h1->len + 1);	// include the '\0'

    return v;
}

//================================================================
/*! (method) +
 */
__GURU__ void
c_string_add(GV v[], U32 argc)
{
    if (v[1].gt != GT_STR) {
        guru_na("str + other type");
    }
    else {
    	SET_RETURN(guru_str_add(v, v+1));
    }
}

//================================================================
/*! (method) *
 */
__GURU__ void
c_string_mul(GV v[], U32 argc)
{
    if (v[1].gt != GT_INT) {
        PRINTF("TypeError\n");	// raise?
        return;
    }
    GV ret = _new(NULL, _size(v) * v[1].i);
    if (ret.str==NULL) return;		// ENOMEM

    U8P p = (U8P)ret.str->data;
    for (U32 i = 0; i < v[1].i; i++) {
        MEMCPY(p, (U8P)_data(v), _size(v));
        p += _size(v);
    }
    *p = '\0';

    SET_RETURN(ret);
}

//================================================================
/*! (method) size, length
 */
__GURU__ void
c_string_size(GV v[], U32 argc)
{
    guru_int size = _size(v);

    SET_INT_RETURN(size);
}

//================================================================
/*! (method) to_i
 */
__GURU__ void
c_string_to_i(GV v[], U32 argc)
{
    U32 base = 10;
    if (argc) {
        base = v[1].i;
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    guru_int i = guru_atoi(_data(v), base);

    SET_INT_RETURN(i);
}

#if GURU_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__GURU__ void
c_string_to_f(GV v[], U32 argc)
{
    guru_float d = ATOF(_data(v));

    SET_FLOAT_RETURN(d);
}
#endif

//================================================================
/*! (method) <<
 */
__GURU__ void
c_string_append(GV v[], U32 argc)
{
    guru_str_append(v, v+1);
}

//================================================================
/*! (method) []
 */
__GURU__ void
c_string_slice(GV v[], U32 argc)
{
    GV *v1 = &v[1];
    GV *v2 = &v[2];

    if (argc==1 && v1->gt==GT_INT) {		// slice(n) -> String | nil
        U32 len = v->str->len;
        S32 idx = v1->i;
        S32 ch = -1;
        if (idx >= 0) {
            if (idx < len) {
                ch = *(v->str->data + idx);
            }
        }
        else {
            idx += len;
            if (idx >= 0) {
                ch = *(v->str->data + idx);
            }
        }
        if (ch < 0) goto RETURN_NIL;

        GV ret = _new(NULL, 1);
        if (!ret.str) goto RETURN_NIL;

        ret.str->data[0] = ch;
        ret.str->data[1] = '\0';

        SET_RETURN(ret);
    }
    else if (argc==2 && v1->gt==GT_INT && v2->gt==GT_INT) { 	// slice(n, len) -> String | nil
        U32 len = v->str->len;
        S32 idx = v1->i;
        if (idx < 0) idx += len;
        if (idx < 0) goto RETURN_NIL;

        S32 rlen = (v2->i < (len - idx)) ? v2->i : (len - idx);
        // min(v2->i, (len-idx))
        if (rlen < 0) goto RETURN_NIL;

        GV ret = _new((U8P)v->str->data + idx, rlen);
        if (!ret.str) goto RETURN_NIL;		// ENOMEM

        SET_RETURN(ret);
    }
    else {
    	PRINTF("Not support such case in String#[].\n");
    }
    return;

RETURN_NIL:
	SET_NIL_RETURN();
}

//================================================================
/*! (method) []=
 */
__GURU__ void
c_string_insert(GV v[], U32 argc)
{
    S32 nth;
    S32 len;
    GV *val;

    if (argc==2 &&								// self[n] = val
        v[1].gt==GT_INT &&
        v[2].gt==GT_STR) {
        nth = v[1].i;
        len = 1;
        val = &v[2];
    }
    else if (argc==3 &&							// self[n, len] = val
             v[1].gt==GT_INT &&
             v[2].gt==GT_INT &&
             v[3].gt==GT_STR) {
        nth = v[1].i;
        len = v[2].i;
        val = &v[3];
    }
    else {
        guru_na("case of c_string_insert");
        return;
    }

    U32 len1 = v->str->len;
    U32 len2 = val->str->len;
    if (nth < 0) nth = len1 + nth;               // adjust to positive number.
    if (len > len1 - nth) len = len1 - nth;
    if (nth < 0 || nth > len1 || len < 0) {
        PRINTF("IndexError\n");  // raise?
        return;
    }

    U8P str = (U8P)guru_realloc(_data(v), len1 + len2 - len + 1);
    if (!str) return;

    MEMCPY(str + nth + len2, str + nth + len, len1 - nth - len + 1);
    MEMCPY(str + nth, (U8P)_data(val), len2);
    v->str->len = len1 + len2 - len;

    v->str->data = str;

    ref_clr(v+1);
}

//================================================================
/*! (method) chomp
 */
__GURU__ void
c_string_chomp(GV v[], U32 argc)
{
    GV ret = _dup(v);
    _chomp(&ret);
    SET_RETURN(ret);
}

//================================================================
/*! (method) chomp!
 */
__GURU__ void
c_string_chomp_self(GV v[], U32 argc)
{
    if (_chomp(v)==0) {
        SET_NIL_RETURN();
    }
}

//================================================================
/*! (method) dup
 */
__GURU__ void
c_string_dup(GV v[], U32 argc)
{
    SET_RETURN(_dup(v));
}

//================================================================
/*! (method) index
 */
__GURU__ void
c_string_index(GV v[], U32 argc)
{
    S32 index;
    S32 offset;

    if (argc==1) {
        offset = 0;
    }
    else if (argc==2 && v[2].gt==GT_INT) {
        offset = v[2].i;
        if (offset < 0) offset += _size(v);
        if (offset < 0) goto NIL_RETURN;
    }
    else {
        goto NIL_RETURN;	// raise? ArgumentError
    }

    index = _index(v, v+1, offset);
    if (index < 0) goto NIL_RETURN;

    ref_clr(v+1);
    SET_INT_RETURN(index);
    return;

NIL_RETURN:
	ref_clr(v+1);
    SET_NIL_RETURN();
}

//================================================================
/*! (method) inspect
 */
#define BUF_SIZE 80

__GURU__ void
c_string_inspect(GV v[], U32 argc)
{
	const char    *hex = "0123456789ABCDEF";
    GV    ret  = guru_str_new("\"");

    U8 buf[BUF_SIZE];
    U8P p = buf;
    U8P s = (U8P)_data(v);

    for (U32 i=0; i < _size(v); i++, s++) {
        if (*s >= ' ' && *s < 0x80) {
        	*p++ = *s;
        }
        else {							// tiny isprint()
        	*p++ = '\\';
        	*p++ = 'x';
            *p++ = hex[*s >> 4];
            *p++ = hex[*s & 0x0f];
        }
    	if ((p-buf) > BUF_SIZE-5) {			// flush buffer
    		*p = '\0';
    		guru_str_append_cstr(&ret, buf);
    		p = buf;
    	}
    }
    *p++ = '\"';
    *p   = '\0';
    guru_str_append_cstr(&ret, buf);

    SET_RETURN(ret);
}

//================================================================
/*! (method) ord
 */
__GURU__ void
c_string_ord(GV v[], U32 argc)
{
    SET_INT_RETURN(_data(v)[0]);
}

//================================================================
/*! (method) split
 */
__GURU__ void
c_string_split(GV v[], U32 argc)
{
    guru_na("string#split");
}

//================================================================
/*! (method) sprintf
 */
__GURU__ void
c_object_sprintf(GV v[], U32 argc)
{
	guru_na("string#sprintf");
}

//================================================================
/*! (method) printf
 */
__GURU__ void
c_object_printf(GV v[], U32 argc)
{
	guru_na("string#printf");
}

//================================================================
/*! (method) lstrip
 */
__GURU__ void
c_string_lstrip(GV v[], U32 argc)
{
    GV ret = _dup(v);

    _strip(&ret, 0x01);	// 1: left side only

    SET_RETURN(ret);
}

//================================================================
/*! (method) lstrip!
 */
__GURU__ void
c_string_lstrip_self(GV v[], U32 argc)
{
    if (_strip(v, 0x01)==0) {	// 1: left side only
        SET_RETURN(GURU_NIL_NEW());
    }
}

//================================================================
/*! (method) rstrip
 */
__GURU__ void
c_string_rstrip(GV v[], U32 argc)
{
    GV ret = _dup(v);

    _strip(&ret, 0x02);							// 2: right side only

    SET_RETURN(ret);
}

//================================================================
/*! (method) rstrip!
 */
__GURU__ void
c_string_rstrip_self(GV v[], U32 argc)
{
    if (_strip(v, 0x02)==0) {				// 2: right side only
        SET_RETURN(GURU_NIL_NEW());			// keep refc
    }
}

//================================================================
/*! (method) strip
 */
__GURU__ void
c_string_strip(GV v[], U32 argc)
{
    GV ret = _dup(v);
    _strip(&ret, 0x03);	// 3: left and right
    SET_RETURN(ret);
}

//================================================================
/*! (method) strip!
 */
__GURU__ void
c_string_strip_self(GV v[], U32 argc)
{
    if (_strip(v, 0x03)==0) {		// 3: left and right
        SET_RETURN(GURU_NIL_NEW());	// keep refc
    }
}

//================================================================
/*! (method) to_sym
 */
__GURU__ void
c_string_to_sym(GV v[], U32 argc)
{
    SET_RETURN(guru_sym_new(_data(v)));
}

//================================================================
/*! initialize
 */
__GURU__ void
guru_init_class_string()
{
    guru_class *c = guru_class_string = guru_add_class("String", guru_class_object);

    guru_add_proc(c, "+",		c_string_add);
    guru_add_proc(c, "*",		c_string_mul);
    guru_add_proc(c, "size",	c_string_size);
    guru_add_proc(c, "length",	c_string_size);
    guru_add_proc(c, "to_i",	c_string_to_i);
    guru_add_proc(c, "<<",		c_string_append);
    guru_add_proc(c, "[]",		c_string_slice);
    guru_add_proc(c, "[]=",		c_string_insert);
    guru_add_proc(c, "chomp",	c_string_chomp);
    guru_add_proc(c, "chomp!",	c_string_chomp_self);
    guru_add_proc(c, "dup",		c_string_dup);
    guru_add_proc(c, "index",	c_string_index);
    guru_add_proc(c, "inspect",	c_string_inspect);
    guru_add_proc(c, "ord",		c_string_ord);
    guru_add_proc(c, "split",	c_string_split);
    guru_add_proc(c, "lstrip",	c_string_lstrip);
    guru_add_proc(c, "lstrip!",	c_string_lstrip_self);
    guru_add_proc(c, "rstrip",	c_string_rstrip);
    guru_add_proc(c, "rstrip!",	c_string_rstrip_self);
    guru_add_proc(c, "strip",	c_string_strip);
    guru_add_proc(c, "strip!",	c_string_strip_self);
    guru_add_proc(c, "to_sym",	c_string_to_sym);
    guru_add_proc(c, "intern",	c_string_to_sym);
#if GURU_USE_FLOAT
    guru_add_proc(c, "to_f",	c_string_to_f);
#endif

    guru_add_proc(guru_class_object, "sprintf",	c_object_sprintf);
    guru_add_proc(guru_class_object, "printf",	c_object_printf);
}

#endif // GURU_USE_STRING
