/*! @file
  @brief
  mruby/c String object

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>

#include "value.h"
#include "alloc.h"
#include "static.h"
#include "symbol.h"
#include "class.h"

#include "console.h"
#include "sprintf.h"

#include "c_string.h"

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

    for (int i = 0; i < sizeof(ws); i++) {
        if (ch==ws[i]) return true;
    }
    return false;
}

//================================================================
/*! get size
 */
__GURU__ __INLINE__ U32
_size(const mrbc_value *v)
{
    return v->str->size;
}

//================================================================
/*! get c-language string (U8 *)
 */
__GURU__ __INLINE__ U8P
_data(const mrbc_value *v)
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
__GURU__ mrbc_value
_new(const U8 *src, U32 len)
{
    mrbc_value ret = {.tt = GURU_TT_STRING};
    /*
      Allocate handle and string buffer.
    */
    mrbc_string *h = (mrbc_string *)mrbc_alloc(sizeof(mrbc_string));

    assert(h!=NULL);			// out of memory
#if GURU_REQUIRE_64BIT_ALIGNMENT
    assert(((U32P)h & 7)==0);
#endif

    U8 *s = (U8 *)mrbc_alloc(len);
    if (s==NULL) {					// ENOMEM
        mrbc_free(h);
        return ret;
    }
#if GURU_REQUIRE_64BIT_ALIGNMENT
    assert(((U32P)s & 7)==0);
#endif

    // deep copy source string
    if (src==NULL) 	s[0] = '\0';
    else 			MEMCPY(s, (U8 *)src, len+1);		// plus '\0'

    h->refc = 1;
    h->tt   = GURU_TT_STRING;	// TODO: for DEBUG
    h->size = len;
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
__GURU__ mrbc_value
_dup(const mrbc_value *v0)
{
    mrbc_string *h0 = v0->str;

    mrbc_value v1 = _new(NULL, h0->size);		// refc already set to 1
    if (v1.str==NULL) return v1;							// ENOMEM

    MEMCPY(v1.str->data, h0->data, h0->size + 1);

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
_index(const mrbc_value *v, const mrbc_value *pattern, U32 offset)
{
    U8  *p0 = _data(v) + offset;
    U8  *p1 = _data(pattern);
    S32 try_cnt = _size(v) - _size(pattern) - offset;

    while (try_cnt >= 0) {
        if (MEMCMP((U8 *)p0, (U8 *)p1, _size(pattern))==0) {
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
_strip(mrbc_value *v, U32 mode)
{
    U8 *p0 = _data(v);
    U8 *p1 = p0 + _size(v) - 1;

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
    U32 new_size = p1 - p0 + 1;
    if (_size(v)==new_size) return 0;

    U8 *buf = _data(v);
    if (p0 != buf) MEMCPY((U8 *)buf, (U8 *)p0, new_size);
    buf[new_size] = '\0';

    v->str->data = (U8 *)mrbc_realloc((U8 *)buf, new_size+1);	// shrink suitable size.
    v->str->size = new_size;

    return 1;
}

//================================================================
/*! remove the CR,LF in myself

  @param  src	pointer to target value
  @return	0 when not removed.
*/
__GURU__ int
_chomp(mrbc_value *v)
{
    U8 *p0 = _data(v);
    U8 *p1 = p0 + _size(v) - 1;

    if (*p1=='\n') p1--;
    if (*p1=='\r') p1--;

    U32 new_size = p1 - p0 + 1;
    if (_size(v)==new_size) return 0;

    U8 *buf = _data(v);
    buf[new_size] = '\0';
    v->str->size = new_size;

    return 1;
}

//================================================================
/*! constructor by c string

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @return 	string object
*/
__GURU__ mrbc_value
mrbc_string_new(const U8 *src)
{
    return _new(src, STRLEN(src));
}

//================================================================
/*! destructor

  @param  str	pointer to target value
*/
__GURU__ void
mrbc_string_delete(mrbc_value *v)
{
    mrbc_free(v->str->data);
    mrbc_free(v->str);
}

//================================================================
/*! append string (s1 += s2)

  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @param	mrbc_error_code
*/
__GURU__ void
mrbc_string_append(mrbc_value *v0, const mrbc_value *v1)
{
    U32 len0 = v0->str->size;
    U32 len1 = (v1->tt==GURU_TT_STRING) ? v1->str->size : 1;

    U8 *s = (U8 *)mrbc_realloc(v0->str->data, len0+len1+1);		// +'\0'

    assert(s!=NULL);						// out of memory
#if GURU_REQUIRE_64BIT_ALIGNMENT
    assert(((U32P)s & 7)==0);
#endif
    if (v1->tt==GURU_TT_STRING) {			// append str2
        MEMCPY((U8 *)s + len0, v1->str->data, len1 + 1);
    }
    else if (v1->tt==GURU_TT_FIXNUM) {
        s[len0]   = v1->i;
        s[len0+1] = '\0';
    }
    v0->str->size = len0 + len1;
    v0->str->data = s;
}

//================================================================
/*! append c string (s1 += s2)

  @param  s1	pointer to target value 1
  @param  s2	pointer to char (c_str)
  @param	mrbc_error_code
*/
__GURU__ void
mrbc_string_append_cstr(mrbc_value *v0, const U8 *v1)
{
    U32 len0 = v0->str->size;
    U32 len1 = STRLEN(v1);

    U8 *s = (U8 *)mrbc_realloc(v0->str->data, len0+len1+1);

    assert(s!=NULL);						// out of memory
#if GURU_REQUIRE_64BIT_ALIGNMENT
    assert(((U32P)s & 7)==0);
#endif
    MEMCPY(s + len0, (U8 *)v1, len1 + 1);

    v0->str->size = len0 + len1;
    v0->str->data = s;
}

//================================================================
/*! add string (s1 + s2)

  @param  vm	pointer to VM.
  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @return	new string as s1 + s2
*/
__GURU__ mrbc_value
mrbc_string_add(const mrbc_value *v1, const mrbc_value *v2)
{
    mrbc_string *h1 = v1->str;
    mrbc_string *h2 = v2->str;

    mrbc_value  v  = _new(NULL, h1->size + h2->size);
    mrbc_string *s = v.str;

    MEMCPY(s->data,            h1->data, h1->size);
    MEMCPY(s->data + h1->size, h2->data, h2->size + 1);	// include the '\0'

    return v;
}

//================================================================
/*! (method) +
 */
__GURU__ void
c_string_add(mrbc_value v[], U32 argc)
{
    if (v[1].tt != GURU_TT_STRING) {
        console_na("str + other type");
    }
    else {
    	SET_RETURN(mrbc_string_add(v, v+1));
    }
}

//================================================================
/*! (method) *
 */
__GURU__ void
c_string_mul(mrbc_value v[], U32 argc)
{
    if (v[1].tt != GURU_TT_FIXNUM) {
        console_str("TypeError\n");	// raise?
        return;
    }
    mrbc_value ret = _new(NULL, _size(v) * v[1].i);
    if (ret.str==NULL) return;		// ENOMEM

    U8 *p = (U8 *)ret.str->data;
    for (int i = 0; i < v[1].i; i++) {
        MEMCPY((U8 *)p, (U8 *)_data(v), _size(v));
        p += _size(v);
    }
    *p = '\0';

    SET_RETURN(ret);
}

//================================================================
/*! (method) size, length
 */
__GURU__ void
c_string_size(mrbc_value v[], U32 argc)
{
    mrbc_int size = _size(v);

    SET_INT_RETURN(size);
}

//================================================================
/*! (method) to_i
 */
__GURU__ void
c_string_to_i(mrbc_value v[], U32 argc)
{
    U32 base = 10;
    if (argc) {
        base = v[1].i;
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    mrbc_int i = guru_atoi(_data(v), base);

    SET_INT_RETURN(i);
}

#if GURU_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__GURU__ void
c_string_to_f(mrbc_value v[], U32 argc)
{
    mrbc_float d = ATOF(_data(v));

    SET_FLOAT_RETURN(d);
}
#endif

//================================================================
/*! (method) <<
 */
__GURU__ void
c_string_append(mrbc_value v[], U32 argc)
{
    mrbc_string_append(v, v+1);
}

//================================================================
/*! (method) []
 */
__GURU__ void
c_string_slice(mrbc_value v[], U32 argc)
{
    mrbc_value *v1 = &v[1];
    mrbc_value *v2 = &v[2];

    if (argc==1 && v1->tt==GURU_TT_FIXNUM) {		// slice(n) -> String | nil
        U32 len = v->str->size;
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

        mrbc_value ret = _new(NULL, 1);
        if (!ret.str) goto RETURN_NIL;

        ret.str->data[0] = ch;
        ret.str->data[1] = '\0';

        SET_RETURN(ret);
    }
    else if (argc==2 && v1->tt==GURU_TT_FIXNUM && v2->tt==GURU_TT_FIXNUM) { 	// slice(n, len) -> String | nil
        U32 len = v->str->size;
        S32 idx = v1->i;
        if (idx < 0) idx += len;
        if (idx < 0) goto RETURN_NIL;

        S32 rlen = (v2->i < (len - idx)) ? v2->i : (len - idx);
        // min(v2->i, (len-idx))
        if (rlen < 0) goto RETURN_NIL;

        mrbc_value ret = _new((U8 *)v->str->data + idx, rlen);
        if (!ret.str) goto RETURN_NIL;		// ENOMEM

        SET_RETURN(ret);
    }
    else {
    	console_str("Not support such case in String#[].\n");
    }
    return;

RETURN_NIL:
	SET_NIL_RETURN();
}

//================================================================
/*! (method) []=
 */
__GURU__ void
c_string_insert(mrbc_value v[], U32 argc)
{
    S32 nth;
    S32 len;
    mrbc_value *val;

    if (argc==2 &&								// self[n] = val
        v[1].tt==GURU_TT_FIXNUM &&
        v[2].tt==GURU_TT_STRING) {
        nth = v[1].i;
        len = 1;
        val = &v[2];
    }
    else if (argc==3 &&							// self[n, len] = val
             v[1].tt==GURU_TT_FIXNUM &&
             v[2].tt==GURU_TT_FIXNUM &&
             v[3].tt==GURU_TT_STRING) {
        nth = v[1].i;
        len = v[2].i;
        val = &v[3];
    }
    else {
        console_na("case of c_string_insert");
        return;
    }

    U32 len1 = v->str->size;
    U32 len2 = val->str->size;
    if (nth < 0) nth = len1 + nth;               // adjust to positive number.
    if (len > len1 - nth) len = len1 - nth;
    if (nth < 0 || nth > len1 || len < 0) {
        console_str("IndexError\n");  // raise?
        return;
    }

    U8 *str = (U8 *)mrbc_realloc(_data(v), len1 + len2 - len + 1);
    if (!str) return;

    MEMCPY(str + nth + len2, str + nth + len, len1 - nth - len + 1);
    MEMCPY(str + nth, (U8 *)_data(val), len2);
    v->str->size = len1 + len2 - len;

    v->str->data = str;

    mrbc_release(v+1);
}

//================================================================
/*! (method) chomp
 */
__GURU__ void
c_string_chomp(mrbc_value v[], U32 argc)
{
    mrbc_value ret = _dup(v);
    _chomp(&ret);
    SET_RETURN(ret);
}

//================================================================
/*! (method) chomp!
 */
__GURU__ void
c_string_chomp_self(mrbc_value v[], U32 argc)
{
    if (_chomp(v)==0) {
        SET_NIL_RETURN();
    }
}

//================================================================
/*! (method) dup
 */
__GURU__ void
c_string_dup(mrbc_value v[], U32 argc)
{
    SET_RETURN(_dup(v));
}

//================================================================
/*! (method) index
 */
__GURU__ void
c_string_index(mrbc_value v[], U32 argc)
{
    S32 index;
    S32 offset;

    if (argc==1) {
        offset = 0;
    }
    else if (argc==2 && v[2].tt==GURU_TT_FIXNUM) {
        offset = v[2].i;
        if (offset < 0) offset += _size(v);
        if (offset < 0) goto NIL_RETURN;
    }
    else {
        goto NIL_RETURN;	// raise? ArgumentError
    }

    index = _index(v, v+1, offset);
    if (index < 0) goto NIL_RETURN;

    mrbc_release(v+1);
    SET_INT_RETURN(index);
    return;

NIL_RETURN:
	mrbc_release(v+1);
    SET_NIL_RETURN();
}

//================================================================
/*! (method) inspect
 */
#define BUF_SIZE 80

__GURU__ void
c_string_inspect(mrbc_value v[], U32 argc)
{
	const char    *hex = "0123456789ABCDEF";
    mrbc_value    ret  = mrbc_string_new("\"");

    U8 buf[BUF_SIZE];
    U8 *p = buf, *s = (U8 *)_data(v);

    for (int i=0; i < _size(v); i++, s++) {
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
    		mrbc_string_append_cstr(&ret, buf);
    		p = buf;
    	}
    }
    *p++ = '\"';
    *p   = '\0';
    mrbc_string_append_cstr(&ret, buf);

    SET_RETURN(ret);
}

//================================================================
/*! (method) ord
 */
__GURU__ void
c_string_ord(mrbc_value v[], U32 argc)
{
    SET_INT_RETURN(_data(v)[0]);
}

//================================================================
/*! (method) split
 */
__GURU__ void
c_string_split(mrbc_value v[], U32 argc)
{
    console_na("string#split");
}

//================================================================
/*! (method) sprintf
 */
__GURU__ void
c_object_sprintf(mrbc_value v[], U32 argc)
{
	U8 buf[20];
	const U8 *str = guru_vprintf(buf, "<#%s:%08x>", v, argc);
	for (int i=1; i<=argc; i++) {
		mrbc_release(v+i);
	}
    SET_RETURN(mrbc_string_new(str));
}

//================================================================
/*! (method) printf
 */
__GURU__ void
c_object_printf(mrbc_value v[], U32 argc)
{
    c_object_sprintf(v, argc);
    console_str(_data(v));
    SET_NIL_RETURN();
}

//================================================================
/*! (method) lstrip
 */
__GURU__ void
c_string_lstrip(mrbc_value v[], U32 argc)
{
    mrbc_value ret = _dup(v);

    _strip(&ret, 0x01);	// 1: left side only

    SET_RETURN(ret);
}

//================================================================
/*! (method) lstrip!
 */
__GURU__ void
c_string_lstrip_self(mrbc_value v[], U32 argc)
{
    if (_strip(v, 0x01)==0) {	// 1: left side only
        SET_RETURN(mrbc_nil_value());
    }
}

//================================================================
/*! (method) rstrip
 */
__GURU__ void
c_string_rstrip(mrbc_value v[], U32 argc)
{
    mrbc_value ret = _dup(v);

    _strip(&ret, 0x02);							// 2: right side only

    SET_RETURN(ret);
}

//================================================================
/*! (method) rstrip!
 */
__GURU__ void
c_string_rstrip_self(mrbc_value v[], U32 argc)
{
    if (_strip(v, 0x02)==0) {				// 2: right side only
        SET_RETURN(mrbc_nil_value());			// keep refc
    }
}

//================================================================
/*! (method) strip
 */
__GURU__ void
c_string_strip(mrbc_value v[], U32 argc)
{
    mrbc_value ret = _dup(v);
    _strip(&ret, 0x03);	// 3: left and right
    SET_RETURN(ret);
}

//================================================================
/*! (method) strip!
 */
__GURU__ void
c_string_strip_self(mrbc_value v[], U32 argc)
{
    if (_strip(v, 0x03)==0) {		// 3: left and right
        SET_RETURN(mrbc_nil_value());	// keep refc
    }
}

//================================================================
/*! (method) to_sym
 */
__GURU__ void
c_string_to_sym(mrbc_value v[], U32 argc)
{
    SET_RETURN(mrbc_symbol_new(_data(v)));
}

//================================================================
/*! initialize
 */
__GURU__ void
mrbc_init_class_string()
{
    mrbc_class *c = mrbc_class_string = mrbc_define_class("String", mrbc_class_object);

    mrbc_define_method(c, "+",		c_string_add);
    mrbc_define_method(c, "*",		c_string_mul);
    mrbc_define_method(c, "size",	c_string_size);
    mrbc_define_method(c, "length",	c_string_size);
    mrbc_define_method(c, "to_i",	c_string_to_i);
    mrbc_define_method(c, "to_s",	c_nop);
    mrbc_define_method(c, "<<",		c_string_append);
    mrbc_define_method(c, "[]",		c_string_slice);
    mrbc_define_method(c, "[]=",	c_string_insert);
    mrbc_define_method(c, "chomp",	c_string_chomp);
    mrbc_define_method(c, "chomp!",	c_string_chomp_self);
    mrbc_define_method(c, "dup",	c_string_dup);
    mrbc_define_method(c, "index",	c_string_index);
    mrbc_define_method(c, "inspect",c_string_inspect);
    mrbc_define_method(c, "ord",	c_string_ord);
    mrbc_define_method(c, "split",	c_string_split);
    mrbc_define_method(c, "lstrip",	c_string_lstrip);
    mrbc_define_method(c, "lstrip!",c_string_lstrip_self);
    mrbc_define_method(c, "rstrip",	c_string_rstrip);
    mrbc_define_method(c, "rstrip!",c_string_rstrip_self);
    mrbc_define_method(c, "strip",	c_string_strip);
    mrbc_define_method(c, "strip!",	c_string_strip_self);
    mrbc_define_method(c, "to_sym",	c_string_to_sym);
    mrbc_define_method(c, "intern",	c_string_to_sym);
#if GURU_USE_FLOAT
    mrbc_define_method(c, "to_f",	c_string_to_f);
#endif

    mrbc_define_method(mrbc_class_object, "sprintf",	c_object_sprintf);
    mrbc_define_method(mrbc_class_object, "printf",		c_object_printf);
}

#endif // GURU_USE_STRING
