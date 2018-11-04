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

#if MRBC_USE_STRING
//================================================================
/*! white space character test

  @param  ch	character code.
  @return	result.
*/
__GURU__ int
_is_space(int ch)
{
    static const char ws[] = " \t\r\n\f\v";	// '\0' on tail

    for (int i = 0; i < sizeof(ws); i++) {
        if (ch==ws[i]) return 1;
    }
    return 0;
}

//================================================================
/*! get size
 */
__GURU__ __INLINE__ int
_size(const mrbc_value *v)
{
    return v->str->size;
}

//================================================================
/*! get c-language string (char *)
 */
__GURU__ __INLINE__ char*
_data(const mrbc_value *v)
{
    return (char*)v->str->data;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @param  len	source length
  @return 	string object
*/
__GURU__ mrbc_value
_new(const char *src, int len)
{
    mrbc_value ret = {.tt = MRBC_TT_STRING};
    /*
      Allocate handle and string buffer.
    */
    mrbc_string *h = (mrbc_string *)mrbc_alloc(sizeof(mrbc_string));

    assert(h!=NULL);			// out of memory
    assert(((uintptr_t)h & 7)==0);

    uint8_t *s = (uint8_t *)mrbc_alloc(len);
    if (s==NULL) {					// ENOMEM
        mrbc_free(h);
        return ret;
    }
    assert(((uintptr_t)s & 7)==0);

    // deep copy source string
    if (src==NULL) 	s[0] = '\0';
    else 			MEMCPY(s, (uint8_t *)src, len+1);		// plus '\0'

    h->refc = 1;
    h->tt   = MRBC_TT_STRING;	// TODO: for DEBUG
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
_dup(mrbc_value *v0)
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
__GURU__ int
_index(const mrbc_value *v, const mrbc_value *pattern, int offset)
{
    char *p0 = _data(v) + offset;
    char *p1 = _data(pattern);
    int try_cnt = _size(v) - _size(pattern) - offset;

    while (try_cnt >= 0) {
        if (MEMCMP((uint8_t *)p0, (uint8_t *)p1, _size(pattern))==0) {
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
__GURU__ int
_strip(mrbc_value *v, int mode)
{
    char *p0 = _data(v);
    char *p1 = p0 + _size(v) - 1;

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
    int new_size = p1 - p0 + 1;
    if (_size(v)==new_size) return 0;

    char *buf = _data(v);
    if (p0 != buf) MEMCPY((uint8_t *)buf, (uint8_t *)p0, new_size);
    buf[new_size] = '\0';

    mrbc_realloc((uint8_t *)buf, new_size+1);	// shrink suitable size.
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
    char *p0 = _data(v);
    char *p1 = p0 + _size(v) - 1;

    if (*p1=='\n') p1--;
    if (*p1=='\r') p1--;

    int new_size = p1 - p0 + 1;
    if (_size(v)==new_size) return 0;

    char *buf = _data(v);
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
mrbc_string_new(const char *src)
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
__GURU__ int
mrbc_string_append(mrbc_value *v1, const mrbc_value *v2)
{
    int len1 = v1->str->size;
    int len2 = (v2->tt==MRBC_TT_STRING) ? v2->str->size : 1;

    uint8_t *s = (uint8_t *)mrbc_realloc(v1->str->data, len1+len2+1);		// +'\0'

    assert(s!=NULL);						// out of memory
    assert(((uintptr_t)s & 7)==0);

    if (v2->tt==MRBC_TT_STRING) {
        MEMCPY((uint8_t *)s + len1, v2->str->data, len2 + 1);
    }
    else if (v2->tt==MRBC_TT_FIXNUM) {
        s[len1]   = v2->i;
        s[len1+1] = '\0';
    }
    v1->str->size = len1 + len2;
    v1->str->data = s;

    return 0;
}

//================================================================
/*! append c string (s1 += s2)

  @param  s1	pointer to target value 1
  @param  s2	pointer to char (c_str)
  @param	mrbc_error_code
*/
__GURU__ int
mrbc_string_append_cstr(mrbc_value *v1, const char *v2)
{
    int len1 = v1->str->size;
    int len2 = STRLEN(v2);

    uint8_t *s = (uint8_t *)mrbc_realloc(v1->str->data, len1+len2+1);

    assert(s!=NULL);						// out of memory
    assert(((uintptr_t)s & 7)==0);

    MEMCPY(s + len1, (uint8_t *)v2, len2 + 1);

    v1->str->size = len1 + len2;
    v1->str->data = s;

    return 0;
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
c_string_add(mrbc_value v[], int argc)
{
    if (v[1].tt != MRBC_TT_STRING) {
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
c_string_mul(mrbc_value v[], int argc)
{
    if (v[1].tt != MRBC_TT_FIXNUM) {
        console_str("TypeError\n");	// raise?
        return;
    }
    mrbc_value ret = _new(NULL, _size(v) * v[1].i);
    if (ret.str==NULL) return;		// ENOMEM

    uint8_t *p = (uint8_t *)ret.str->data;
    for (int i = 0; i < v[1].i; i++) {
        MEMCPY((uint8_t *)p, (uint8_t *)_data(v), _size(v));
        p += _size(v);
    }
    *p = 0;

    SET_RETURN(ret);
}

//================================================================
/*! (method) size, length
 */
__GURU__ void
c_string_size(mrbc_value v[], int argc)
{
    mrbc_int size = _size(v);

    SET_INT_RETURN(size);
}

//================================================================
/*! (method) to_i
 */
__GURU__ void
c_string_to_i(mrbc_value v[], int argc)
{
    int base = 10;
    if (argc) {
        base = v[1].i;
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    mrbc_int i = guru_atoi(_data(v), base);

    SET_INT_RETURN(i);
}

#if MRBC_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__GURU__ void
c_string_to_f(mrbc_value v[], int argc)
{
    mrbc_float d = ATOF(_data(v));

    SET_FLOAT_RETURN(d);
}
#endif

//================================================================
/*! (method) <<
 */
__GURU__ void
c_string_append(mrbc_value v[], int argc)
{
    if (!mrbc_string_append(v, v+1)) {
        // raise ? ENOMEM
    }
}

//================================================================
/*! (method) []
 */
__GURU__ void
c_string_slice(mrbc_value v[], int argc)
{
    mrbc_value *v1 = &v[1];
    mrbc_value *v2 = &v[2];

    if (argc==1 && v1->tt==MRBC_TT_FIXNUM) {		// slice(n) -> String | nil
        int len = v->str->size;
        int idx = v1->i;
        int ch = -1;
        if (idx >= 0) {
            if (idx < len) {
                ch = *(v->str->data + idx);
            }
        } else {
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
    else if (argc==2 && v1->tt==MRBC_TT_FIXNUM && v2->tt==MRBC_TT_FIXNUM) { 	// slice(n, len) -> String | nil
        int len = v->str->size;
        int idx = v1->i;
        if (idx < 0) idx += len;
        if (idx < 0) goto RETURN_NIL;

        int rlen = (v2->i < (len - idx)) ? v2->i : (len - idx);
        // min(v2->i, (len-idx))
        if (rlen < 0) goto RETURN_NIL;

        mrbc_value ret = _new((char *)v->str->data + idx, rlen);
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
c_string_insert(mrbc_value v[], int argc)
{
    int nth;
    int len;
    mrbc_value *val;

    if (argc==2 &&								// self[n] = val
        v[1].tt==MRBC_TT_FIXNUM &&
        v[2].tt==MRBC_TT_STRING) {
        nth = v[1].i;
        len = 1;
        val = &v[2];
    }
    else if (argc==3 &&							// self[n, len] = val
             v[1].tt==MRBC_TT_FIXNUM &&
             v[2].tt==MRBC_TT_FIXNUM &&
             v[3].tt==MRBC_TT_STRING) {
        nth = v[1].i;
        len = v[2].i;
        val = &v[3];
    }
    else {
        console_na("case of c_string_insert");
        return;
    }

    int len1 = v->str->size;
    int len2 = val->str->size;
    if (nth < 0) nth = len1 + nth;               // adjust to positive number.
    if (len > len1 - nth) len = len1 - nth;
    if (nth < 0 || nth > len1 || len < 0) {
        console_str("IndexError\n");  // raise?
        return;
    }

    uint8_t *str = (uint8_t *)mrbc_realloc(_data(v), len1 + len2 - len + 1);
    if (!str) return;

    MEMCPY(str + nth + len2, str + nth + len, len1 - nth - len + 1);
    MEMCPY(str + nth, (uint8_t *)_data(val), len2);
    v->str->size = len1 + len2 - len;

    v->str->data = str;
}

//================================================================
/*! (method) chomp
 */
__GURU__ void
c_string_chomp(mrbc_value v[], int argc)
{
    mrbc_value ret = _dup(v);
    _chomp(&ret);
    SET_RETURN(ret);
}

//================================================================
/*! (method) chomp!
 */
__GURU__ void
c_string_chomp_self(mrbc_value v[], int argc)
{
    if (_chomp(v)==0) {
        SET_NIL_RETURN();
    }
}

//================================================================
/*! (method) dup
 */
__GURU__ void
c_string_dup(mrbc_value v[], int argc)
{
    SET_RETURN(_dup(v));
}

//================================================================
/*! (method) index
 */
__GURU__ void
c_string_index(mrbc_value v[], int argc)
{
    int index;
    int offset;

    if (argc==1) {
        offset = 0;
    }
    else if (argc==2 && v[2].tt==MRBC_TT_FIXNUM) {
        offset = v[2].i;
        if (offset < 0) offset += _size(v);
        if (offset < 0) goto NIL_RETURN;
    }
    else {
        goto NIL_RETURN;	// raise? ArgumentError
    }

    index = _index(v, v+1, offset);
    if (index < 0) goto NIL_RETURN;

    SET_INT_RETURN(index);
    return;

NIL_RETURN:
    SET_NIL_RETURN();
}

//================================================================
/*! (method) inspect
 */
__GURU__ void
c_string_inspect(mrbc_value v[], int argc)
{
    char buf[10] = "\\x";
    mrbc_value ret = mrbc_string_new("\"");
    const unsigned char *s = (const unsigned char *)_data(v);
  
    for (int i = 0; i < _size(v); i++) {
        if (s[i] < ' ' || 0x7f <= s[i]) {	// tiny isprint()
            buf[2] = "0123456789ABCDEF"[s[i] >> 4];
            buf[3] = "0123456789ABCDEF"[s[i] & 0x0f];
            mrbc_string_append_cstr(&ret, buf);
        } else {
            buf[3] = s[i];
            mrbc_string_append_cstr(&ret, buf+3);
        }
    }
    mrbc_string_append_cstr(&ret, "\"");

    SET_RETURN(ret);
}

//================================================================
/*! (method) ord
 */
__GURU__ void
c_string_ord(mrbc_value v[], int argc)
{
    SET_INT_RETURN(_data(v)[0]);
}

//================================================================
/*! (method) split
 */
__GURU__ void
c_string_split(mrbc_value v[], int argc)
{
    console_na("string#split");
}

//================================================================
/*! (method) sprintf
 */
__GURU__ void
c_object_sprintf(mrbc_value v[], int argc)
{
	char buf[20];
	const char *str = guru_vprintf(buf, "<#%s:%08x>", v, argc);

    SET_RETURN(mrbc_string_new(str));
}

//================================================================
/*! (method) printf
 */
__GURU__ void
c_object_printf(mrbc_value v[], int argc)
{
    c_object_sprintf(v, argc);
    console_str(_data(v));
    SET_NIL_RETURN();
}

//================================================================
/*! (method) lstrip
 */
__GURU__ void
c_string_lstrip(mrbc_value v[], int argc)
{
    mrbc_value ret = _dup(v);

    _strip(&ret, 0x01);	// 1: left side only

    SET_RETURN(ret);
}

//================================================================
/*! (method) lstrip!
 */
__GURU__ void
c_string_lstrip_self(mrbc_value v[], int argc)
{
    if (_strip(v, 0x01)==0) {	// 1: left side only
        SET_RETURN(mrbc_nil_value());
    }
}

//================================================================
/*! (method) rstrip
 */
__GURU__ void
c_string_rstrip(mrbc_value v[], int argc)
{
    mrbc_value ret = _dup(v);

    _strip(&ret, 0x02);							// 2: right side only

    SET_RETURN(ret);
}

//================================================================
/*! (method) rstrip!
 */
__GURU__ void
c_string_rstrip_self(mrbc_value v[], int argc)
{
    if (_strip(v, 0x02)==0) {				// 2: right side only
        SET_RETURN(mrbc_nil_value());			// keep refc
    }
}

//================================================================
/*! (method) strip
 */
__GURU__ void
c_string_strip(mrbc_value v[], int argc)
{
    mrbc_value ret = _dup(v);
    _strip(&ret, 0x03);	// 3: left and right
    SET_RETURN(ret);
}

//================================================================
/*! (method) strip!
 */
__GURU__ void
c_string_strip_self(mrbc_value v[], int argc)
{
    if (_strip(v, 0x03)==0) {		// 3: left and right
        SET_RETURN(mrbc_nil_value());	// keep refc
    }
}

//================================================================
/*! (method) to_sym
 */
__GURU__ void
c_string_to_sym(mrbc_value v[], int argc)
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
#if MRBC_USE_FLOAT
    mrbc_define_method(c, "to_f",	c_string_to_f);
#endif

    mrbc_define_method(mrbc_class_object, "sprintf",	c_object_sprintf);
    mrbc_define_method(mrbc_class_object, "printf",		c_object_printf);
}

#endif // MRBC_USE_STRING
