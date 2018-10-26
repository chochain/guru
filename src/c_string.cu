/*! @file
  @brief
  mruby/c String object

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "value.h"
#include "alloc.h"
#include "static.h"
#include "symbol.h"
#include "errorcode.h"
#include "console.h"
#include "class.h"
#include "sprintf.h"
#include "c_string.h"

#if MRBC_USE_ARRAY
//#include "c_array.h"
#endif

#if MRBC_USE_STRING
//================================================================
/*! white space character test

  @param  ch	character code.
  @return	result.
*/
__GURU__
int _is_space(int ch)
{
    static const char ws[] = " \t\r\n\f\v";	// '\0' on tail

    for (int i = 0; i < sizeof(ws); i++) {
        if (ch==ws[i]) return 1;
    }
    return 0;
}

//================================================================
/*! get c-language string (char *)
 */
__GURU__ __INLINE__
char *_string_cstr(const mrbc_value *v)
{
    return (char*)v->str->data;
}

//================================================================
/*! get size
 */
__GURU__ __INLINE__
int _string_size(const mrbc_value *v)
{
    return v->str->size;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @param  len	source length
  @return 	string object
*/
__GURU__
mrbc_value _mrbc_string_new(const char *src, int len)
{
    mrbc_value value = {.tt = MRBC_TT_STRING};
    /*
      Allocate handle and string buffer.
    */
    mrbc_string *h = (mrbc_string *)mrbc_alloc(sizeof(mrbc_string));
    if (!h) return value;		// ENOMEM

    uint8_t *str = (uint8_t *)mrbc_alloc(len);
    if (!str) {					// ENOMEM
        mrbc_free(h);
        return value;
    }

    // deep copy source string
    if (src==NULL) 	str[0] = '\0';
    else 			MEMCPY(str, (uint8_t *)src, len+1);		// plus '\0'

    h->refc = 1;
    h->tt   = MRBC_TT_STRING;	// TODO: for DEBUG
    h->size = len;
    h->data = str;

    value.str = h;

    return value;
}

//================================================================
/*! constructor by c string

  @param  vm	pointer to VM.
  @param  src	source string or NULL
  @return 	string object
*/
__GURU__
mrbc_value mrbc_string_new(const char *src)
{
    return _mrbc_string_new(src, STRLEN(src));
}

//================================================================
/*! destructor

  @param  str	pointer to target value
*/
__GURU__
void mrbc_string_delete(mrbc_value *v)
{
    mrbc_free(v->str->data);
    mrbc_free(v->str);
}

//================================================================
/*! duplicate string

  @param  vm	pointer to VM.
  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @return	new string as s1 + s2
*/
__GURU__
mrbc_value mrbc_string_dup(mrbc_value *v0)
{
    mrbc_string *h0 = v0->str;

    mrbc_value v1 = _mrbc_string_new(NULL, h0->size);
    if (v1.str==NULL) return v1;		// ENOMEM

    MEMCPY(v1.str->data, h0->data, h0->size + 1);

    return v1;
}

//================================================================
/*! add string (s1 + s2)

  @param  vm	pointer to VM.
  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @return	new string as s1 + s2
*/
__GURU__
mrbc_value mrbc_string_add(const mrbc_value *v1, const mrbc_value *v2)
{
    mrbc_string *h1 = v1->str;
    mrbc_string *h2 = v2->str;

    mrbc_value  v  = _mrbc_string_new(NULL, h1->size + h2->size);
    mrbc_string *s = v.str;
    if (s==NULL) return v;		// ENOMEM

    MEMCPY(s->data,            h1->data, h1->size);
    MEMCPY(s->data + h1->size, h2->data, h2->size + 1);	// include the '\0'

    return v;
}

//================================================================
/*! append string (s1 += s2)

  @param  s1	pointer to target value 1
  @param  s2	pointer to target value 2
  @param	mrbc_error_code
*/
__GURU__
int mrbc_string_append(mrbc_value *v1, const mrbc_value *v2)
{
    int len1 = v1->str->size;
    int len2 = (v2->tt==MRBC_TT_STRING) ? v2->str->size : 1;

    uint8_t *str = (uint8_t *)mrbc_realloc(v1->str->data, len1+len2+1);
    if (!str) return E_NOMEMORY_ERROR;

    if (v2->tt==MRBC_TT_STRING) {
        MEMCPY((uint8_t *)str + len1, v2->str->data, len2 + 1);
    }
    else if (v2->tt==MRBC_TT_FIXNUM) {
        str[len1]   = v2->i;
        str[len1+1] = '\0';
    }
    v1->str->size = len1 + len2;
    v1->str->data = str;

    return 0;
}

//================================================================
/*! append c string (s1 += s2)

  @param  s1	pointer to target value 1
  @param  s2	pointer to char (c_str)
  @param	mrbc_error_code
*/
__GURU__
int mrbc_string_append_cstr(mrbc_value *v1, const char *v2)
{
    int len1 = v1->str->size;
    int len2 = STRLEN(v2);

    uint8_t *str = (uint8_t *)mrbc_realloc(v1->str->data, len1+len2+1);
    if (!str) return E_NOMEMORY_ERROR;

    MEMCPY(str + len1, (uint8_t *)v2, len2 + 1);

    v1->str->size = len1 + len2;
    v1->str->data = str;

    return 0;
}

//================================================================
/*! locate a substring in a string

  @param  src		pointer to target string
  @param  pattern	pointer to substring
  @param  offset	search offset
  @return		position index. or minus value if not found.
*/
__GURU__
int mrbc_string_index(const mrbc_value *v, const mrbc_value *pattern, int offset)
{
    char *p1 = VSTR(v) + offset;
    char *p2 = VSTR(pattern);
    int try_cnt = VSTRLEN(v) - VSTRLEN(pattern) - offset;

    while (try_cnt >= 0) {
        if (MEMCMP((uint8_t *)p1, (uint8_t *)p2, VSTRLEN(pattern))==0) {
            return p1 - VSTR(v);	// matched.
        }
        try_cnt--;
        p1++;
    }
    return -1;
}

//================================================================
/*! remove the whitespace in myself

  @param  src	pointer to target value
  @param  mode	1:left-side, 2:right-side, 3:each
  @return	0 when not removed.
*/
__GURU__
int mrbc_string_strip(mrbc_value *v, int mode)
{
    char *p1 = VSTR(v);
    char *p2 = p1 + VSTRLEN(v) - 1;

    // left-side
    if (mode & 0x01) {
        while (p1 <= p2) {
            if (*p1=='\0') break;
            if (!_is_space(*p1)) break;
            p1++;
        }
    }
    // right-side
    if (mode & 0x02) {
        while (p1 <= p2) {
            if (!_is_space(*p2)) break;
            p2--;
        }
    }
    int new_size = p2 - p1 + 1;
    if (VSTRLEN(v)==new_size) return 0;

    char *buf = VSTR(v);
    if (p1 != buf) MEMCPY((uint8_t *)buf, (uint8_t *)p1, new_size);
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
__GURU__
int mrbc_string_chomp(mrbc_value *v)
{
    char *p1 = VSTR(v);
    char *p2 = p1 + VSTRLEN(v) - 1;

    if (*p2=='\n') p2--;
    if (*p2=='\r') p2--;

    int new_size = p2 - p1 + 1;
    if (VSTRLEN(v)==new_size) return 0;

    char *buf = VSTR(v);
    buf[new_size] = '\0';
    v->str->size = new_size;

    return 1;
}

//================================================================
/*! (method) +
 */
__GURU__
void c_string_add(mrbc_value v[], int argc)
{
    if (v[1].tt != MRBC_TT_STRING) {
        console_str("Not support STRING + Other\n");
        return;
    }
    mrbc_value value = mrbc_string_add(&v[0], &v[1]);
    SET_RETURN(value);
}

//================================================================
/*! (method) *
 */
__GURU__
void c_string_mul(mrbc_value v[], int argc)
{
    if (v[1].tt != MRBC_TT_FIXNUM) {
        console_str("TypeError\n");	// raise?
        return;
    }
    mrbc_value value = _mrbc_string_new(NULL, VSTRLEN(&v[0]) * v[1].i);
    if (value.str==NULL) return;		// ENOMEM

    uint8_t *p = (uint8_t *)value.str->data;
    for (int i = 0; i < v[1].i; i++) {
        MEMCPY((uint8_t *)p, (uint8_t *)VSTR(&v[0]), VSTRLEN(&v[0]));
        p += VSTRLEN(&v[0]);
    }
    *p = 0;

    SET_RETURN(value);
}

//================================================================
/*! (method) size, length
 */
__GURU__
void c_string_size(mrbc_value v[], int argc)
{
    mrbc_int size = VSTRLEN(&v[0]);

    SET_INT_RETURN(size);
}

//================================================================
/*! (method) to_i
 */
__GURU__
void c_string_to_i(mrbc_value v[], int argc)
{
    int base = 10;
    if (argc) {
        base = v[1].i;
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    mrbc_int i = guru_atoi(VSTR(v), base);

    SET_INT_RETURN(i);
}

#if MRBC_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__GURU__
void c_string_to_f(mrbc_value v[], int argc)
{
    mrbc_float d = ATOF(VSTR(v));

    SET_FLOAT_RETURN(d);
}
#endif

//================================================================
/*! (method) <<
 */
__GURU__
void c_string_append(mrbc_value v[], int argc)
{
    if (!mrbc_string_append(&v[0], &v[1])) {
        // raise ? ENOMEM
    }
}

//================================================================
/*! (method) []
 */
__GURU__
void c_string_slice(mrbc_value v[], int argc)
{
    mrbc_value *v1 = &v[1];
    mrbc_value *v2 = &v[2];

    /*
      in case of slice(nth) -> String | nil
    */
    if (argc==1 && v1->tt==MRBC_TT_FIXNUM) {
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

        mrbc_value value = _mrbc_string_new(NULL, 1);
        if (!value.str) goto RETURN_NIL;		// ENOMEM

        value.str->data[0] = ch;
        value.str->data[1] = '\0';

        SET_RETURN(value);
        return;		// normal return
    }

    /*
      in case of slice(nth, len) -> String | nil
    */
    if (argc==2 && v1->tt==MRBC_TT_FIXNUM && v2->tt==MRBC_TT_FIXNUM) {
        int len = v->str->size;
        int idx = v1->i;
        if (idx < 0) idx += len;
        if (idx < 0) goto RETURN_NIL;

        int rlen = (v2->i < (len - idx)) ? v2->i : (len - idx);
        // min(v2->i, (len-idx))
        if (rlen < 0) goto RETURN_NIL;

        mrbc_value value = _mrbc_string_new((char *)v->str->data + idx, rlen);
        if (!value.str) goto RETURN_NIL;		// ENOMEM

        SET_RETURN(value);
        return;		// normal return
    }
    /*
      other case
    */
    console_str("Not support such case in String#[].\n");
    return;


RETURN_NIL:
    SET_NIL_RETURN();
}

//================================================================
/*! (method) []=
 */
__GURU__
void c_string_insert(mrbc_value v[], int argc)
{
    int nth;
    int len;
    mrbc_value *val;

    /*
      in case of self[nth] = val
    */
    if (argc==2 &&
        v[1].tt==MRBC_TT_FIXNUM &&
        v[2].tt==MRBC_TT_STRING) {
        nth = v[1].i;
        len = 1;
        val = &v[2];
    }
    /*
      in case of self[nth, len] = val
    */
    else if (argc==3 &&
             v[1].tt==MRBC_TT_FIXNUM &&
             v[2].tt==MRBC_TT_FIXNUM &&
             v[3].tt==MRBC_TT_STRING) {
        nth = v[1].i;
        len = v[2].i;
        val = &v[3];
    }
    /*
      other cases
    */
    else {
        console_str("Not support\n");
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

    uint8_t *str = (uint8_t *)mrbc_realloc(VSTR(v), len1 + len2 - len + 1);
    if (!str) return;

    MEMCPY(str + nth + len2, str + nth + len, len1 - nth - len + 1);
    MEMCPY(str + nth, (uint8_t *)VSTR(val), len2);
    v->str->size = len1 + len2 - len;

    v->str->data = str;
}

//================================================================
/*! (method) chomp
 */
__GURU__
void c_string_chomp(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_dup(&v[0]);

    mrbc_string_chomp(&ret);

    SET_RETURN(ret);
}

//================================================================
/*! (method) chomp!
 */
__GURU__
void c_string_chomp_self(mrbc_value v[], int argc)
{
    if (mrbc_string_chomp(&v[0])==0) {
        SET_RETURN(mrbc_nil_value());
    }
}

//================================================================
/*! (method) dup
 */
__GURU__
void c_string_dup(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_dup(&v[0]);

    SET_RETURN(ret);
}

//================================================================
/*! (method) index
 */
__GURU__
void c_string_index(mrbc_value v[], int argc)
{
    int index;
    int offset;

    if (argc==1) {
        offset = 0;

    } else if (argc==2 && v[2].tt==MRBC_TT_FIXNUM) {
        offset = v[2].i;
        if (offset < 0) offset += VSTRLEN(&v[0]);
        if (offset < 0) goto NIL_RETURN;

    } else {
        goto NIL_RETURN;	// raise? ArgumentError
    }

    index = mrbc_string_index(&v[0], &v[1], offset);
    if (index < 0) goto NIL_RETURN;

    SET_INT_RETURN(index);
    return;

NIL_RETURN:
    SET_NIL_RETURN();
}

//================================================================
/*! (method) inspect
 */
__GURU__
void c_string_inspect(mrbc_value v[], int argc)
{
    char buf[10] = "\\x";
    mrbc_value ret = mrbc_string_new("\"");
    const unsigned char *s = (const unsigned char *)VSTR(v);
  
    for (int i = 0; i < VSTRLEN(v); i++) {
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
__GURU__
void c_string_ord(mrbc_value v[], int argc)
{
    int i = VSTR(v)[0];

    SET_INT_RETURN(i);
}

//================================================================
/*! (method) split
 */
__GURU__
void c_string_split(mrbc_value v[], int argc)
{
#if MRBC_USE_ARRAY
    mrbc_value ret = mrbc_array_new(0);
    if (mrbc_string_size(&v[0])==0) goto DONE;

    // check limit parameter.
    int limit = 0;
    if (argc >= 2) {
        if (v[2].tt != MRBC_TT_FIXNUM) {
            console_str("TypeError\n");     // raise?
            return;
        }
        limit = v[2].i;
        if (limit==1) {
            mrbc_array_push(&ret, &v[0]);
            mrbc_dup(&v[0]);
            goto DONE;
        }
    }

    // check separator parameter.
    mrb_value sep = (argc==0) ? mrbc_string_new(" ") : v[1];
    switch(sep.tt) {
    case MRBC_TT_NIL:
        sep = mrbc_string_new(" ");
        break;

    case MRBC_TT_STRING:
        break;

    default:
        console_str("TypeError\n");     // raise?
        return;
    }

    int flag_strip = (mrbc_string_cstr(&sep)[0]==' ') &&
        (_string_size(&sep)==1);
    int offset = 0;
    int sep_len = _string_size(&sep);
    if (sep_len==0) sep_len++;

    while (1) {
        int pos, len;

        if (flag_strip) {
            for (; offset < _string_size(&v[0]); offset++) {
                if (!_is_space(_string_cstr(&v[0])[offset])) break;
            }
            if (offset > _string_size(&v[0])) break;
        }

        // check limit
        if (limit > 0 && mrbc_array_size(&ret)+1 >= limit) {
            pos = -1;
            goto SPLIT_ITEM;
        }

        // split by space character.
        if (flag_strip) {
            pos = offset;
            for (; pos < _string_size(&v[0]); pos++) {
                if (_is_space(_string_cstr(&v[0])[pos])) break;
            }
            len = pos - offset;
            goto SPLIT_ITEM;
        }

        // split by each character.
        if (_string_size(&sep)==0) {
            pos = (offset < _string_size(&v[0])-1) ? offset : -1;
            len = 1;
            goto SPLIT_ITEM;
        }

        // split by specified character.
        pos = mrbc_string_index(&v[0], &sep, offset);
        len = pos - offset;

    SPLIT_ITEM:
        if (pos < 0) len = _string_size(&v[0]) - offset;

        mrb_value v1 = _mrbc_string_new(_string_cstr(&v[0]) + offset, len);
        mrbc_array_push(&ret, &v1);

        if (pos < 0) break;
        offset = pos + sep_len;
    }

    // remove trailing empty item
    if (limit==0) {
        while (1) {
            int idx = mrbc_array_size(&ret) - 1;
            if (idx < 0) break;

            mrb_value v1 = mrbc_array_get(&ret, idx);
            if (_string_size(&v1) != 0) break;

            mrbc_array_remove(&ret, idx);
            mrbc_string_delete(&v1);
        }
    }

    if (argc==0 || v[1].tt==MRBC_TT_NIL) {
        mrbc_string_delete(&sep);
    }
DONE:
#else
    mrbc_value ret = mrbc_string_new("not supported");
#endif
    SET_RETURN(ret);
}

//================================================================
/*! (method) sprintf
 */
__GURU__
void c_object_sprintf(mrbc_value v[], int argc)
{
	char buf[20];
	const char *str = guru_vprintf(buf, "<#%s:%08x>", v, argc);

    SET_RETURN(mrbc_string_new(str));
}

//================================================================
/*! (method) printf
 */
__GURU__
void c_object_printf(mrbc_value v[], int argc)
{
    c_object_sprintf(v, argc);
    console_str(VSTR(v));
    SET_NIL_RETURN();
}

//================================================================
/*! (method) lstrip
 */
__GURU__
void c_string_lstrip(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_dup(&v[0]);

    mrbc_string_strip(&ret, 0x01);	// 1: left side only

    SET_RETURN(ret);
}

//================================================================
/*! (method) lstrip!
 */
__GURU__
void c_string_lstrip_self(mrbc_value v[], int argc)
{
    if (mrbc_string_strip(&v[0], 0x01)==0) {	// 1: left side only
        SET_RETURN(mrbc_nil_value());
    }
}

//================================================================
/*! (method) rstrip
 */
__GURU__
void c_string_rstrip(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_dup(&v[0]);

    mrbc_string_strip(&ret, 0x02);	// 2: right side only

    SET_RETURN(ret);
}

//================================================================
/*! (method) rstrip!
 */
__GURU__
void c_string_rstrip_self(mrbc_value v[], int argc)
{
    if (mrbc_string_strip(&v[0], 0x02)==0) {	// 2: right side only
        SET_RETURN(mrbc_nil_value());
    }
}

//================================================================
/*! (method) strip
 */
__GURU__
void c_string_strip(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_dup(&v[0]);

    mrbc_string_strip(&ret, 0x03);	// 3: left and right

    SET_RETURN(ret);
}

//================================================================
/*! (method) strip!
 */
__GURU__
void c_string_strip_self(mrbc_value v[], int argc)
{
    if (mrbc_string_strip(&v[0], 0x03)==0) {	// 3: left and right
        SET_RETURN(mrbc_nil_value());
    }
}

//================================================================
/*! (method) to_sym
 */
__GURU__
void c_string_to_sym(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_symbol_new(VSTR(&v[0]));

    SET_RETURN(ret);
}

//================================================================
/*! initialize
 */
__GURU__
void mrbc_init_class_string()
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
