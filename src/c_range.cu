/*! @file
  @brief
  mruby/c Range object

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#include "vm_config.h"

#include "guru.h"
#include "alloc.h"
#include "static.h"
#include "class.h"
#include "symbol.h"

#include "console.h"
#include "opcode.h"

#include "object.h"		// mrbc_send
#include "c_range.h"
#include "c_string.h"

//================================================================
/*! get first value
 */
__GURU__
mrbc_value mrbc_range_first(const mrbc_value *v)
{
    return v->range->first;
}

//================================================================
/*! get last value
 */
__GURU__
mrbc_value mrbc_range_last(const mrbc_value *v)
{
    return v->range->last;
}

//================================================================
/*! constructor

  @param  vm		pointer to VM.
  @param  first		pointer to first value.
  @param  last		pointer to last value.
  @param  flag_exclude	true: exclude the end object, otherwise include.
  @return		range object.
*/
__GURU__
mrbc_value mrbc_range_new(mrbc_value *first, mrbc_value *last, int exclude_end)
{
    mrbc_value value = {.tt = MRBC_TT_RANGE};

    value.range = (mrbc_range *)mrbc_alloc(sizeof(mrbc_range));
    if (!value.range) return value;		// ENOMEM

    if (exclude_end) value.range->flag |= EXCLUDE_END;
    else		     value.range->flag &= ~EXCLUDE_END;
    value.range->refc  = 1;
    value.range->tt    = MRBC_TT_STRING;	// TODO: for DEBUG
    value.range->first = *first;
    value.range->last  = *last;

    return value;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__
void mrbc_range_delete(mrbc_value *v)
{
    mrbc_release(&v->range->first);
    mrbc_release(&v->range->last);

    mrbc_free(v->range);
}

//================================================================
/*! compare
 */
__GURU__
int mrbc_range_compare(const mrbc_value *v1, const mrbc_value *v2)
{
    int res;

    res = mrbc_compare(&v1->range->first, &v2->range->first);
    if (res != 0) return res;

    res = mrbc_compare(&v1->range->last, &v2->range->last);
    if (res != 0) return res;

    return (int)IS_EXCLUDE_END(v2->range) - (int)IS_EXCLUDE_END(v1->range);
}

//================================================================
/*! (method) ===
 */
__GURU__
void c_range_equal3(mrbc_value v[], int argc)
{
    if (v[0].tt == MRBC_TT_CLASS) {
        mrbc_value result = mrbc_send(v+argc, &v[1], "kind_of?", 1, &v[0]);
        SET_RETURN(result);
        return;
    }

    int cmp_first = mrbc_compare(&v[0].range->first, &v[1]);
    int result = (cmp_first <= 0);
    if (!result) {
        SET_BOOL_RETURN(result);
        return;
    }

    int cmp_last  = mrbc_compare(&v[1], &v[0].range->last);
    result = IS_EXCLUDE_END(v[0].range) ? (cmp_last < 0) : (cmp_last <= 0);

    SET_BOOL_RETURN(result);
}

//================================================================
/*! (method) first
 */
__GURU__
void c_range_first(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_range_first(v);
    SET_RETURN(ret);
}

//================================================================
/*! (method) last
 */
__GURU__
void c_range_last(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_range_last(v);
    SET_RETURN(ret);
}

//================================================================
/*! (method) exclude_end?
 */
__GURU__
void c_range_exclude_end(mrbc_value v[], int argc)
{
    int result = IS_EXCLUDE_END(v[0].range);
    SET_BOOL_RETURN(result);
}

#if MRBC_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__
void c_range_inspect(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_new(NULL);
    if (!ret.str) {
        SET_NIL_RETURN();
        return;
    }

    for (int i = 0; i < 2; i++) {
        if (i != 0) mrbc_string_append_cstr(&ret, "..");
        mrbc_value v1 = (i == 0) ? mrbc_range_first(v) : mrbc_range_last(v);
        mrbc_value s1 = mrbc_send(v+argc, &v1, "inspect", 0);
        mrbc_string_append(&ret, &s1);
        mrbc_string_delete(&s1);
    }

    SET_RETURN(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__
void mrbc_init_class_range()
{
    mrbc_class *c = mrbc_class_range = mrbc_define_class("Range", mrbc_class_object);

    mrbc_define_method(c, "===",          c_range_equal3);
    mrbc_define_method(c, "first",        c_range_first);
    mrbc_define_method(c, "last",         c_range_last);
    mrbc_define_method(c, "exclude_end?", c_range_exclude_end);

#if MRBC_USE_STRING
    mrbc_define_method(c, "inspect",      c_range_inspect);
    mrbc_define_method(c, "to_s",         c_range_inspect);
#endif
}
