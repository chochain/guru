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
/*! constructor

  @param  vm		pointer to VM.
  @param  first		pointer to first value.
  @param  last		pointer to last value.
  @param  flag_exclude	true: exclude the end object, otherwise include.
  @return		range object.
*/
__GURU__ mrbc_value
mrbc_range_new(mrbc_value *first, mrbc_value *last, int exclude_end)
{
    mrbc_value ret = {.tt = MRBC_TT_RANGE};

    ret.range = (mrbc_range *)mrbc_alloc(sizeof(mrbc_range));
    if (!ret.range) return ret;		// ENOMEM

    if (exclude_end) ret.range->flag |= EXCLUDE_END;
    else		     ret.range->flag &= ~EXCLUDE_END;

    ret.range->refc  = 1;
    ret.range->tt    = MRBC_TT_STRING;	// TODO: for DEBUG
    ret.range->first = *first;
    ret.range->last  = *last;

    return ret;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__ void
mrbc_range_delete(mrbc_value *v)
{
    mrbc_release(&v->range->first);
    mrbc_release(&v->range->last);

    mrbc_free(v->range);
}

//================================================================
/*! compare
 */
__GURU__ int
mrbc_range_compare(const mrbc_value *v1, const mrbc_value *v2)
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
__GURU__ void
c_range_equal3(mrbc_value v[], int argc)
{
    if (v[0].tt == MRBC_TT_CLASS) {
        mrbc_value ret = mrbc_send(v+argc, v+1, "kind_of?", 1, v);
        SET_RETURN(ret);
        return;
    }

    int first = mrbc_compare(&v->range->first, v+1);
    if (first <= 0) {
        SET_FALSE_RETURN();
        return;
    }

    int last = mrbc_compare(v+1, &v->range->last);
    int flag = IS_EXCLUDE_END(v->range) ? (last < 0) : (last <= 0);

    SET_BOOL_RETURN(flag);
}

//================================================================
/*! (method) first
 */
__GURU__ void
c_range_first(mrbc_value v[], int argc)
{
    SET_RETURN(v->range->first);
}

//================================================================
/*! (method) last
 */
__GURU__ void
c_range_last(mrbc_value v[], int argc)
{
    SET_RETURN(v->range->last);
}

//================================================================
/*! (method) exclude_end?
 */
__GURU__ void
c_range_exclude_end(mrbc_value v[], int argc)
{
    SET_BOOL_RETURN(IS_EXCLUDE_END(v[0].range));
}

#if MRBC_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__ void
c_range_inspect(mrbc_value v[], int argc)
{
    mrbc_value ret = mrbc_string_new(NULL);
    if (!ret.str) {
        SET_NIL_RETURN();
        return;
    }
    mrbc_value v1, s1;
    for (int i = 0; i < 2; i++) {
        if (i != 0) mrbc_string_append_cstr(&ret, "..");
        v1 = (i == 0) ? v->range->first : v->range->last;
        s1 = mrbc_send(v+argc, &v1, "inspect", 0);

        mrbc_string_append(&ret, &s1);
        mrbc_release(&s1);					// free locally allocated memory
    }
    SET_RETURN(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void
mrbc_init_class_range()
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
