/*! @file
  @brief
  GURU Range object

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#include "vm_config.h"

#include "guru.h"
#include "alloc.h"
#include "static.h"
#include "symbol.h"

#include "ucode.h"
#include "object.h"		// guru_kind_of, guru_inspect

#include "c_range.h"
#include "c_string.h"

#include "puts.h"

//================================================================
/*! constructor

  @param  vm		pointer to VM.
  @param  first		pointer to first value.
  @param  last		pointer to last value.
  @param  flag_exclude	true: exclude the end object, otherwise include.
  @return		range object.
*/
__GURU__ GV
guru_range_new(GV *first, GV *last, int exclude_end)
{
    GV ret = {.gt = GT_RANGE, .rc = 1 };

    ret.range = (guru_range *)guru_alloc(sizeof(guru_range));

    if (exclude_end) ret.range->flag |= EXCLUDE_END;
    else		     ret.range->flag &= ~EXCLUDE_END;

    ret.range->first = *first;
    ret.range->last  = *last;
    ref_inc(first);
    ref_inc(last);

    return ret;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__ void
guru_range_del(GV *v)
{
    ref_clr(&v->range->first);
    ref_clr(&v->range->last);

    guru_free(v->range);
}

//================================================================
/*! compare
 */
__GURU__ int
guru_range_cmp(const GV *v1, const GV *v2)
{
    int res;

    res = guru_cmp(&v1->range->first, &v2->range->first);
    if (res != 0) return res;

    res = guru_cmp(&v1->range->last, &v2->range->last);
    if (res != 0) return res;

    return (int)IS_EXCLUDE_END(v2->range) - (int)IS_EXCLUDE_END(v1->range);
}

//================================================================
/*! (method) ===
 */
__GURU__ void
rng_eq3(GV v[], U32 argc)
{
    if (v[0].gt == GT_CLASS) {
        GV ret = guru_kind_of(v, argc);
        RETURN_VAL(ret);
    }

    int first = guru_cmp(&v->range->first, v+1);
    if (first <= 0) {
        RETURN_FALSE();
    }

    int last = guru_cmp(v+1, &v->range->last);
    int flag = IS_EXCLUDE_END(v->range) ? (last < 0) : (last <= 0);

    RETURN_BOOL(flag);
}

//================================================================
/*! (method) first
 */
__GURU__ void
rng_first(GV v[], U32 argc)
{
    RETURN_VAL(v->range->first);
}

//================================================================
/*! (method) last
 */
__GURU__ void
rng_last(GV v[], U32 argc)
{
    RETURN_VAL(v->range->last);
}

//================================================================
/*! (method) exclude_end?
 */
__GURU__ void
rng_exclude_end(GV v[], U32 argc)
{
    RETURN_BOOL(IS_EXCLUDE_END(v[0].range));
}

#if GURU_USE_STRING
//================================================================
/*! (method) inspect
 */
__GURU__ void
rng_inspect(GV v[], U32 argc)
{
    GV ret = guru_str_new(NULL);
    if (!ret.str) {
        RETURN_NIL();
    }
    GV v1, s1;
    for (U32 i=0; i<2; i++) {
        if (i != 0) guru_str_append_cstr(&ret, (U8P)"..");
        v1 = (i == 0) ? v->range->first : v->range->last;
        s1 = guru_inspect(v+argc, &v1);

        guru_str_append(&ret, &s1);
        ref_clr(&s1);					// free locally allocated memory
    }
    RETURN_VAL(ret);
}
#endif

//================================================================
/*! initialize
 */
__GURU__ void
guru_init_class_range()
{
    guru_class *c = guru_class_range = NEW_CLASS("Range", guru_class_object);

    NEW_PROC("===",          rng_eq3);
    NEW_PROC("first",        rng_first);
    NEW_PROC("last",         rng_last);
    NEW_PROC("exclude_end?", rng_exclude_end);

#if GURU_USE_STRING
    NEW_PROC("inspect",      rng_inspect);
    NEW_PROC("to_s",         rng_inspect);
#endif
}
