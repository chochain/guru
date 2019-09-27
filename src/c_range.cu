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
#include "object.h"		// guru_kind_of

#include "c_range.h"
#include "inspect.h"

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
    GV ret;  { ret.gt=GT_RANGE; ret.fil=0; }

    guru_range *r = ret.range = (guru_range *)guru_alloc(sizeof(guru_range));

    if (exclude_end) r->flag |= EXCLUDE_END;
    else		     r->flag &= ~EXCLUDE_END;

    r->rc    = 1;
    r->first = *first;
    r->last  = *last;
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
guru_range_cmp(const GV *v0, const GV *v1)
{
    int res;

    res = guru_cmp(&v0->range->first, &v1->range->first);
    if (res != 0) return res;

    res = guru_cmp(&v0->range->last, &v1->range->last);
    if (res != 0) return res;

    return (int)IS_EXCLUDE_END(v1->range) - (int)IS_EXCLUDE_END(v0->range);
}

//================================================================
/*! (method) ===
 */
__CFUNC__
rng_eq3(GV v[], U32 vi)
{
    if (v[0].gt == GT_CLASS) {
        GV ret = guru_kind_of(v);
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
__CFUNC__
rng_first(GV v[], U32 vi)
{
    RETURN_VAL(v->range->first);
}

//================================================================
/*! (method) last
 */
__CFUNC__
rng_last(GV v[], U32 vi)
{
    RETURN_VAL(v->range->last);
}

//================================================================
/*! (method) exclude_end?
 */
__CFUNC__
rng_exclude_end(GV v[], U32 vi)
{
    RETURN_BOOL(IS_EXCLUDE_END(v[0].range));
}

//================================================================
/*! initialize
 */
__GURU__ void
guru_init_class_range()
{
	static Vfunc vtbl[] = {
		{ "===",          rng_eq3			},
		{ "first",        rng_first			},
		{ "last",         rng_last			},
		{ "exclude_end?", rng_exclude_end	},

		{ "to_s",         gv_to_s			},
		{ "inspect",      gv_to_s			}
	};
    guru_class_range = guru_add_class(
    	"Range", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}
