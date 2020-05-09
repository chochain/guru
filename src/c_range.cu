/*! @file
  @brief
  GURU Range object

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#include "guru.h"
#include "mmu.h"

#include "base.h"
#include "class.h"
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
__GURU__ GR
guru_range_new(GR *first, GR *last, int inc)
{
    guru_range *g = (guru_range *)guru_alloc(sizeof(guru_range));

    g->rc    = 1;
    g->kt 	 = inc ? RANGE_INCLUDE : 0;
    g->first = *first;
    g->last  = *last;
    ref_inc(first);
    ref_inc(last);

    GR r; { r.gt=GT_RANGE; r.acl=ACL_HAS_REF; r.range=g; }

    return r;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__ void
guru_range_del(GR *r)
{
    ref_dec(&r->range->first);
    ref_dec(&r->range->last);

    guru_free(r->range);
}

//================================================================
/*! compare
 */
__GURU__ int
guru_range_cmp(const GR *r0, const GR *r1)
{
    int res;

    res = guru_cmp(&r0->range->first, &r1->range->first);
    if (res != 0) return res;

    res = guru_cmp(&r0->range->last, &r1->range->last);
    if (res != 0) return res;

    return (int)IS_INCLUDE(r1->range) - (int)IS_INCLUDE(r0->range);
}

//================================================================
/*! (method) ===
 */
__CFUNC__
rng_eq3(GR r[], U32 ri)
{
    if (r->gt == GT_CLASS) {
        GR ret = kind_of(r);
        RETURN_VAL(ret);
    }

    int first = guru_cmp(&r->range->first, r+1);
    if (first <= 0) {
        RETURN_FALSE();
    }

    int last = guru_cmp(r+1, &r->range->last);
    int flag = IS_INCLUDE(r->range) ? (last<=0) : (last < 0);

    RETURN_BOOL(flag);
}

//================================================================
/*! (method) first
 */
__CFUNC__
rng_first(GR r[], U32 ri)
{
    RETURN_VAL(r->range->first);
}

//================================================================
/*! (method) last
 */
__CFUNC__
rng_last(GR r[], U32 ri)
{
    RETURN_VAL(r->range->last);
}

//================================================================
/*! (method) exclude_end?
 */
__CFUNC__
rng_exclude_end(GR r[], U32 ri)
{
    RETURN_BOOL(!IS_INCLUDE(r->range));
}

//================================================================
/*! initialize
 */
__GURU__ __const__ Vfunc rng_vtbl[] = {
	{ "===",          rng_eq3			},
	{ "first",        rng_first			},
	{ "last",         rng_last			},
	{ "exclude_end?", rng_exclude_end	},

	{ "to_s",         gr_to_s			},
	{ "inspect",      gr_to_s			}
};

__GURU__ void
guru_init_class_range()
{
    guru_rom_set_class(GT_RANGE, "Range", GT_OBJ, rng_vtbl, VFSZ(rng_vtbl));
    guru_register_func(GT_RANGE, NULL, guru_range_del, guru_range_cmp);
}
