/*! @file
  @brief
  GURU Range object

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#include <assert.h>
#include "vm_config.h"

#include "guru.h"
#include "mmu.h"
#include "class.h"
#include "value.h"
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
guru_range_new(GV *first, GV *last, int inc)
{
    GV v;  { v.gt=GT_RANGE; v.acl=ACL_HAS_REF; }

    guru_range *r = v.range = (guru_range *)guru_alloc(sizeof(guru_range));

    if (inc) r->flag |= RANGE_INCLUDE;
    else	 r->flag &= ~RANGE_INCLUDE;

    r->rc    = 1;
    r->first = *first;
    r->last  = *last;
    ref_inc(first);
    ref_inc(last);

    return v;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__ void
guru_range_del(GV *v)
{
    ref_dec(&v->range->first);
    ref_dec(&v->range->last);

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

    return (int)IS_INCLUDE(v1->range) - (int)IS_INCLUDE(v0->range);
}

//================================================================
/*! (method) ===
 */
__CFUNC__
rng_eq3(GV v[], U32 vi)
{
    if (v->gt == GT_CLASS) {
        GV ret = guru_kind_of(v);
        RETURN_VAL(ret);
    }

    int first = guru_cmp(&v->range->first, v+1);
    if (first <= 0) {
        RETURN_FALSE();
    }

    int last = guru_cmp(v+1, &v->range->last);
    int flag = IS_INCLUDE(v->range) ? (last<=0) : (last < 0);

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
    RETURN_BOOL(!IS_INCLUDE(v->range));
}

//================================================================
/*! initialize
 */
__GURU__ __const__ Vfunc rng_vtbl[] = {
	{ "===",          rng_eq3			},
	{ "first",        rng_first			},
	{ "last",         rng_last			},
	{ "exclude_end?", rng_exclude_end	},

	{ "to_s",         gv_to_s			},
	{ "inspect",      gv_to_s			}
};

__GURU__ void
guru_init_class_range()
{
    guru_rom_set_class(GT_RANGE, "Range", GT_OBJ, rng_vtbl, VFSZ(rng_vtbl));
}
