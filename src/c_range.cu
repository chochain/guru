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
#include "static.h"
#include "class.h"				// kind_of
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

    GR r { GT_RANGE, ACL_HAS_REF, 0, MEMOFF(g) };

    return r;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__ void
guru_range_del(GR *r)
{
	guru_range *rng = GR_RNG(r);

    ref_dec(&rng->first);
    ref_dec(&rng->last);

    guru_free(rng);
}

//================================================================
/*! compare
 */
__GURU__ int
guru_range_cmp(const GR *r0, const GR *r1)
{
    int res;

    guru_range *rng0 = GR_RNG(r0);
    guru_range *rng1 = GR_RNG(r1);
    res = guru_cmp(&rng0->first, &rng1->first);
    if (res != 0) return res;

    res = guru_cmp(&rng0->last, &rng1->last);
    if (res != 0) return res;

    return (int)IS_INCLUDE(rng1) - (int)IS_INCLUDE(rng0);
}

//================================================================
/*! (method) ===
 */
__CFUNC__
rng_eq3(GR r[], S32 ri)
{
    if (r->gt == GT_CLASS) {
#if GURU_CXX_CODEBASE
        GR ret = ClassMgr::getInstance()->kind_of(r);
#else
        GR ret = kind_of(r);
#endif // GURU_CXX_CODEBASE
        RETURN_VAL(ret);
    }

    guru_range *rng = GR_RNG(r);
    int first = guru_cmp(&rng->first, r+1);
    if (first <= 0) {
        RETURN_FALSE();
    }

    int last = guru_cmp(r+1, &rng->last);
    int flag = IS_INCLUDE(rng) ? (last<=0) : (last < 0);

    RETURN_BOOL(flag);
}

//================================================================
/*! (method) first
 */
__CFUNC__
rng_first(GR r[], S32 ri)
{
    RETURN_VAL(GR_RNG(r)->first);
}

//================================================================
/*! (method) last
 */
__CFUNC__
rng_last(GR r[], S32 ri)
{
    RETURN_VAL(GR_RNG(r)->last);
}

//================================================================
/*! (method) exclude_end?
 */
__CFUNC__
rng_exclude_end(GR r[], S32 ri)
{
    RETURN_BOOL(!IS_INCLUDE(GR_RNG(r)));
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
    guru_rom_add_class(GT_RANGE, "Range", GT_OBJ, rng_vtbl, VFSZ(rng_vtbl));
    guru_register_func(GT_RANGE, NULL, guru_range_del, guru_range_cmp);
}
