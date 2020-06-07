/*! @file
  @brief
  GURU Iterator object

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "guru.h"
#include "mmu.h"
#include "base.h"

#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

#include "iter.h"

//================================================================
/*! constructor

  @param  obj	range or array object
  @param  step	stepping object
  @return
*/
__GURU__ GR
guru_iter_new(GR *obj, GR *step)
{
    guru_iter *ix = (guru_iter *)guru_alloc(sizeof(guru_iter));
    ix->rc   = 1;
    ix->n    = obj->gt;			// reuse the field
    ix->step = step;
    ix->range= ref_inc(obj);

    GR r { GT_ITER, ACL_HAS_REF, 0, MEMOFF(ix) };
    switch (obj->gt) {
    case GT_INT: {
    	ix->i	= obj->i;
    	ix->inc = (obj->i=0, obj);
    } break;
    case GT_RANGE: {
    	guru_range *r = GR_RNG(obj);
    	ASSERT(r->first.gt==GT_INT || r->first.gt==GT_FLOAT);

    	ix->i    = 0;
    	ix->inc  = guru_gr_alloc(1);
    	*ix->inc = r->first;
    } break;
    case GT_ARRAY: {
    	guru_array *a = GR_ARY(obj);
    	ix->i    = 0;
    	ix->inc  = ref_inc(a->data);
    } break;
    case GT_HASH: {
    	guru_hash *h = GR_HSH(obj);
    	ix->i	 = 0;
    	ix->inc  = ref_inc(h->data);	ref_inc(h->data+1);
    } break;
    default: ASSERT(1==0);			// TODO: other types not supported yet
    }
    return r;
}

// return next iterator element
//
__GURU__ U32
guru_iter_next(GR *r)
{
	ASSERT(r->gt==GT_ITER);

	guru_iter *ix = GR_ITR(r);
	U32 nvar;
	switch (ix->n) {				// ranging object type (field reused)
	case GT_INT: {
		ix->inc->i += ix->step ? ix->step->i : 1;
		nvar = (ix->inc->i < ix->i);
	} break;
	case GT_RANGE: {
		guru_range *r = GR_RNG(ix->range);
		U32 keep;
		if (ix->inc->gt==GT_FLOAT) {
			ix->inc->f += (ix->step ? ix->step->f : 1.0f);
			keep = IS_INCLUDE(r) ? (ix->inc->f <= r->last.f) : (ix->inc->f < r->last.f);
		}
		else {
			ix->inc->i += (ix->step ? ix->step->i : 1);
			keep = IS_INCLUDE(r) ? (ix->inc->i <= r->last.i) : (ix->inc->i < r->last.i);
		}
		nvar = (keep) ? 1 : 0;
	} break;
	case GT_ARRAY: {
		guru_array *a = GR_ARY(ix->range);
		GR         *d = &a->data[ix->i];
		ref_dec(d);
		if ((ix->i + 1) < a->n) {
			ix->i += 1;
			ix->inc = ref_inc(++d);
			nvar  = 1;
		}
		else nvar=0;
	} break;
	case GT_HASH: {
		guru_hash *h = GR_HSH(ix->range);
		GR        *d = &h->data[ix->i];
		ref_dec(d);
		ref_dec(d+1);
		if ((ix->i+2) < h->n) {
			ix->i += nvar = 2;
			ix->inc = ref_inc(d+=2);	ref_inc(d+1);
		}
		else nvar=0;
	} break;
	default: ASSERT(1==0);			// TODO: other types not supported yet
	}
	return nvar;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__ void
guru_iter_del(GR *r)
{
	ASSERT(r->gt==GT_ITER);
	guru_iter *ix = GR_ITR(r);

	if (ix->n==GT_RANGE) guru_free(ix->inc);

    ref_dec(ix->range);
    guru_free(ix);

    *r = EMPTY;
}
