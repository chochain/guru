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
    guru_iter *it = (guru_iter *)guru_alloc(sizeof(guru_iter));
    it->rc   = 1;
    it->n    = obj->gt;			// reuse the field
    it->step = step;
    it->range= ref_inc(obj);

    GR r; { r.gt=GT_ITER; r.acl=ACL_HAS_REF;  r.itr=MEMOFF(it); }
    switch (obj->gt) {
    case GT_INT: {
    	it->i	= obj->i;
    	it->inc = (obj->i=0, obj);
    } break;
    case GT_RANGE: {
    	guru_range *r = GR_RNG(obj);
    	ASSERT(r->first.gt==GT_INT || r->first.gt==GT_FLOAT);

    	it->i    = 0;
    	it->inc  = guru_gr_alloc(1);
    	*it->inc = r->first;
    } break;
    case GT_ARRAY: {
    	guru_array *a = GR_ARY(obj);
    	it->i    = 0;
    	it->inc  = ref_inc(a->data);
    } break;
    case GT_HASH: {
    	guru_hash *h = GR_HSH(obj);
    	it->i	 = 0;
    	it->inc  = ref_inc(h->data);	ref_inc(h->data+1);
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

	guru_iter *it = GR_ITR(r);
	U32 nvar;
	switch (it->n) {				// ranging object type (field reused)
	case GT_INT: {
		it->inc->i += it->step ? it->step->i : 1;
		nvar = (it->inc->i < it->i);
	} break;
	case GT_RANGE: {
		guru_range *r = GR_RNG(it->range);
		U32 keep;
		if (it->inc->gt==GT_FLOAT) {
			it->inc->f += (it->step ? it->step->f : 1.0);
			keep = IS_INCLUDE(r) ? (it->inc->f <= r->last.f) : (it->inc->f < r->last.f);
		}
		else {
			it->inc->i += (it->step ? it->step->i : 1);
			keep = IS_INCLUDE(r) ? (it->inc->i <= r->last.i) : (it->inc->i < r->last.i);
		}
		nvar = (keep) ? 1 : 0;
	} break;
	case GT_ARRAY: {
		guru_array *a = GR_ARY(it->range);
		GR         *d = &a->data[it->i];
		ref_dec(d);
		if ((it->i + 1) < a->n) {
			it->i += (nvar = 1);
			it->inc = ref_inc(++d);
		}
		else nvar=0;
	} break;
	case GT_HASH: {
		guru_hash *h = GR_HSH(it->range);
		GR        *d = &h->data[it->i];
		ref_dec(d);
		ref_dec(d+1);
		if ((it->i+2) < h->n) {
			it->i += nvar = 2;
			it->inc = ref_inc(d+=2);	ref_inc(d+1);
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
	guru_iter *it = GR_ITR(r);

	if (it->n==GT_RANGE) guru_free(it->inc);

    ref_dec(it->range);
    guru_free(it);

    *r = EMPTY;
}
