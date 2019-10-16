/*! @file
  @brief
  GURU Iterator object

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#include <assert.h>
#include "vm_config.h"

#include "guru.h"
#include "mmu.h"
#include "static.h"
#include "value.h"

#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"
#include "iter.h"

__GURU__ void
_inc(GV *v, GV *s)
{
	S32 i = s ? s->i : 1;

	switch (v->gt) {
	case GT_FLOAT:	v->f += i; 	break;
	default: 		v->i += i;
	}
}

//================================================================
/*! constructor

  @param  obj	range or array object
  @param  step	stepping object
  @return
*/
__GURU__ GV
guru_iter_new(GV *obj, GV *step)
{
    GV v; { v.gt=GT_ITER; v.acl=ACL_HAS_REF; }

    guru_iter *i = v.iter = (guru_iter *)guru_alloc(sizeof(guru_iter));
    i->rc   = 1;
    i->meta = obj->gt;			// reuse the field
    i->step = step;

    i->range = ref_inc(obj);
    switch (obj->gt) {
    case GT_INT: {
    	i->n	 = obj->i;
    	i->ivar  = (obj->i=0, obj);
    } break;
    case GT_RANGE: {
    	guru_range 	*r = obj->range;
    	GV  		*x = &r->first;
    	assert(x->gt==GT_INT || x->gt==GT_FLOAT);

    	i->n     = x->i;
    	i->ivar  = x;
    } break;
    case GT_ARRAY: {
    	guru_array *a = obj->array;
    	i->n     = 0;
    	i->ivar  = ref_inc(a->data);
    } break;
    case GT_HASH: {
    	guru_hash *h = obj->hash;
    	i->n	 = 0;
    	i->ivar	 = ref_inc(h->data);	ref_inc(h->data+1);
    } break;
    default: assert(1==0);			// TODO: other types not supported yet
    }
    return v;
}

// return next iterator element
//
__GURU__ U32
guru_iter_next(GV *v)
{
	assert(v->gt==GT_ITER);

	guru_iter *it = v->iter;
	U32 nvar;
	switch (it->meta) {				// ranging object type (field reused)
	case GT_INT: {
		it->ivar->i += it->step ? it->step->i : 1;
		nvar = (it->ivar->i < it->n);
	} break;
	case GT_RANGE: {
		guru_range *r = it->range->range;
		_inc(it->ivar, it->step);
		U32 out = IS_INCLUDE(r)
			? guru_cmp(it->ivar, &r->last) > 0
			: guru_cmp(it->ivar, &r->last) >= 0;
		nvar = (out) ? 0 : 1;
	} break;
	case GT_ARRAY: {
		guru_array *a = it->range->array;
		GV         *d = &a->data[it->n];
		ref_dec(d);
		if ((it->n + 1) < a->n) {
			it->n += nvar = 1;
			it->ivar = ref_inc(++d);
		}
		else nvar=0;
	} break;
	case GT_HASH: {
		guru_hash *h = it->range->hash;
		GV        *d = &h->data[it->n];
		ref_dec(d);
		ref_dec(d+1);
		if ((it->n+2) < h->n) {
			it->n += nvar = 2;
			it->ivar = ref_inc(d+=2);	ref_inc(d+1);
		}
		else nvar=0;
	} break;
	default: assert(1==0);			// TODO: other types not supported yet
	}
	return nvar;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__ void
guru_iter_del(GV *v)
{
	assert(v->gt==GT_ITER);

    ref_dec(v->iter->range);
    guru_free(v->iter);
    *v = EMPTY();
}
