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
#include "c_range.h"
#include "iter.h"

__GURU__ void
_inc(GV *v, GV *s)
{
	switch (v->gt) {
	case GT_FLOAT:	v->f += s->i; 	break;
	default: 		v->i += s->i;
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
	GV s; { s.gt=GT_INT;  s.acl=0; s.fil=0; s.i=1; }
    GV v; { v.gt=GT_ITER; v.acl=0; v.fil=0; }

    guru_iter *i = v.iter = (guru_iter *)guru_alloc(sizeof(guru_iter));
    i->step = (step==NULL) ? s : *step;
    i->size = obj->gt;			// reuse the field

    switch (obj->gt) {
    case GT_RANGE: {
    	guru_range *r = obj->range;
    	GV  v    = r->first;
    	assert(v.gt==GT_INT||v.gt==GT_FLOAT);

    	i->n     = v.i;
    	i->ivar  = v;
    	i->range = *ref_inc(obj);
    } break;
    case GT_ARRAY: {
    	guru_array *a = obj->array;
    	i->n     = 1;
    	i->ivar  = *ref_inc(&a->data[0]);
    	i->range = *ref_inc(obj);

    } break;
    default: assert(1==0);			// TODO: other types not supported yet
    }
    return v;
}

// return next iterator element
//
__GURU__ GV
guru_iter_next(GV *obj)
{
	assert(obj->gt==GT_ITER);

	guru_iter *it = obj->iter;
	switch (it->size) {				// ranging object type (field reused)
	case GT_RANGE: {
		guru_range *r = it->range.range;
		_inc(&it->ivar, &it->step);
		U32 out = IS_INCLUDE(r)
			? guru_cmp(&it->ivar, &r->last) > 0
			: guru_cmp(&it->ivar, &r->last) >= 0;
		if (out) it->ivar = NIL();
	} break;
	case GT_ARRAY: {
		guru_array *a = it->range.array;
		ref_dec(&a->data[it->n-1]);	// n=1-based (nth number of elements)
		it->ivar = (it->n < a->n)
			? *ref_inc(&a->data[it->n++])
			: NIL();
	} break;
	default: assert(1==0);			// TODO: other types not supported yet
	}
	return it->ivar;
}

//================================================================
/*! destructor

  @param  target 	pointer to range object.
*/
__GURU__ void
guru_iter_del(GV *v)
{
    ref_dec(&v->iter->range);
    guru_free(v->iter);
}
