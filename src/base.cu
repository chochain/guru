/*! @file
  @brief
  GURU common values and constructor/destructor/comparator registry

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include "util.h"
#include "base.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

__GURU__ guru_init_func 	_i_vtbl[GT_MAX];
__GURU__ guru_destroy_func 	_d_vtbl[GT_MAX];
__GURU__ guru_cmp_func 		_c_vtbl[GT_MAX];

//================================================================
// common values, forwarded in guru.h
//
__GURU__ GV NIL   = { .gt=GT_NIL,   .acl=0 };
__GURU__ GV EMPTY = { .gt=GT_EMPTY, .acl=0 };

//================================================================
// constructor/destroctor function pointers
//
__GURU__ void
guru_register_func(GT t, guru_init_func fi, guru_destroy_func fd, guru_cmp_func fc)
{
	_i_vtbl[t] = fi;
	_d_vtbl[t] = fd;
	_c_vtbl[t] = fc;
}

__GURU__ void
guru_destroy(GV *v)
{
	_d_vtbl[v->gt](v);
}

//================================================================
/*! compare two GVs

  @param  v1	Pointer to GV
  @param  v2	Pointer to another GV
  @retval 0	v1 == v2
  @retval plus	v1 >  v2
  @retval minus	v1 <  v2
*/
__GURU__ S32
guru_cmp(const GV *v0, const GV *v1)
{
    if (v0->gt != v1->gt) { 						// GT different
#if GURU_USE_FLOAT
    	GF f0, f1;

        if (v0->gt==GT_INT && v1->gt==GT_FLOAT) {
            f0 = v0->i;
            f1 = v1->f;
            return -1 + (f0 == f1) + (f0 > f1)*2;	// caution: NaN == NaN is false
        }
        if (v0->gt==GT_FLOAT && v1->gt==GT_INT) {
            f0 = v0->f;
            f1 = v1->i;
            return -1 + (f0 == f1) + (f0 > f1)*2;	// caution: NaN == NaN is false
        }
#endif // GURU_USE_FLOAT
        // leak Empty?
        if ((v0->gt==GT_EMPTY && v1->gt==GT_NIL) ||
            (v0->gt==GT_NIL   && v1->gt==GT_EMPTY)) return 0;

        // other case
        return v0->gt - v1->gt;
    }

    // check value
    switch(v1->gt) {
    case GT_NIL:
    case GT_FALSE:
    case GT_TRUE:   return 0;
    case GT_SYM: 	return -1 + (v0->i==v1->i) + (v0->i > v1->i)*2;
    case GT_CLASS:
    case GT_OBJ:
    case GT_PROC:   return -1 + (v0->self==v1->self) + (v0->self > v1->self)*2;
    default:
    	return _c_vtbl[v1->gt](v0, v1);
    }
}

//================================================================
/*!@brief
  Decrement reference counter

  @param   v     Pointer to target GV
*/
__GURU__ GV *
ref_dec(GV *v)
{
    if (HAS_NO_REF(v))  	return v;		// skip simple or ROMable objects

    ASSERT(v->self->rc);					// rc > 0?
    if (--v->self->rc > 0) return v;		// still used, keep going

    _d_vtbl[v->gt](v);						// table driven (no branch divergence)

    return v;
}

//================================================================
/*!@brief
  Duplicate GV

  @param   v     Pointer to GV
*/
__GURU__ GV *
ref_inc(GV *v)
{
	if (HAS_REF(v)) {						// TODO: table lookup reduce branch divergence
		v->self->rc++;
	}
	return v;
}



