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

#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

__GURU__ guru_func _init_vtbl[GT_MAX];
__GURU__ guru_func _dest_vtbl[GT_MAX];
__GURU__ guru_func _cmp_vtbl[GT_MAX];

//================================================================
// common values
//
__GURU__ GV 	NIL()	{ GV v; { v.gt=GT_NIL;   v.acl=0; } return v; }
__GURU__ GV 	EMPTY()	{ GV v; { v.gt=GT_EMPTY; v.acl=0; } return v; }

//================================================================
// constructor/destroctor function pointers
//
__GURU__ void
guru_register_init_func(GT gt, guru_func f)
{
	_init_vtbl[gt] = f;
}

__GURU__ void
guru_register_destroy_func(GT gt, guru_func f)
{
	_dest_vtbl[gt] = f;
/*
	switch(v->gt) {
    case GT_OBJ:		guru_obj_del(v);	break;	// delete object instance
    case GT_STR:		guru_str_del(v);	break;

#if GURU_USE_ARRAY
    case GT_ARRAY:	    guru_array_del(v);	break;
    case GT_RANGE:	    guru_range_del(v);	break;
    case GT_HASH:	    guru_hash_del(v);	break;
    case GT_ITER:		guru_iter_del(v);	break;
#endif // GURU_USE_ARRAY

    default: ASSERT(1==0);
    }
    */
}

__GURU__ void
guru_register_cmp_func(GT gt, guru_func f)
{
	_cmp_vtbl[gt] = f;
}

//================================================================
/*! compare
 */
__GURU__ S32
_string_cmp(const GV *v0, const GV *v1)
{
	S32 x  = (U32)v0->str->bsz - (U32)v1->str->bsz;
	if (x) return x;

	return STRCMP(v0->str->raw, v1->str->raw);
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
    case GT_INT:
    case GT_SYM: 	return -1 + (v0->i==v1->i) + (v0->i > v1->i)*2;
#if GURU_USE_FLOAT
    case GT_FLOAT:  return -1 + (v0->f==v1->f) + (v0->f > v1->f)*2;	// caution: NaN == NaN is false
#endif // GURU_USE_FLOAT

    case GT_CLASS:
    case GT_OBJ:
    case GT_PROC:   return -1 + (v0->self==v1->self) + (v0->self > v1->self)*2;
    case GT_STR: 	return _string_cmp(v0, v1);
#if GURU_USE_ARRAY
    case GT_ARRAY:  return guru_array_cmp(v0, v1);
    case GT_RANGE:  return guru_range_cmp(v0, v1);
    case GT_HASH:   return guru_hash_cmp(v0, v1);
#endif // GURU_USE_ARRAY
    default:
        return 1;
    }
}

__GURU__ GV *
ref_get(GV *v)
{
	if (HAS_NO_REF(v) || IS_READ_ONLY(v)) return v;

	return v;
}

__GURU__ GV	*
ref_free(GV *v)
{
	if (HAS_NO_REF(v) || IS_READ_ONLY(v)) return v;

	return v;
}

//================================================================
/*!@brief
  Decrement reference counter

  @param   v     Pointer to target GV
*/
__GURU__ GV *
ref_dec(GV *v)
{
    if (HAS_NO_REF(v))  	return v;		// skip simple objects
    if (IS_READ_ONLY(v)) 	return v;		// ROMable objects?

    ASSERT(v->self->rc);					// rc > 0?
    if (--v->self->rc > 0) return v;		// still used, keep going

    _dest_vtbl[v->gt](v);

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
	if (HAS_REF(v) && !IS_READ_ONLY(v)) {
		v->self->rc++;
	}
	return v;
}



