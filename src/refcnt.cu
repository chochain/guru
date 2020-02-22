/*! @file
  @brief
  GURU value and macro definitions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include <stdio.h>

#include "object.h"
#include "refcnt.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"
#include "iter.h"

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

