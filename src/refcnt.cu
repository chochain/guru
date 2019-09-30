/*! @file
  @brief
  GURU value and macro definitions

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#include <assert.h>
#include "object.h"
#include "refcnt.h"

#include "c_string.h"
#include "c_array.h"
#include "c_hash.h"
#include "c_range.h"

//================================================================
/*!@brief
  Decrement reference counter

  @param   v     Pointer to target GV
*/
__GURU__ GV *
ref_dec(GV *v)
{
    if (!(v->acl & ACL_HAS_REF)) return v;	// simple objects
    if (v->acl & ACL_READ_ONLY)  return v;	// ROMable objects

    assert(v->self->rc);					// rc > 0
    if (--v->self->rc > 0) return v;		// still used, keep going

    switch(v->gt) {
    case GT_OBJ:		guru_obj_del(v);	break;	// delete object instance
#if GURU_USE_STRING
    case GT_STR:		guru_str_del(v);	break;
#endif // GURU_USE_STRING

#if GURU_USE_ARRAY
    case GT_ARRAY:	    guru_array_del(v);	break;
    case GT_RANGE:	    guru_range_del(v);	break;
    case GT_HASH:	    guru_hash_del(v);	break;
#endif // GURU_USE_ARRAY

    default: break;
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
	if ((v->acl&ACL_HAS_REF) && !(v->acl&ACL_READ_ONLY)) {
		v->self->rc++;
	}
	return v;
}

