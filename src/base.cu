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
#include "mmu.h"

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
__GURU__ GR NIL   = { .gt=GT_NIL,   .acl=0 };
__GURU__ GR EMPTY = { .gt=GT_EMPTY, .acl=0 };

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
guru_destroy(GR *r)
{
	_d_vtbl[r->gt](r);
}

//================================================================
/*! compare two GRs

  @param  r0	Pointer to GR
  @param  r1	Pointer to another GR
  @retval 0	r0 == r1
  @retval plus	r0 >  r1
  @retval minus	r0 <  r1
*/
__GURU__ S32
guru_cmp(const GR *r0, const GR *r1)
{
    if (r0->gt != r1->gt) { 						// GT different
#if GURU_USE_FLOAT
    	GF f0, f1;

        if (r0->gt==GT_INT && r1->gt==GT_FLOAT) {
            f0 = r0->i;
            f1 = r1->f;
            return -1 + (f0==f1) + (f0 > f1)*2;		// caution: NaN == NaN is false
        }
        if (r0->gt==GT_FLOAT && r1->gt==GT_INT) {
            f0 = r0->f;
            f1 = r1->i;
            return -1 + (f0==f1) + (f0 > f1)*2;		// caution: NaN == NaN is false
        }
#endif // GURU_USE_FLOAT
        // leak Empty?
        if ((r0->gt==GT_EMPTY && r1->gt==GT_NIL) ||
            (r0->gt==GT_NIL   && r1->gt==GT_EMPTY)) return 0;

        // other case
        return r0->gt - r1->gt;
    }

    // check value
    switch(r1->gt) {
    case GT_NIL:
    case GT_FALSE:
    case GT_TRUE:   return 0;
    case GT_SYM:
    case GT_OBJ:
    case GT_PROC: 	return -1 + (r0->i==r1->i) + (r0->i > r1->i)*2;				// 32-bit offset
    case GT_CLASS:	return -1 + (r0->cls ==r1->cls)  + (r0->cls  > r1->cls) *2;
    default:
    	return _c_vtbl[r1->gt](r0, r1);
    }
}

//================================================================
/*!@brief
  Decrement reference counter

  @param   r     Pointer to target GR
*/
__GURU__ GR*
ref_dec(GR *r)
{
    if (HAS_NO_REF(r))     return r;		// skip simple or ROMable objects

    ASSERT(GR_OFF(r)->rc);					// rc > 0?
    if (--GR_OFF(r)->rc > 0) return r;		// still used, keep going

    _d_vtbl[r->gt](r);						// table driven (no branch divergence)

    return r;
}

//================================================================
/*!@brief
  Duplicate GR

  @param   r     Pointer to GR
*/
__GURU__ GR *
ref_inc(GR *r)
{
	if (HAS_REF(r)) {						// TODO: table lookup reduce branch divergence
		GR_OFF(r)->rc++;
	}
	return r;
}



