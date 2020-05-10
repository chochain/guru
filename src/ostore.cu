/*! @file
  @brief
  GURU - object store implementation (as a sorted array of GR, indexed by GR.sid)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "guru.h"
#include "mmu.h"
#include "base.h"
#include "ostore.h"

#define SET_VAL(d, oid, val)		(*(d)=*(val), (d)->oid=(oid))
//================================================================
/*! sorted array binary search

  @param  st	pointer to instance store handle.
  @param  oid	attribute id.
  @return		result. It's not necessarily found.
*/
__GURU__ S32
_bsearch(guru_obj *o, GU oid)
{
    S32 i0 = 0;
    S32 i1 = o->n - 1;	if (i1 < 0) return -1;

    GR *v = o->var;								// point at 1st attribute
    while (i0 < i1) {
    	S32 m = (i0 + i1) >>1;					// middle i.e. div by 2
        if ((v+m)->oid < oid) {
            i0 = m + 1;
        } else {
            i1 = m;
        }
    }
    return i0;
}

//================================================================
/*! resize buffer

  @param  st	pointer to instance store handle.
  @param  size	size.
  @return		0: success, 1: failed
*/
__GURU__ GR*
_resize(GR *r0, U32 sz)
{
    return (GR*)guru_realloc(r0, sizeof(GR) * sz);
}

//================================================================
/*! setter

  @param  st		pointer to instance store handle.
  @param  oid		object store id (as attribute name)
  @param  val		value to be set.
  @return			0: success, -1:failed
*/
__GURU__ S32
_set(guru_obj *o, GU oid, GR*val)
{
    S32 idx = _bsearch(o, oid);
	GR  *v  = o->var;
    GR  *r  = v + idx;
    if (idx >= 0 && r->oid==oid) {
        ref_dec(r);									// replace existed attribute
        SET_VAL(r, oid, val);
        return 0;
    }
    // new attribute
    if ((o->n+1) > o->sz) {							// too small?
    	U32 nsz = o->sz + 4;						// expand some
        o->var = v = _resize(v, nsz);
        if (!v) return -1;
        o->sz  = nsz;
    }
    r = v + (++idx);								// use next slot

    // shift attributes out for insertion
    GR *t = v + o->n;
    for (U32 i=o->n; i > idx; i--, t--) {
    	*(t) = *(t-1);
    }
    SET_VAL(r, oid, val);
    o->n++;

    return 0;
}

//================================================================
/*! getter

  @param  st	pointer to instance store handle.
  @param  oid	object store ID.
  @return		pointer to GR .
*/
__GURU__ GR*
_get(guru_obj *o, GU oid)
{
    S32 idx = _bsearch(o, oid);
    GR  *r  = o->var + idx;
    if (idx < 0 || r->oid != oid) return NULL;

    return r;
}

//================================================================
/*! guru_var constructor

  @param  cls	Pointer to Class (guru_class).
  @return       guru_ostore object with zero attribute
*/
__GURU__ GR
ostore_new(guru_class *cls)
{
    guru_obj *o = (guru_obj *)guru_alloc(sizeof(guru_obj));

    o->rc  = 1;
    o->var = NULL;	// attributes, lazy allocation until _set is called
    o->cls = cls;
    o->sz  = o->n = 0;

    GR r; { r.gt=GT_OBJ; r.acl=ACL_HAS_REF|ACL_SELF; r.obj=MEMOFF(o); }

    return r;
}

//================================================================
/*! instance variable destructor

  @param  v	pointer to target value
*/
__GURU__ void
ostore_del(GR *r)
{
	guru_obj *o = GR_OBJ(r);
	GR       *v = o->var;

	if (v==NULL) return;

    for (U32 i=0; i<o->n; i++, ref_dec(v++));

    guru_free(r);
}

//================================================================
/*! instance variable setter

  @param  v		pointer to target.
  @param  oid	attribute id.
  @param  val	pointer to value.
*/
__GURU__ void
ostore_set(GR *r, GU oid, GR*val)
{
	guru_obj *o = GR_OBJ(r);			// NOTE: guru_obj->var, guru_class->var share the same struct
	if (o->var==NULL) {
		o->var = guru_gr_alloc(4);		// lazy allocation
	    o->sz  = 4;						// number of local variables
		ref_inc(r);						// itself has been referenced now
	}
	_set(o, oid, ref_inc(val));			// referenced by the object now
}

//================================================================
/*! instance variable getter

  @param  v		pointer to target.
  @param  oid	attribute id.
  @return		value.
*/
__GURU__ GR
ostore_get(GR *r, GU oid)
{
//	NOTE: common struct header
// 		GT_OBJ:   v->self->var
//      GT_CLASS: v->cls->var
//
	GR *val = _get(GR_OBJ(r), oid);

    return val ? *ref_inc(val) : NIL;
}
