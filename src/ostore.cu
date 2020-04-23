/*! @file
  @brief
  GURU - object store implementation (as a sorted array of GV, indexed by GV.sid)

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

    GV *v = o->var;								// point at 1st attribute
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
__GURU__ U32
_resize(guru_obj *o, U32 sz)
{
    GV *v = o->var = (GV *)guru_realloc(o->var, sizeof(GV) * sz);

    if (!v) return -1;

    o->sz  = sz;

    return 0;
}

//================================================================
/*! setter

  @param  st		pointer to instance store handle.
  @param  oid		object store id (as attribute name)
  @param  val		value to be set.
  @return			0: success, -1:failed
*/
__GURU__ S32
_set(guru_obj *o, GU oid, GV *val)
{
    S32 idx = _bsearch(o, oid);
	GV  *r  = o->var;
    GV  *v  = r + idx;
    if (idx >= 0 && v->oid==oid) {
        ref_dec(v);									// replace existed attribute
        SET_VAL(v, oid, val);
        return 0;
    }
    // new attribute
    v = r + (++idx);								// use next slot
    if ((o->n+1) > o->sz) {							// need resize?
        if (_resize(o, o->sz + 4)) return -1;		// allocation, error?
        v = r + idx;
    }
    // shift attributes out for insertion
    GV *t = r + o->n;
    for (U32 i=o->n; i > idx; i--, t--) {
    	*(t) = *(t-1);
    }
    SET_VAL(v, oid, val);
    o->n++;

    return 0;
}

//================================================================
/*! getter

  @param  st	pointer to instance store handle.
  @param  oid	object store ID.
  @return		pointer to GV .
*/
__GURU__ GV*
_get(guru_obj *o, GU oid)
{
    S32 idx = _bsearch(o, oid);
    GV  *v  = o->var + idx;
    if (idx < 0 || v->oid != oid) return NULL;

    return v;
}

//================================================================
/*! guru_var constructor

  @param  cls	Pointer to Class (guru_class).
  @return       guru_ostore object with zero attribute
*/
__GURU__ GV
ostore_new(guru_class *cls)
{
    GV v; { v.gt=GT_OBJ; v.acl = ACL_HAS_REF|ACL_SELF; }

    guru_obj *o = v.self = (guru_obj *)guru_alloc(sizeof(guru_obj));

    o->rc    = 1;
    o->var  = NULL;	// attributes, lazy allocation until _set is called
    o->cls   = cls;
    o->sz    = o->n = 0;

    return v;
}

//================================================================
/*! instance variable destructor

  @param  v	pointer to target value
*/
__GURU__ void
ostore_del(GV *v)
{
	GV *r = v->self->var;

	if (r==NULL) return;

    for (U32 i=0; i<v->self->n; i++, ref_dec(r++));

    guru_free(v);
}

//================================================================
/*! instance variable setter

  @param  v		pointer to target.
  @param  oid	attribute id.
  @param  val	pointer to value.
*/
__GURU__ void
ostore_set(GV *v, GU oid, GV *val)
{
	guru_obj *o = v->self;
	if (o->var==NULL) {
		o->var = guru_gv_alloc(4);		// lazy allocation
	    o->sz   = 4;					// number of local variables
		ref_inc(v);						// itself has been referenced now
	}
	_set(o, oid, ref_inc(val));			// referenced by the object now
}

//================================================================
/*! instance variable getter

  @param  v		pointer to target.
  @param  oid	attribute id.
  @return		value.
*/
__GURU__ GV
ostore_get(GV *v, GU oid)
{
//	(v->gt==GT_CLASS) ? v->cls->var : v->self->var (common struct)
	GV *val = _get(v->self, oid);

    return val ? *ref_inc(val) : NIL;
}
