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

#define SET_VAL(d, vid, val)		(*(d)=*(val), (d)->vid=(vid))
//================================================================
/*! sorted array binary search

  @param  st	pointer to instance store handle.
  @param  vid	attribute id.
  @return		result. It's not necessarily found.
*/
__GURU__ S32
_bsearch(guru_var *r, GS vid)
{
    S32 i0 = 0;
    S32 i1 = r->n - 1;		if (i1 < 0) return -1;

    GV *v = r->attr;							// point at 1st attribute
    while (i0 < i1) {
    	S32 m = (i0 + i1) >>1;					// middle i.e. div by 2
        if ((v+m)->vid < vid) {
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
_resize(guru_var *r, U32 sz)
{
    GV *v = (GV *)guru_realloc(r->attr, sizeof(GV) * sz);

    r->attr = v;
    r->sz   = sz;

    return 0;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  nlv	number of local variables
  @return instance store handle
*/
__GURU__ guru_var *
_new(U32 nlv)
{
    guru_var *r = (guru_var *)guru_alloc(sizeof(guru_var));

    r->rc   = 0;
    r->n    = 0;		// currently zero allocated
    r->sz   = nlv;		// number of local variables
    r->attr = guru_gv_alloc(nlv);

    return r;
}

//================================================================
/*! setter

  @param  st		pointer to instance store handle.
  @param  vid		symbol id (as attribute name)
  @param  val		value to be set.
  @return			0: success, -1:failed
*/
__GURU__ S32
_set(guru_var *r, GS vid, GV *val)
{
    S32 idx = _bsearch(r, vid);
    GV  *v  = r->attr + idx;
    if (idx >= 0 && v->vid==vid) {
        ref_dec(v);									// replace existed attribute
        SET_VAL(v, vid, val);
        return 0;
    }
    // new attribute
    v = r->attr + (++idx);							// use next slot
    if ((r->n+1) > r->sz) {						    // need resize?
        if (_resize(r, r->sz + 4)) return -1;		// allocation, error?
        v = r->attr + idx;
    }
    // shift attributes out for insertion
    GV *t = r->attr + r->n;
    for (U32 i=r->n; i > idx; i--, t--) {
    	*(t) = *(t-1);
    }
    SET_VAL(v, vid, val);
    r->n++;

    return 0;
}

//================================================================
/*! getter

  @param  st	pointer to instance store handle.
  @param  sid	symbol ID.
  @return		pointer to GV .
*/
__GURU__ GV*
_get(guru_var *r, GS vid)
{
    S32 idx = _bsearch(r, vid);
    GV  *v  = r->attr + idx;
    if (idx < 0 || v->vid != vid) return NULL;

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
    o->ivar  = NULL;	// attributes, lazy allocation until _set is called
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
	guru_var *r = v->self->ivar;

	if (r==NULL) return;

    GV *d = r->attr;
    for (U32 i=0; i<r->n; i++, ref_dec(d++));

    guru_free(r->attr);					// free physical
    guru_free(r);
}

//================================================================
/*! instance variable setter

  @param  v		pointer to target.
  @param  vid	attribute id.
  @param  val	pointer to value.
*/
__GURU__ void
ostore_set(GV *v, GS vid, GV *val)
{
	guru_var *r = v->self->ivar;		// RObj and RClass share same header
	if (r==NULL) {
		r = v->self->ivar = _new(4);	// lazy allocation
		ref_inc(v);						// itself has been referenced now
	}
	_set(r, vid, ref_inc(val));			// referenced by the object now
}

//================================================================
/*! instance variable getter

  @param  v		pointer to target.
  @param  vid	attribute id.
  @return		value.
*/
__GURU__ GV
ostore_get(GV *v, GS vid)
{
//	(v->gt==GT_CLASS) ? v->cls->ivar : v->self->ivar (common struct)
	guru_var *r = v->self->ivar;		// class or instance var
	GV 		 *val = r ? _get(r, vid) : NULL;

    return val ? *ref_inc(val) : NIL();
}
