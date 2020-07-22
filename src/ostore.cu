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
#if !GURU_DEBUG
__GURU__ S32
_search(guru_obj *o, GS oid)
{
	GR *v = _IVAR(o);
	for (int i=0; i<o->n; i++, v++) {
		if (v->oid==oid) return i;
	}
	return -1;
}
#else
__GURU__ S32
_bsearch(guru_obj *o, GS oid)
{
    S32 i0 = 0;
    S32 i1 = o->n - 1;	if (i1<0) return -1;

    GR *v = _IVAR(o);							// point at 1st attribute
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
#endif // GURU_DEBUG

//================================================================
/*! resize buffer

  @param  st	pointer to instance store handle.
  @param  size	size.
  @return		0: success, 1: failed
*/
__GURU__ GR*
_resize(GR *r0, U32 nsz)
{
    return (GR*)guru_realloc(r0, sizeof(GR) * nsz);
}

//================================================================
/*! setter

  @param  st		pointer to instance store handle.
  @param  oid		object store id (as attribute name)
  @param  val		value to be set.
  @return			0: success, -1:failed
*/
#if !GURU_DEBUG
__GURU__ S32
_set(guru_obj *o, GS oid, GR*val)
{
    S32 idx = _search(o, oid);
	GR  *v  = _IVAR(o);
    if (idx >= 0) {
        GR  *r = v + idx;
        ref_dec(r);									// replace existed attribute
        SET_VAL(r, oid, val);
        return 0;
    }
    // new attribute
    U32 sz = o->sz;
    if ((o->n+1) > sz) {							// too small?
    	U32 nsz = sz + 4;							// fixed size expansion helps reuse blocks
        v = _resize(v, nsz);
        if (!v) return (o->ivar=0, -1);
        o->ivar = MEMOFF(v);
        o->sz  = nsz;
    }
    GR *r = v + o->n++;								// use next slot
    SET_VAL(r, oid, val);

    return 0;
}
#else
__GURU__ S32
_bset(guru_obj *o, GS oid, GR*val)
{
	S32 idx = _bsearch(o, oid);
	GR  *v  = _IVAR(o);
    GR  *r  = v + idx;
    if (idx >= 0 && r->oid==oid) {
        ref_dec(r);									// replace existed attribute
        SET_VAL(r, oid, val);
        return 0;
    }
    // new attribute
    U32 sz = o->sz;
    if ((o->n+1) > sz) {							// too small?
    	U32 nsz = sz + 4;							// fixed size expansion helps reuse blocks
        v = _resize(v, nsz);
        r = v + idx;
        if (!v) return (o->ivar=0, -1);
        o->ivar = MEMOFF(v);
        o->sz  = nsz;
    }
    // shift attributes out for insertion
    if (r->oid < oid) ++idx;						// insert to right
    GR *t = v + o->n;
    for (int i=o->n; i > idx; i--, t--) {
    	*(t) = *(t-1);
    }
    SET_VAL(v+idx, oid, val);
    o->n++;

    return 0;
}
#endif // GURU_DEBUG
//================================================================
/*! getter the following objects which shared the same structure
	GT_OBJ:   r->klass->ivar
	GT_CLASS: r->klass->ivar

  @param  st	pointer to instance store handle.
  @param  oid	object store ID.
  @return		pointer to GR .
*/
#if !GURU_DEBUG
__GURU__ GR*
_get(guru_obj *o, GS oid)
{
    S32 idx = _search(o, oid);

    return (idx>=0) ? _IVAR(o)+idx : NULL;
}
#else
__GURU__ GR*
_bget(guru_obj *o, GS oid)
{
    S32 idx = _bsearch(o, oid);
    GR  *v  = _IVAR(o) + idx;
    if (idx < 0 || v->oid != oid) return NULL;

    return v;
}
#endif // GURU_DEBUG

//================================================================
/*! guru_var constructor

  @param  ns	object/class namespace.
  @return       guru_ostore object with zero attribute
*/
__GURU__ GR
ostore_new(GP ns)
{
    guru_obj *o = (guru_obj *)guru_alloc(sizeof(guru_obj));

    o->rc    = 1;
    o->ivar  = 0;					// attributes, lazy allocation until _set is called
    o->klass = ns;					// namespace
    o->sz = o->n = 0;

    GR r { GT_OBJ, ACL_HAS_REF|ACL_TCLASS, 0, MEMOFF(o) };

    return r;
}

//================================================================
/*! instance variable destructor

  @param  v	pointer to target value
*/
__GURU__ void
ostore_del(guru_obj *o)
{
	GR *p = _IVAR(o);

    for (int i=0; i<o->n; i++, ref_dec(p++));

    if (o->ivar) guru_free(MEMPTR(o->ivar));
    guru_free(o);
}

//================================================================
/*! instance variable setter

  @param  o		pointer to guru_obj.
  @param  oid	attribute id.
  @param  val	pointer to value.
*/
__GURU__ void
ostore_set(guru_obj *o, GS oid, GR *val)
{
	if (!o->ivar) {							// NOTE: guru_obj->ivar, guru_class->ivar share the same struct
		o->ivar = MEMOFF(guru_gr_alloc(4));	// lazy allocation
	    o->sz   = 4;						// space allocated for local variables
	    o->n    = 0;						// local variable count
	}

#if !GURU_DEBUG
	_set(o, oid, ref_inc(val));				// referenced by the object now
#else
	_bset(o, oid, ref_inc(val));			// referenced by the object now
#endif // GURU_DEBUG
}

//================================================================
/*! instance variable getter

  @param  o		pointer to guru_obj.
  @param  oid	attribute id.
  @return		value.
*/
__GURU__ GR
ostore_get(guru_obj *o, GS oid)
{
#if !GURU_DEBUG
	GR *val = _get(o, oid);
#else
	GR *val = _bget(o, oid);				// get via binary search
#endif // GURU_DEBUG
    return val ? *ref_inc(val) : NIL;
}

//================================================================
/*! class instance variable getter

  @param  v		pointer to target.
  @param  oid	attribute id.
  @return		value.
*/
__GURU__ GR
ostore_getcv(guru_obj *o, GS oid)
{
	guru_class *cx = (guru_class*)o;
	GR ret { GT_NIL };
	while (cx) {
		if ((ret=ostore_get((guru_obj*)cx, oid)).gt!=GT_NIL) break;		// fetch class variable
		cx = cx->super ? _CLS(cx->super) : 0;
	}
	return ret;
}
