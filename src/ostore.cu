/*! @file
  @brief
  GURU - object store implementation (as a sorted array of GV, indexed by GV.sid)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "mmu.h"
#include "value.h"
#include "ostore.h"

#define SET_VAL(d, val, sid)		(*d = *val, d->sid = sid)
//================================================================
/*! sorted array binary search

  @param  st	pointer to instance store handle.
  @param  sid	symbol ID.
  @return		result. It's not necessarily found.
*/
__GURU__ S32
_bsearch(guru_obj *o, GS sid)
{
    int p0 = 0;
    int p1 = o->n - 1;		if (p1 <= 0) return -1;

    GV *v = o->data;
    while (p0 < p1) {
        int m = (p0 + p1) >> 1; 		// middle i.e. div by 2
        if ((v+m)->sid < sid) {
            p0 = m + 1;
        } else {
            p1 = m;
        }
    }
    return p0;
}

//================================================================
/*! resize buffer

  @param  st	pointer to instance store handle.
  @param  size	size.
  @return		0: success, 1: failed
*/
__GURU__ U32
_resize(guru_obj *o, U32 size)
{
    GV *d2 = (GV *)guru_realloc(o->data, sizeof(GV) * size);

    o->data = d2;
    o->size = size;

    return 0;
}

//================================================================
/*! constructor

  @param  vm	pointer to VM.
  @param  nlv	number of local variables
  @return instance store handle
*/
__GURU__ guru_obj *
_new(U32 nlv)
{
    guru_obj *o = (guru_obj *)guru_alloc(sizeof(guru_obj));

    o->data = (GV *)guru_alloc(sizeof(GV) * nlv);
    o->size = nlv;		// number of local variables
    o->n    = 0;		// currently zero allocated

    return o;
}

//================================================================
/*! destructor

  @param  st	pointer to instance store handle.
*/
__GURU__ void
_del(guru_obj *o)
{
	if (o==NULL) return;

    GV *d = o->data;
    for (U32 i=0; i<o->n; i++, ref_dec(d++));

    guru_free(o->data);					// free physical
    guru_free(o);
}

//================================================================
/*! setter

  @param  st		pointer to instance store handle.
  @param  sid		symbol id (as attribute name)
  @param  val		value to be set.
  @return			0: success, -1:failed
*/
__GURU__ S32
_set(guru_obj *o, GS sid, GV *val)
{
    S32 idx = _bsearch(o, sid);
    GV *d   = o->data + idx;
    if (idx >= 0) {
        ref_dec(d);					// replace existed attribute
        SET_VAL(d, val, sid);
        return 0;
    }
    // new attribute
    if (d->sid < sid) idx++;
    if ((o->n+1) >= o->size) {							// need resize?
        if (_resize(o, o->size + 4)) return -1;			// allocation, error?
    }
    if (idx < o->n) {									// need more data?
    	GV *t = o->data + o->n;
    	for (U32 i=o->n; i > idx; i--, t--) {			// shift out for insertion
    		*(t) = *(t-1);
    	}
    }
    SET_VAL(d, val, sid);
    o->n++;

    return 0;
}

//================================================================
/*! getter

  @param  st	pointer to instance store handle.
  @param  sid	symbol ID.
  @return		pointer to GV .
*/
__GURU__ GV*
_get(guru_obj *o, GS sid)
{
    S32 idx = _bsearch(o, sid);
    if (idx < 0) return NULL;

    GV *d = o->data + idx;
    if (d->sid != sid) return NULL;

    return d;
}

//================================================================
/*! guru_obj constructor

  @param  vm    Pointer to VM.
  @param  cls	Pointer to Class (guru_class).
  @param  nlv	number of local variables
  @return       guru_ostore object.
*/
__GURU__ GV
ostore_new(guru_class *cls)
{
    GV v; { v.gt=GT_OBJ; v.acl = ACL_HAS_REF; v.fil=0xffffffff; }

    guru_var *r = v.self = (guru_var *)guru_alloc(sizeof(guru_var));

    r->rc    = 1;
    r->ivar  = NULL;	// lazy allocation until _set is called
    r->cls   = cls;

    return v;
}

//================================================================
/*! instance variable destructor

  @param  v	pointer to target value
*/
__GURU__ void
ostore_del(GV *v)
{
    _del(v->self->ivar);
    guru_free(v->self);
}

//================================================================
/*! instance variable setter

  @param  obj	pointer to target.
  @param  sid	key symbol ID.
  @param  v		pointer to value.
*/
__GURU__ void
ostore_set(GV *v, GS sid, GV *val)
{
	U32 is_o = v->gt==GT_OBJ;
	guru_obj *o = is_o ? v->self->ivar : v->cls->cvar;
	if (o==NULL) {		// lazy allocation
		o = is_o
			? (v->self->ivar = _new(1))
			: (v->cls->cvar  = _new(1));
	}
	_set(o, sid, val);
    ref_inc(val);			// referenced by the object now
}

//================================================================
/*! instance variable getter

  @param  obj	pointer to target.
  @param  sid	key symbol ID.
  @return		value.
*/
__GURU__ GV
ostore_get(GV *v, GS sid)
{
	guru_obj *o = (v->gt==GT_OBJ) ? v->self->ivar : v->cls->cvar;

	GV *val = _get(o, sid);

    return val ? *val : NIL();
}

