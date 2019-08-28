#include <assert.h>
#include "value.h"
#include "global.h"
/*

  GLobal objects are stored in 'mrbc_global' array.
  'mrbc_global' array is decending order by sym_id.
  In case of searching a global object, binary search is used.
  In case of adding a global object, insertion sort is used.

*/
typedef enum {
    GURU_GLOBAL_OBJECT = 1,
    GURU_CONST_OBJECT,
} _gtype;

typedef struct _gobject_ {
    mrbc_sym 		sym_id;
    mrbc_object 	obj;
    _gtype 			gtype 	:8;
} _gobject;

// max of global object in mrbc_global[]
__GURU__ U32 		_mutex_gobj;

__GURU__ U32 		_global_idx;
__GURU__ _gobject 	_global[MAX_GLOBAL_COUNT];

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
__GURU__ U32
_get_idx(mrbc_sym sid, _gtype gt)
{
    for (U32 i=0; i < _global_idx ; i++) {
        _gobject *obj = &_global[i];
        if (obj->sym_id == sid && obj->gtype == gt) return i;
    }
    return MAX_GLOBAL_COUNT;
}

__GURU__ mrbc_value
_get_obj(mrbc_sym sid, _gtype gt)
{
    U32 idx = _get_idx(sid, gt);

    if (idx==MAX_GLOBAL_COUNT) return mrbc_nil_value();

    mrbc_retain(&_global[idx].obj);
    return _global[idx].obj;
}

/* add */
/* TODO: Check reference count */
__GURU__ void
global_object_add(mrbc_sym sid, mrbc_value v)
{
    int idx = _get_idx(sid, GURU_GLOBAL_OBJECT);

    MUTEX_LOCK(_mutex_gobj);

    if (idx < MAX_GLOBAL_COUNT) {
        mrbc_release(&(_global[idx].obj));
     }
    else {
        idx = ++_global_idx;
         assert(idx < MAX_GLOBAL_COUNT);	// maybe raise ex
    }
    _global[idx].sym_id = sid;
    _global[idx].obj    = v;
    _global[idx].gtype  = GURU_GLOBAL_OBJECT;

    MUTEX_FREE(_mutex_gobj);
    
    mrbc_retain(&v);
}

__GURU__ void
const_object_add(mrbc_sym sid, mrbc_object *obj)
{
    int idx = _get_idx(sid, GURU_CONST_OBJECT);

    MUTEX_LOCK(_mutex_gobj);

    if (idx < MAX_GLOBAL_COUNT) {
        // warning: already initialized constant.
        mrbc_release(&(_global[idx].obj));
    }
    else {
        idx = ++_global_idx;
        assert(idx < MAX_GLOBAL_COUNT);	// maybe raise ex
    }

    _global[idx].sym_id = sid;
    _global[idx].obj    = *obj;
    _global[idx].gtype  = GURU_CONST_OBJECT;

    MUTEX_FREE(_mutex_gobj);

    mrbc_retain(obj);
}

/* get */
__GURU__ mrbc_value
global_object_get(mrbc_sym sid)
{
    return _get_obj(sid, GURU_GLOBAL_OBJECT);
}

/* add const */
__GURU__ mrbc_object
const_object_get(mrbc_sym sid)
{
    return _get_obj(sid, GURU_CONST_OBJECT);
}
//
__GPU__ void
guru_global_init(void)
{
	if (blockIdx.x!=0 || threadIdx.x!=0) return;

	_mutex_gobj = 0;
	_global_idx = 0;
}




