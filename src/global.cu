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
} mrbc_gtype;

typedef struct mrbc_gobject_ {
    mrbc_gtype 	gtype 	:8;
    mrbc_sym 	sym_id;
    mrbc_object obj;
} mrbc_gobject;

// max of global object in mrbc_global[]
__GURU__ int _mutex_glb;
__GURU__ int _global_end;
__GURU__ mrbc_gobject _mrbc_global[MAX_GLOBAL_OBJECT_SIZE];

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
__GURU__ int
_get_idx(mrbc_sym sid, mrbc_gtype gtype)
{
    for (int i=0 ; i<_global_end ; i++) {
        mrbc_gobject *obj = &_mrbc_global[i];
        if (obj->sym_id == sid && obj->gtype == gtype) return i;
    }
    return -1;
}

__GURU__ mrbc_value
_get_obj(mrbc_sym sid, mrbc_gtype gtype)
{
    int index = _get_idx(sid, gtype);
    if (index < 0) mrbc_nil_value();

    mrbc_retain(&_mrbc_global[index].obj);
    return _mrbc_global[index].obj;
}

/* add */
/* TODO: Check reference count */
__GURU__ void
global_object_add(mrbc_sym sid, mrbc_value v)
{
    int idx = _get_idx(sid, GURU_GLOBAL_OBJECT);

    MUTEX_LOCK(_mutex_glb);

    if (idx == -1) {
        idx = _global_end++;
        assert(idx < MAX_GLOBAL_OBJECT_SIZE);	// maybe raise ex
    }
    else {
        mrbc_release(&(_mrbc_global[idx].obj));
    }
    _mrbc_global[idx].gtype  = GURU_GLOBAL_OBJECT;
    _mrbc_global[idx].sym_id = sid;
    _mrbc_global[idx].obj    = v;

    MUTEX_FREE(_mutex_glb);
    
    mrbc_retain(&v);
}

__GURU__ void
const_object_add(mrbc_sym sid, mrbc_object *obj)
{
    int idx = _get_idx(sid, GURU_CONST_OBJECT);

    MUTEX_LOCK(_mutex_glb);

    if (idx == -1) {
        idx = _global_end++;
        assert(idx < MAX_GLOBAL_OBJECT_SIZE);	// maybe raise ex
    }
    else {
        // warning: already initialized constant.
        mrbc_release(&(_mrbc_global[idx].obj));
    }
    _mrbc_global[idx].gtype  = GURU_CONST_OBJECT;
    _mrbc_global[idx].sym_id = sid;
    _mrbc_global[idx].obj    = *obj;

    MUTEX_FREE(_mutex_glb);

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

	_mutex_glb  = 0;
	_global_end = 0;
}




