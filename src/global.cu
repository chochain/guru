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
} mrbc_globaltype;

typedef struct GLOBAL_OBJECT {
    mrbc_globaltype gtype : 8;
    mrbc_sym 		sym_id;
    mrbc_object 	obj;
} mrbc_globalobject;

// max of global object in mrbc_global[]
__GURU__ int               global_end;
__GURU__ mrbc_globalobject mrbc_global[MAX_GLOBAL_OBJECT_SIZE];

/* search */
/* linear search is not efficient! */
/* TODO: Use binary search */
__GURU__ int
_get_idx(mrbc_sym sid, mrbc_globaltype gtype)
{
    for (int i=0 ; i<global_end ; i++) {
        mrbc_globalobject *obj = &mrbc_global[i];
        if (obj->sym_id == sid && obj->gtype == gtype) return i;
    }
    return -1;
}

__GURU__ mrbc_value
_get_obj(mrbc_sym sid, mrbc_globaltype gtype)
{
    int index = _get_idx(sid, gtype);
    if (index < 0) mrbc_nil_value();

    mrbc_retain(&mrbc_global[index].obj);
    return mrbc_global[index].obj;
}

//
__GURU__ void
 mrbc_init_global(void)
{
	global_end = 0;
}

/* add */
/* TODO: Check reference count */
__GURU__ void
global_object_add(mrbc_sym sid, mrbc_value v)
{
    int index = _get_idx(sid, GURU_GLOBAL_OBJECT);

    if (index == -1) {
        index = global_end++;
        assert(index < MAX_GLOBAL_OBJECT_SIZE);	// maybe raise ex
    }
    else {
        mrbc_release(&(mrbc_global[index].obj));
    }
    mrbc_global[index].gtype  = GURU_GLOBAL_OBJECT;
    mrbc_global[index].sym_id = sid;
    mrbc_global[index].obj    = v;
    
    mrbc_retain(&v);
}

__GURU__ void
const_object_add(mrbc_sym sid, mrbc_object *obj)
{
    int index = _get_idx(sid, GURU_CONST_OBJECT);

    if (index == -1) {
        index = global_end;
        global_end++;
        assert(index < MAX_GLOBAL_OBJECT_SIZE);	// maybe raise ex
    }
    else {
        // warning: already initialized constant.
        mrbc_release(&(mrbc_global[index].obj));
    }
    mrbc_global[index].gtype  = GURU_CONST_OBJECT;
    mrbc_global[index].sym_id = sid;
    mrbc_global[index].obj    = *obj;

    mrbc_retain(obj);
}

/* get */
__GURU__ mrbc_value
global_object_get(mrbc_sym sid)
{
    return _get_obj(sid, GURU_GLOBAL_OBJECT);
}

/* add const */
/* TODO: Check reference count */
/* TODO: Integrate with global_add */
__GURU__
/* get const */
/* TODO: Integrate with get_global_object */

__GURU__
mrbc_object const_object_get(mrbc_sym sid)
{
    return _get_obj(sid, GURU_CONST_OBJECT);
}



