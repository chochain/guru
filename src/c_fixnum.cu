/*! @file
 *
  @brief
  GURU Integer and Float class

  <pre>
  Copyright (C) 2019- GreenII.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <assert.h>

#include "guru.h"
#include "value.h"
#include "static.h"

#include "c_fixnum.h"
#include "inspect.h"

// macro to fetch from stack objects
#define ARG_INT(n)		(v[(n)].i)
#define ARG_FLOAT(n)	(v[(n)].f)

//================================================================
/*! (operator) [] bit reference
 */
__GURU__ void
int_bitref(GV v[], U32 argc)
{
    if (0 <= v[1].i && v[1].i < 32) {
        RETURN_INT((v[0].i & (1 << v[1].i)) ? 1 : 0);
    }
    else {
        RETURN_INT(0);
    }
}

//================================================================
/*! (operator) unary -
 */
__GURU__ void
int_negative(GV v[], U32 argc)
{
    GI n = ARG_INT(0);
    RETURN_INT(-n);
}

//================================================================
/*! (operator) ** power
 */
__GURU__ void
int_power(GV v[], U32 argc)
{
    if (v[1].gt == GT_INT) {
        GI x = 1;

        if (v[1].i < 0) x = 0;
        for (U32 i=0; i < v[1].i; i++) {
            x *= v[0].i;;
        }
        RETURN_INT(x);
    }

#if GURU_USE_FLOAT && GURU_USE_MATH
    else if (v[1].gt == GT_FLOAT) {
        RETURN_FLOAT(pow(v[0].i, v[1].f));
    }
#endif
}


//================================================================
/*! (operator) %
 */
__GURU__ void
int_mod(GV v[], U32 argc)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i % n);
}

//================================================================
/*! (operator) &; bit operation AND
 */
__GURU__ void
int_and(GV v[], U32 argc)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i & n);
}

//================================================================
/*! (operator) |; bit operation OR
 */
__GURU__ void
int_or(GV v[], U32 argc)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i | n);
}

//================================================================
/*! (operator) ^; bit operation XOR
 */
__GURU__ void
int_xor(GV v[], U32 argc)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i ^ n);
}

//================================================================
/*! (operator) ~; bit operation NOT
 */
__GURU__ void
int_not(GV v[], U32 argc)
{
    GI n = ARG_INT(0);
    RETURN_INT(~n);
}

//================================================================
/*! (operator) <<; bit operation LEFT_SHIFT
 */
__GURU__ void
int_lshift(GV v[], U32 argc)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i << n);
}

//================================================================
/*! (operator) >>; bit operation RIGHT_SHIFT
 */
__GURU__ void
int_rshift(GV v[], U32 argc)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i >> n);
}

//================================================================
/*! (method) abs
 */
__GURU__ void
int_abs(GV v[], U32 argc)
{
    if (v[0].i < 0) {
        v[0].i = -v[0].i;
    }
}

#if GURU_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__GURU__ void
int_to_f(GV v[], U32 argc)
{
    GF f = ARG_INT(0);
    RETURN_FLOAT(f);
}
#endif

__GURU__ void
guru_init_class_int(void)
{
    // int
    guru_class *c = guru_class_int = NEW_CLASS("int", guru_class_object);

    NEW_PROC("[]", 		int_bitref);
    NEW_PROC("-@", 		int_negative);
    NEW_PROC("**", 		int_power);
    NEW_PROC("%", 		int_mod);
    NEW_PROC("&", 		int_and);
    NEW_PROC("|", 		int_or);
    NEW_PROC("^", 		int_xor);
    NEW_PROC("~", 		int_not);
    NEW_PROC("<<", 		int_lshift);
    NEW_PROC(">>", 		int_rshift);
    NEW_PROC("abs",		int_abs);
    NEW_PROC("to_f",	int_to_f);

    NEW_PROC("chr", 	int_chr);

    NEW_PROC("to_s", 	gv_to_s);
    NEW_PROC("inspect",	gv_to_s);
}

// Float
#if GURU_USE_FLOAT
//================================================================
/*! (operator) unary -
 */
__GURU__ void
flt__negative(GV v[], U32 argc)
{
    GF f = ARG_FLOAT(0);
    RETURN_FLOAT(-f);
}

#if GURU_USE_MATH
//================================================================
/*! (operator) ** power
 */
__GURU__ void
flt__power(GV v[], U32 argc)
{
    GF n = 0;
    switch (v[1].gt) {
    case GT_INT: 	n = v[1].i;	break;
    case GT_FLOAT:	n = v[1].d;	break;
    default: break;
    }

    RETURN_FLOAT(pow(v[0].d, n));
}
#endif

//================================================================
/*! (method) abs
 */
__GURU__ void
flt_abs(GV v[], U32 argc)
{
    if (v[0].f < 0) {
        v[0].f = -v[0].f;
    }
}

//================================================================
/*! (method) to_i
 */
__GURU__ void
flt_to_i(GV v[], U32 argc)
{
    GI i = (GI)ARG_FLOAT(0);
    RETURN_INT(i);
}

//================================================================
/*! initialize class Float
 */
__GURU__ void
guru_init_class_float(void)
{
    // Float
    guru_class *c = guru_class_float = NEW_CLASS("Float", guru_class_object);

    NEW_PROC("-@", 		flt__negative);
#if GURU_USE_MATH
    NEW_PROC("**", 		flt__power);
#endif
    NEW_PROC("abs", 	flt_abs);
    NEW_PROC("to_i", 	flt_to_i);

    NEW_PROC("to_s", 	gv_to_s);
    NEW_PROC("inspect", gv_to_s);
}

#endif
