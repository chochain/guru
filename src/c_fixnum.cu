/*! @file
 *
  @brief
  GURU Integer and Float class

  <pre>
  Copyright (C) 2019- GreenII.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "guru.h"
#include "value.h"
#include "static.h"

#include "c_fixnum.h"
#include "c_string.h"
#include "inspect.h"

// macro to fetch from stack objects
#define ARG_INT(n)		(v[(n)].i)
#define ARG_FLOAT(n)	(v[(n)].f)

//================================================================
/*! (operator) [] bit reference
 */
__CFUNC__
int_bitref(GV v[], U32 vi)
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
__CFUNC__
int_negative(GV v[], U32 vi)
{
    GI n = ARG_INT(0);
    RETURN_INT(-n);
}

//================================================================
/*! (operator) ** power
 */
__CFUNC__
int_power(GV v[], U32 vi)
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
#endif // GURU_USE_FLOAT && GURU_USE_MATH
}


//================================================================
/*! (operator) %
 */
__CFUNC__
int_mod(GV v[], U32 vi)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i % n);
}

//================================================================
/*! (operator) &; bit operation AND
 */
__CFUNC__
int_and(GV v[], U32 vi)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i & n);
}

//================================================================
/*! (operator) |; bit operation OR
 */
__CFUNC__
int_or(GV v[], U32 vi)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i | n);
}

//================================================================
/*! (operator) ^; bit operation XOR
 */
__CFUNC__
int_xor(GV v[], U32 vi)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i ^ n);
}

//================================================================
/*! (operator) ~; bit operation NOT
 */
__CFUNC__
int_not(GV v[], U32 vi)
{
    GI n = ARG_INT(0);
    RETURN_INT(~n);
}

//================================================================
/*! (operator) <<; bit operation LEFT_SHIFT
 */
__CFUNC__
int_lshift(GV v[], U32 vi)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i << n);
}

//================================================================
/*! (operator) >>; bit operation RIGHT_SHIFT
 */
__CFUNC__
int_rshift(GV v[], U32 vi)
{
    GI n = ARG_INT(1);
    RETURN_INT(v->i >> n);
}

//================================================================
/*! (method) abs
 */
__CFUNC__
int_abs(GV v[], U32 vi)
{
    if (v[0].i < 0) {
        v[0].i = -v[0].i;
    }
}

#if GURU_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__CFUNC__
int_to_f(GV v[], U32 vi)
{
    GF f = ARG_INT(0);
    RETURN_FLOAT(f);
}
#endif // GURU_USE_FLOAT

#if !GURU_USE_STRING
__CFUNC__ int_chr(GV v[], U32 vi) {}
#else
__CFUNC__
int_chr(GV v[], U32 vi)
{
    U8 buf[2] = { (U8)v->i, '\0' };

    RETURN_VAL(guru_str_new(buf));
}
#endif // GURU_USE_STRING

__GURU__ void
guru_init_class_int(void)
{
    // int
	static Vfunc vtbl[] = {
		{ "[]", 	int_bitref		},
		{ "-@", 	int_negative	},
		{ "**", 	int_power		},
		{ "%", 		int_mod			},
		{ "&", 		int_and			},
		{ "|", 		int_or			},
		{ "^", 		int_xor			},
		{ "~", 		int_not			},
		{ "<<", 	int_lshift		},
		{ ">>", 	int_rshift		},
		{ "abs",	int_abs			},
		{ "to_f",	int_to_f		},

		{ "chr", 	int_chr			},
		{ "to_s", 	gv_to_s			},
		{ "inspect",gv_to_s			}
	};
    guru_class_int = guru_add_class(
    	"int", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );

}

// Float
#if GURU_USE_FLOAT
//================================================================
/*! (operator) unary -
 */
__CFUNC__
flt_negative(GV v[], U32 vi)
{
    GF f = ARG_FLOAT(0);
    RETURN_FLOAT(-f);
}

#if GURU_USE_MATH
//================================================================
/*! (operator) ** power
 */
__CFUNC__
flt_power(GV v[], U32 vi)
{
    GF n = 0;
    switch (v[1].gt) {
    case GT_INT: 	n = v[1].i;	break;
    case GT_FLOAT:	n = v[1].d;	break;
    default: break;
    }

    RETURN_FLOAT(pow(v[0].d, n));
}
#endif // GURU_USE_MATH

//================================================================
/*! (method) abs
 */
__CFUNC__
flt_abs(GV v[], U32 vi)
{
    if (v[0].f < 0) {
        v[0].f = -v[0].f;
    }
}

//================================================================
/*! (method) to_i
 */
__CFUNC__
flt_to_i(GV v[], U32 vi)
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
	static Vfunc vtbl[] = {
		{ "-@", 		flt_negative	},
#if     GURU_USE_MATH
		{ "**", 		flt_power		},
#endif // GURU_USE_MATH
		{ "abs", 		flt_abs			},
		{ "to_i", 		flt_to_i		},
		{ "to_s", 		gv_to_s			},
		{ "inspect", 	gv_to_s			}
	};
    guru_class_float = guru_add_class(
    	"Float", guru_class_object, vtbl, sizeof(vtbl)/sizeof(Vfunc)
    );
}

#endif // GURU_USE_FLOAT
