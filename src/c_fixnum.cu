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
#include "base.h"
#include "class.h"

#include "c_fixnum.h"

#include "inspect.h"

// macro to fetch from stack objects
#define _INT(n)		(v[(n)].i)
#define _FLOAT(n)	(v[(n)].f)

__GURU__ S32
guru_int_cmp(const GV *v0, const GV *v1)
{
	return -1 + (v0->i==v1->i) + (v0->i > v1->i)*2;
}

__GURU__ S32
guru_flt_cmp(const GV *v0, const GV *v1)
{
	return -1 + (v0->f==v1->f) + (v0->f > v1->f)*2;
}

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
    GI n = _INT(0);
    RETURN_INT(-n);
}

//================================================================
/*! (operator) ** power
 */
__CFUNC__
int_power(GV v[], U32 vi)
{
    ASSERT(v[1].gt==GT_INT);

    GI x = (_INT(1) < 0) ? 0 : 1;
    for (U32 i=0; i < _INT(1); i++, x *= _INT(0));

    RETURN_INT(x);

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
    GI n = _INT(1);
    RETURN_INT(v->i % n);
}

//================================================================
/*! (operator) &; bit operation AND
 */
__CFUNC__
int_and(GV v[], U32 vi)
{
    GI n = _INT(1);
    RETURN_INT(v->i & n);
}

//================================================================
/*! (operator) |; bit operation OR
 */
__CFUNC__
int_or(GV v[], U32 vi)
{
    GI n = _INT(1);
    RETURN_INT(v->i | n);
}

//================================================================
/*! (operator) ^; bit operation XOR
 */
__CFUNC__
int_xor(GV v[], U32 vi)
{
    GI n = _INT(1);
    RETURN_INT(v->i ^ n);
}

//================================================================
/*! (operator) ~; bit operation NOT
 */
__CFUNC__
int_not(GV v[], U32 vi)
{
    GI n = _INT(0);
    RETURN_INT(~n);
}

//================================================================
/*! (operator) <<; bit operation LEFT_SHIFT
 */
__CFUNC__
int_lshift(GV v[], U32 vi)
{
    GI n = _INT(1);
    RETURN_INT(v->i << n);
}

//================================================================
/*! (operator) >>; bit operation RIGHT_SHIFT
 */
__CFUNC__
int_rshift(GV v[], U32 vi)
{
    GI n = _INT(1);
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
    GF f = _INT(0);
    RETURN_FLOAT(f);
}
#endif // GURU_USE_FLOAT

__GURU__ __const__ Vfunc int_vtbl[] = {
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

	// the following functions require string, implemented in inspect.cu
	{ "chr", 	int_chr			},
	{ "to_s", 	gv_to_s			},
	{ "inspect",gv_to_s			}
};

__GURU__ void
guru_init_class_int(void)
{
    guru_rom_set_class(
    	GT_INT, "Integer", GT_OBJ, int_vtbl, sizeof(int_vtbl)/sizeof(Vfunc)
    );
    guru_register_func(GT_INT, NULL, NULL, guru_int_cmp);
}

// Float
#if GURU_USE_FLOAT
//================================================================
/*! (operator) unary -
 */
__CFUNC__
flt_negative(GV v[], U32 vi)
{
    GF f = _FLOAT(0);
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
    GI i = (GI)_FLOAT(0);
    RETURN_INT(i);
}

//================================================================
/*! initialize class Float
 */
__GURU__ __const__ Vfunc flt_vtbl[] = {
	{ "-@", 		flt_negative	},
#if     GURU_USE_MATH
	{ "**", 		flt_power		},
#endif // GURU_USE_MATH
	{ "abs", 		flt_abs			},
	{ "to_i", 		flt_to_i		},
	{ "to_s", 		gv_to_s			},
	{ "inspect", 	gv_to_s			}
};
__GURU__ void
guru_init_class_float(void)
{
    guru_rom_set_class(GT_FLOAT, "Float", GT_OBJ, flt_vtbl, VFSZ(flt_vtbl));
    guru_register_func(GT_FLOAT, NULL, NULL, guru_flt_cmp);
}

#endif // GURU_USE_FLOAT
