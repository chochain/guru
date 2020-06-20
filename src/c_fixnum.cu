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
#include "static.h"

#include "c_fixnum.h"

#include "inspect.h"

// macro to fetch from stack objects
#define _INT(n)		(r[(n)].i)
#define _FLOAT(n)	(r[(n)].f)

__GURU__ S32
guru_int_cmp(const GR *r0, const GR *r1)
{
	return -1 + (r0->i==r1->i) + (r0->i > r1->i)*2;
}

__GURU__ S32
guru_flt_cmp(const GR *r0, const GR *r1)
{
	return -1 + (r0->f==r1->f) + (r0->f > r1->f)*2;
}

//================================================================
/*! (operator) [] bit reference
 */
__CFUNC__
int_bitref(GR r[], S32 ri)
{
    if (0 <= r[1].i && r[1].i < 32) {
        RETURN_INT((r[0].i & (1 << r[1].i)) ? 1 : 0);
    }
    else {
        RETURN_INT(0);
    }
}

//================================================================
/*! (operator) unary -
 */
__CFUNC__
int_negative(GR r[], S32 ri)
{
    GI n = _INT(0);
    RETURN_INT(-n);
}

//================================================================
/*! (operator) ** power
 */
__CFUNC__
int_power(GR r[], S32 ri)
{
    ASSERT(r[1].gt==GT_INT);

    GI x = (_INT(1) < 0) ? 0 : 1;
    for (int i=0; i < _INT(1); i++, x *= _INT(0));

    RETURN_INT(x);

#if GURU_USE_FLOAT && GURU_USE_MATH
    else if (r[1].gt == GT_FLOAT) {
        RETURN_FLOAT(pow(r[0].i, r[1].f));
    }
#endif // GURU_USE_FLOAT && GURU_USE_MATH
}


//================================================================
/*! (operator) %
 */
__CFUNC__
int_mod(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i % n);
}

//================================================================
/*! (operator) &; bit operation AND
 */
__CFUNC__
int_and(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i & n);
}

//================================================================
/*! (operator) |; bit operation OR
 */
__CFUNC__
int_or(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i | n);
}

//================================================================
/*! (operator) ^; bit operation XOR
 */
__CFUNC__
int_xor(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i ^ n);
}

//================================================================
/*! (operator) ~; bit operation NOT
 */
__CFUNC__
int_not(GR r[], S32 ri)
{
    GI n = _INT(0);
    RETURN_INT(~n);
}

//================================================================
/*! (operator) <<; bit operation LEFT_SHIFT
 */
__CFUNC__
int_lshift(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i << n);
}

//================================================================
/*! (operator) >>; bit operation RIGHT_SHIFT
 */
__CFUNC__
int_rshift(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i >> n);
}

//================================================================
/*! (operator) %
 */
__CFUNC__
int_mod_set(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i %= n);
}

//================================================================
/*! (operator) &; bit operation AND
 */
__CFUNC__
int_and_set(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i &= n);
}

//================================================================
/*! (operator) |; bit operation OR
 */
__CFUNC__
int_or_set(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i |= n);
}

//================================================================
/*! (operator) ^; bit operation XOR
 */
__CFUNC__
int_xor_set(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i ^= n);
}

//================================================================
/*! (operator) <<; bit operation LEFT_SHIFT
 */
__CFUNC__
int_lshift_set(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i <<= n);
}

//================================================================
/*! (operator) >>; bit operation RIGHT_SHIFT
 */
__CFUNC__
int_rshift_set(GR r[], S32 ri)
{
    GI n = _INT(1);
    RETURN_INT(r->i >>= n);
}

//================================================================
/*! (method) abs
 */
__CFUNC__
int_abs(GR r[], S32 ri)
{
    if (r[0].i < 0) {
        r[0].i = -r[0].i;
    }
}

#if GURU_USE_FLOAT
//================================================================
/*! (method) to_f
 */
__CFUNC__
int_to_f(GR r[], S32 ri)
{
    GF f = _INT(0);
    RETURN_FLOAT(f);
}
#endif // GURU_USE_FLOAT

__GURU__ __const__ Vfunc int_mtbl[] = {
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
	{ "%=", 	int_mod_set		},
	{ "&=", 	int_and_set		},
	{ "|=", 	int_or_set		},
	{ "^=", 	int_xor_set		},
	{ "<<=", 	int_lshift_set	},
	{ ">>=", 	int_rshift_set	},
	{ "abs",	int_abs			},
	{ "to_f",	int_to_f		},

	// the following functions require string, implemented in inspect.cu
	{ "chr", 	int_chr			},
	{ "to_s", 	gr_to_s			},
	{ "inspect",gr_to_s			}
};

__GURU__ void
guru_init_class_int(void)
{
    guru_rom_add_class(
    	GT_INT, "Integer", GT_OBJ, int_mtbl, sizeof(int_mtbl)/sizeof(Vfunc)
    );
    guru_register_func(GT_INT, NULL, NULL, guru_int_cmp);
}

// Float
#if GURU_USE_FLOAT
//================================================================
/*! (operator) unary -
 */
__CFUNC__
flt_negative(GR r[], S32 ri)
{
    GF f = _FLOAT(0);
    RETURN_FLOAT(-f);
}

#if GURU_USE_MATH
//================================================================
/*! (operator) ** power
 */
__CFUNC__
flt_power(GR r[], S32 ri)
{
    GF n = 0;
    switch (r[1].gt) {
    case GT_INT: 	n = r[1].i;	break;
    case GT_FLOAT:	n = r[1].d;	break;
    default: break;
    }

    RETURN_FLOAT(pow(r[0].d, n));
}
#endif // GURU_USE_MATH

//================================================================
/*! (method) abs
 */
__CFUNC__
flt_abs(GR r[], S32 ri)
{
    if (r[0].f < 0) {
        r[0].f = -r[0].f;
    }
}

//================================================================
/*! (method) to_i
 */
__CFUNC__
flt_to_i(GR r[], S32 ri)
{
    GI i = (GI)_FLOAT(0);
    RETURN_INT(i);
}

//================================================================
/*! initialize class Float
 */
__GURU__ __const__ Vfunc flt_mtbl[] = {
	{ "-@", 		flt_negative	},
#if     GURU_USE_MATH
	{ "**", 		flt_power		},
#endif // GURU_USE_MATH
	{ "abs", 		flt_abs			},
	{ "to_i", 		flt_to_i		},
	{ "to_s", 		gr_to_s			},
	{ "inspect", 	gr_to_s			}
};
__GURU__ void
guru_init_class_float(void)
{
    guru_rom_add_class(GT_FLOAT, "Float", GT_OBJ, flt_mtbl, VFSZ(flt_mtbl));
    guru_register_func(GT_FLOAT, NULL, NULL, guru_flt_cmp);
}

#endif // GURU_USE_FLOAT
