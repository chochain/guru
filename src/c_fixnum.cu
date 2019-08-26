/*! @file
  @brief
  mruby/c Fixnum and Float class

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "value.h"
#include "static.h"
#include "class.h"
#include "c_fixnum.h"

#if GURU_USE_STRING
#include "sprintf.h"
#include "c_string.h"
#endif

//================================================================
/*! x-bit left shift for x
 */
__GURU__ mrbc_int
_shift(mrbc_int x, mrbc_int y)
{
    // Don't support environments that include padding in int.
    const U32 INT_BITS = sizeof(mrbc_int) * CHAR_BIT;

    if (y >= INT_BITS)  return 0;
    if (y >= 0)         return x << y;
    if (y <= -INT_BITS) return 0;

    return x >> -y;
}

//================================================================
/*! (operator) [] bit reference
 */
__GURU__ void
c_fixnum_bitref(mrbc_value v[], U32 argc)
{
    if (0 <= v[1].i && v[1].i < 32) {
        SET_INT_RETURN((v[0].i & (1 << v[1].i)) ? 1 : 0);
    }
    else {
        SET_INT_RETURN(0);
    }
}

//================================================================
/*! (operator) unary -
 */
__GURU__ void
c_fixnum_negative(mrbc_value v[], U32 argc)
{
    mrbc_int num = GET_INT_ARG(0);
    SET_INT_RETURN(-num);
}

//================================================================
/*! (operator) ** power
 */
__GURU__ void
c_fixnum_power(mrbc_value v[], U32 argc)
{
    if (v[1].tt == GURU_TT_FIXNUM) {
        mrbc_int x = 1;

        if (v[1].i < 0) x = 0;
        for (U32 i=0; i < v[1].i; i++) {
            x *= v[0].i;;
        }
        SET_INT_RETURN(x);
    }

#if GURU_USE_FLOAT && GURU_USE_MATH
    else if (v[1].tt == GURU_TT_FLOAT) {
        SET_FLOAT_RETURN(pow(v[0].i, v[1].f));
    }
#endif
}


//================================================================
/*! (operator) %
 */
__GURU__ void
c_fixnum_mod(mrbc_value v[], U32 argc)
{
    mrbc_int num = GET_INT_ARG(1);
    SET_INT_RETURN(v->i % num);
}

//================================================================
/*! (operator) &; bit operation AND
 */
__GURU__ void
c_fixnum_and(mrbc_value v[], U32 argc)
{
    mrbc_int num = GET_INT_ARG(1);
    SET_INT_RETURN(v->i & num);
}

//================================================================
/*! (operator) |; bit operation OR
 */
__GURU__ void
c_fixnum_or(mrbc_value v[], U32 argc)
{
    mrbc_int num = GET_INT_ARG(1);
    SET_INT_RETURN(v->i | num);
}

//================================================================
/*! (operator) ^; bit operation XOR
 */
__GURU__ void
c_fixnum_xor(mrbc_value v[], U32 argc)
{
    mrbc_int num = GET_INT_ARG(1);
    SET_INT_RETURN(v->i ^ num);
}

//================================================================
/*! (operator) ~; bit operation NOT
 */
__GURU__ void
c_fixnum_not(mrbc_value v[], U32 argc)
{
    mrbc_int num = GET_INT_ARG(0);
    SET_INT_RETURN(~num);
}

//================================================================
/*! (operator) <<; bit operation LEFT_SHIFT
 */
__GURU__ void
c_fixnum_lshift(mrbc_value v[], U32 argc)
{
    U32 num = GET_INT_ARG(1);
    SET_INT_RETURN(_shift(v->i, num));
}

//================================================================
/*! (operator) >>; bit operation RIGHT_SHIFT
 */
__GURU__ void
c_fixnum_rshift(mrbc_value v[], U32 argc)
{
    U32 num = GET_INT_ARG(1);
    SET_INT_RETURN(_shift(v->i, -num));
}

//================================================================
/*! (method) abs
 */
__GURU__ void
c_fixnum_abs(mrbc_value v[], U32 argc)
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
c_fixnum_to_f(mrbc_value v[], U32 argc)
{
    mrbc_float f = GET_INT_ARG(0);
    SET_FLOAT_RETURN(f);
}
#endif

#if GURU_USE_STRING
//================================================================
/*! (method) chr
 */
__GURU__ void
c_fixnum_chr(mrbc_value v[], U32 argc)
{
    U8 buf[2] = { (U8)GET_INT_ARG(0), '\0' };

    SET_RETURN(mrbc_string_new(buf));
}

//================================================================
/*! (method) to_s
 */
__GURU__ void
c_fixnum_to_s(mrbc_value v[], U32 argc)
{
    U32 base = 10;
    if (argc) {
        base = GET_INT_ARG(1);
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    U8 buf[64+2];
    guru_vprintf(buf, "%d", v, 1);

    SET_RETURN(mrbc_string_new(buf));
}
#endif

__GURU__ void
mrbc_init_class_fixnum(void)
{
    // Fixnum
    mrbc_class *c = mrbc_class_fixnum = guru_add_class("Fixnum", mrbc_class_object);

    guru_add_proc(c, "[]", 		c_fixnum_bitref);
    guru_add_proc(c, "-@", 		c_fixnum_negative);
    guru_add_proc(c, "**", 		c_fixnum_power);
    guru_add_proc(c, "%", 		c_fixnum_mod);
    guru_add_proc(c, "&", 		c_fixnum_and);
    guru_add_proc(c, "|", 		c_fixnum_or);
    guru_add_proc(c, "^", 		c_fixnum_xor);
    guru_add_proc(c, "~", 		c_fixnum_not);
    guru_add_proc(c, "<<", 		c_fixnum_lshift);
    guru_add_proc(c, ">>", 		c_fixnum_rshift);
    guru_add_proc(c, "abs",		c_fixnum_abs);
    guru_add_proc(c, "to_i",	c_nop);
#if GURU_USE_FLOAT
    guru_add_proc(c, "to_f",	c_fixnum_to_f);
#endif
#if GURU_USE_STRING
    guru_add_proc(c, "chr", 	c_fixnum_chr);
    guru_add_proc(c, "inspect",	c_fixnum_to_s);
    guru_add_proc(c, "to_s", 	c_fixnum_to_s);
#endif
}

// Float
#if GURU_USE_FLOAT

//================================================================
/*! (operator) unary -
 */
__GURU__ void
c_float_negative(mrbc_value v[], U32 argc)
{
    mrbc_float num = GET_FLOAT_ARG(0);
    SET_FLOAT_RETURN(-num);
}

#if GURU_USE_MATH
//================================================================
/*! (operator) ** power
 */
__GURU__ void
c_float_power(mrbc_value v[], U32 argc)
{
    mrbc_float n = 0;
    switch (v[1].tt) {
    case GURU_TT_FIXNUM: n = v[1].i;	break;
    case GURU_TT_FLOAT:	 n = v[1].d;	break;
    default: break;
    }

    SET_FLOAT_RETURN(pow(v[0].d, n));
}
#endif

//================================================================
/*! (method) abs
 */
__GURU__ void
c_float_abs(mrbc_value v[], U32 argc)
{
    if (v[0].f < 0) {
        v[0].f = -v[0].f;
    }
}

//================================================================
/*! (method) to_i
 */
__GURU__ void
c_float_to_i(mrbc_value v[], U32 argc)
{
    mrbc_int i = (mrbc_int)GET_FLOAT_ARG(0);
    SET_INT_RETURN(i);
}

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
c_float_to_s(mrbc_value v[], U32 argc)
{
	U8 buf[64+2];
    guru_vprintf(buf, "%g", v, argc);
    
    SET_RETURN(mrbc_string_new(buf));
}
#endif

//================================================================
/*! initialize class Float
 */
__GURU__ void
mrbc_init_class_float(void)
{
    // Float
    mrbc_class *c = mrbc_class_float = guru_add_class("Float", mrbc_class_object);

    guru_add_proc(c, "-@", 		c_float_negative);
#if GURU_USE_MATH
    guru_add_proc(c, "**", 		c_float_power);
#endif
    guru_add_proc(c, "abs", 	c_float_abs);
    guru_add_proc(c, "to_i", 	c_float_to_i);
    guru_add_proc(c, "to_f", 	c_nop);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect", c_float_to_s);
    guru_add_proc(c, "to_s", 	c_float_to_s);
#endif
}

#endif
