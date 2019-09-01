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

#if GURU_USE_STRING
#include "c_string.h"
#include "puts.h"
#endif

//================================================================
/*! x-bit left shift for x
 */
__GURU__ GI
_shift(GI x, GI y)
{
    // Don't support environments that include padding in int.
    const U32 INT_BITS = sizeof(GI) * CHAR_BIT;

    if (y >= INT_BITS)  return 0;
    if (y >= 0)         return x << y;
    if (y <= -INT_BITS) return 0;

    return x >> -y;
}

//================================================================
/*! (operator) [] bit reference
 */
__GURU__ void
c_int_bitref(GV v[], U32 argc)
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
c_int_negative(GV v[], U32 argc)
{
    GI num = GET_INT_ARG(0);
    SET_INT_RETURN(-num);
}

//================================================================
/*! (operator) ** power
 */
__GURU__ void
c_int_power(GV v[], U32 argc)
{
    if (v[1].gt == GT_INT) {
        GI x = 1;

        if (v[1].i < 0) x = 0;
        for (U32 i=0; i < v[1].i; i++) {
            x *= v[0].i;;
        }
        SET_INT_RETURN(x);
    }

#if GURU_USE_FLOAT && GURU_USE_MATH
    else if (v[1].gt == GT_FLOAT) {
        SET_FLOAT_RETURN(pow(v[0].i, v[1].f));
    }
#endif
}


//================================================================
/*! (operator) %
 */
__GURU__ void
c_int_mod(GV v[], U32 argc)
{
    GI num = GET_INT_ARG(1);
    SET_INT_RETURN(v->i % num);
}

//================================================================
/*! (operator) &; bit operation AND
 */
__GURU__ void
c_int_and(GV v[], U32 argc)
{
    GI num = GET_INT_ARG(1);
    SET_INT_RETURN(v->i & num);
}

//================================================================
/*! (operator) |; bit operation OR
 */
__GURU__ void
c_int_or(GV v[], U32 argc)
{
    GI num = GET_INT_ARG(1);
    SET_INT_RETURN(v->i | num);
}

//================================================================
/*! (operator) ^; bit operation XOR
 */
__GURU__ void
c_int_xor(GV v[], U32 argc)
{
    GI num = GET_INT_ARG(1);
    SET_INT_RETURN(v->i ^ num);
}

//================================================================
/*! (operator) ~; bit operation NOT
 */
__GURU__ void
c_int_not(GV v[], U32 argc)
{
    GI num = GET_INT_ARG(0);
    SET_INT_RETURN(~num);
}

//================================================================
/*! (operator) <<; bit operation LEFT_SHIFT
 */
__GURU__ void
c_int_lshift(GV v[], U32 argc)
{
    U32 num = GET_INT_ARG(1);
    SET_INT_RETURN(_shift(v->i, num));
}

//================================================================
/*! (operator) >>; bit operation RIGHT_SHIFT
 */
__GURU__ void
c_int_rshift(GV v[], U32 argc)
{
    U32 num = GET_INT_ARG(1);
    SET_INT_RETURN(_shift(v->i, -num));
}

//================================================================
/*! (method) abs
 */
__GURU__ void
c_int_abs(GV v[], U32 argc)
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
c_int_to_f(GV v[], U32 argc)
{
    GF f = GET_INT_ARG(0);
    SET_FLOAT_RETURN(f);
}
#endif

#if GURU_USE_STRING
//================================================================
/*! (method) chr
 */
__GURU__ void
c_int_chr(GV v[], U32 argc)
{
    U8 buf[2] = { (U8)GET_INT_ARG(0), '\0' };

    SET_RETURN(guru_str_new(buf));
}

//================================================================
/*! (method) to_s
 */
__GURU__ void
c_int_to_s(GV v[], U32 argc)
{
	U32 i    = GET_INT_ARG(0);
    U32 bias = 'a' - 10;
    U32 base = 10;

    if (argc) {
        base = GET_INT_ARG(1);
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    U8  buf[64+2];				// int64 + terminate + 1
    U8P p = buf + sizeof(buf) - 1;
    U32 x;
    *p = '\0';
    do {
        x = i % base;
        *--p = (x < 10)? x + '0' : x + bias;
        x /= base;
    } while (x != 0);

    SET_RETURN(guru_str_new(buf));
}
#endif

__GURU__ void
guru_init_class_int(void)
{
    // int
    guru_class *c = guru_class_int = guru_add_class("int", guru_class_object);

    guru_add_proc(c, "[]", 		c_int_bitref);
    guru_add_proc(c, "-@", 		c_int_negative);
    guru_add_proc(c, "**", 		c_int_power);
    guru_add_proc(c, "%", 		c_int_mod);
    guru_add_proc(c, "&", 		c_int_and);
    guru_add_proc(c, "|", 		c_int_or);
    guru_add_proc(c, "^", 		c_int_xor);
    guru_add_proc(c, "~", 		c_int_not);
    guru_add_proc(c, "<<", 		c_int_lshift);
    guru_add_proc(c, ">>", 		c_int_rshift);
    guru_add_proc(c, "abs",		c_int_abs);
#if GURU_USE_FLOAT
    guru_add_proc(c, "to_f",	c_int_to_f);
#endif
#if GURU_USE_STRING
    guru_add_proc(c, "chr", 	c_int_chr);
    guru_add_proc(c, "inspect",	c_int_to_s);
    guru_add_proc(c, "to_s", 	c_int_to_s);
#endif
}

// Float
#if GURU_USE_FLOAT
//================================================================
/*! (operator) unary -
 */
__GURU__ void
c_float_negative(GV v[], U32 argc)
{
    GF num = GET_FLOAT_ARG(0);
    SET_FLOAT_RETURN(-num);
}

#if GURU_USE_MATH
//================================================================
/*! (operator) ** power
 */
__GURU__ void
c_float_power(GV v[], U32 argc)
{
    GF n = 0;
    switch (v[1].gt) {
    case GT_INT: 	n = v[1].i;	break;
    case GT_FLOAT:	n = v[1].d;	break;
    default: break;
    }

    SET_FLOAT_RETURN(pow(v[0].d, n));
}
#endif

//================================================================
/*! (method) abs
 */
__GURU__ void
c_float_abs(GV v[], U32 argc)
{
    if (v[0].f < 0) {
        v[0].f = -v[0].f;
    }
}

//================================================================
/*! (method) to_i
 */
__GURU__ void
c_float_to_i(GV v[], U32 argc)
{
    GI i = (GI)GET_FLOAT_ARG(0);
    SET_INT_RETURN(i);
}

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
c_float_to_s(GV v[], U32 argc)
{
	guru_na("float#to_s");
}
#endif

//================================================================
/*! initialize class Float
 */
__GURU__ void
guru_init_class_float(void)
{
    // Float
    guru_class *c = guru_class_float = guru_add_class("Float", guru_class_object);

    guru_add_proc(c, "-@", 		c_float_negative);
#if GURU_USE_MATH
    guru_add_proc(c, "**", 		c_float_power);
#endif
    guru_add_proc(c, "abs", 	c_float_abs);
    guru_add_proc(c, "to_i", 	c_float_to_i);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect", c_float_to_s);
    guru_add_proc(c, "to_s", 	c_float_to_s);
#endif
}

#endif
