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

#if GURU_USE_STRING
//================================================================
/*! (method) chr
 */
__GURU__ void
int_chr(GV v[], U32 argc)
{
    U8 buf[2] = { (U8)ARG_INT(0), '\0' };

    RETURN_VAL(guru_str_new(buf));
}

//================================================================
/*! (method) to_s
 */
__GURU__ void
int_to_s(GV v[], U32 argc)
{
	U32 i    = ARG_INT(0);
    U32 bias = 'a' - 10;
    U32 base = 10;

    if (argc) {
        base = ARG_INT(1);
        if (base < 2 || base > 36) return;	// raise ? ArgumentError
    }
    U8  buf[64+2];							// int64 + terminate + 1
    U8P p = buf + sizeof(buf) - 1;			// fill from the tail of the buffer
    U32 x;
    *p = '\0';
    do {
        x = i % base;
        *--p = (x < 10)? x + '0' : x + bias;
        x /= base;
    } while (x != 0);

    RETURN_VAL(guru_str_new(p));
}
#endif

__GURU__ void
guru_init_class_int(void)
{
    // int
    guru_class *c = guru_class_int = guru_add_class("int", guru_class_object);

    guru_add_proc(c, "[]", 		int_bitref);
    guru_add_proc(c, "-@", 		int_negative);
    guru_add_proc(c, "**", 		int_power);
    guru_add_proc(c, "%", 		int_mod);
    guru_add_proc(c, "&", 		int_and);
    guru_add_proc(c, "|", 		int_or);
    guru_add_proc(c, "^", 		int_xor);
    guru_add_proc(c, "~", 		int_not);
    guru_add_proc(c, "<<", 		int_lshift);
    guru_add_proc(c, ">>", 		int_rshift);
    guru_add_proc(c, "abs",		int_abs);
#if GURU_USE_FLOAT
    guru_add_proc(c, "to_f",	int_to_f);
#endif
#if GURU_USE_STRING
    guru_add_proc(c, "chr", 	int_chr);
    guru_add_proc(c, "inspect",	int_to_s);
    guru_add_proc(c, "to_s", 	int_to_s);
#endif
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
flt__abs(GV v[], U32 argc)
{
    if (v[0].f < 0) {
        v[0].f = -v[0].f;
    }
}

//================================================================
/*! (method) to_i
 */
__GURU__ void
flt__to_i(GV v[], U32 argc)
{
    GI i = (GI)ARG_FLOAT(0);
    RETURN_INT(i);
}

#if GURU_USE_STRING
//================================================================
/*! (method) to_s
 */
__GURU__ void
flt__to_s(GV v[], U32 argc)
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

    guru_add_proc(c, "-@", 		flt__negative);
#if GURU_USE_MATH
    guru_add_proc(c, "**", 		flt__power);
#endif
    guru_add_proc(c, "abs", 	flt__abs);
    guru_add_proc(c, "to_i", 	flt__to_i);
#if GURU_USE_STRING
    guru_add_proc(c, "inspect", flt__to_s);
    guru_add_proc(c, "to_s", 	flt__to_s);
#endif
}

#endif
