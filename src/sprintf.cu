/*! @file
  @brief
  console output module. (not yet input)

  <pre>
  Copyright (C) 2015-2018 Kyushu Institute of Technology.
  Copyright (C) 2015-2018 Shimane IT Open-Innovation Center.

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <stdarg.h>

#include "alloc.h"
#include "value.h"
#include "symbol.h"
#include "console.h"
#include "sprintf.h"

#define BUF_STEP_SIZE 80

//================================================================
/*! initialize data container.

  @param  pf	pointer to mrbc_printf
  @param  buf	pointer to output buffer.
  @param  size	buffer size.
  @param  fstr	format string.
*/
__GURU__ void
_init(mrbc_printf *pf, U8P buf, U32 sz, const U8P fstr)
{
    pf->buf = pf->p = buf;    
    pf->end = (U8 *)buf + sz - 1;
    pf->fstr= fstr;
    pf->fmt = (mrbc_print_fmt){0};
}

//================================================================
/*! return string length in buffer

  @param  pf	pointer to mrbc_printf
  @return	length
*/
__GURU__ __INLINE__ int
_size(mrbc_printf *pf)
{
    return pf->end - pf->p;
}

//================================================================
/*! sprintf subcontract function

  @param  pf	pointer to mrbc_printf
  @retval 0	(format string) done.
  @retval 1	found a format identifier.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ int
_next(mrbc_printf *pf)
{
    U8  ch = '\0';
    pf->fmt = (mrbc_print_fmt){0};

    while (_size(pf) && (ch = *pf->fstr) != '\0') {
        pf->fstr++;
        if (ch == '%') {
            if (*pf->fstr == '%') {	// is "%%"
                pf->fstr++;
            } else {
                goto PARSE_FLAG;
            }
        }
        *pf->p++ = ch;
#if 0		// no auto buffer resize, to prevent memory leak
        if (!_size(pf)) _resize(pf);
#endif
    }
    return -(_size(pf) && ch != '\0');

PARSE_FLAG:
    // parse format - '%' [flag] [width] [.prec] type
    //   e.g. "%05d"
    while ((ch = *pf->fstr)) {
        switch(ch) {
        case '+': pf->fmt.plus  = 1; break;
        case ' ': pf->fmt.space = 1; break;
        case '-': pf->fmt.minus = 1; break;
        case '0': pf->fmt.zero  = 1; break;
        default : goto PARSE_WIDTH;
        }
        pf->fstr++;
    }

PARSE_WIDTH:
	S32 n;
    while ((n = *pf->fstr - '0'), (0 <= n && n <= 9)) {	// isdigit()
        pf->fmt.width = pf->fmt.width * 10 + n;
        pf->fstr++;
    }
    if (*pf->fstr == '.') {
        pf->fstr++;
        while ((n = *pf->fstr - '0'), (0 <= n && n <= 9)) {
            pf->fmt.prec = pf->fmt.prec * 10 + n;
            pf->fstr++;
        }
    }
    if (*pf->fstr) pf->fmt.type = *pf->fstr++;

    return 1;
}

//================================================================
/*! clear output buffer in container.

  @param  pf	pointer to mrbc_printf
*/
__GURU__ __INLINE__ void
_clear(mrbc_printf *pf)
{
    pf->p = pf->buf;		// back to head
}

//================================================================
/*! terminate ('\0') output buffer.

  @param  pf	pointer to mrbc_printf
*/
__GURU__ __INLINE__ void
_end(mrbc_printf *pf)
{
    *pf->p = '\0';
}

//================================================================
/*! replace output buffer

  @param  pf	pointer to mrbc_printf
  @param  buf	pointer to output buffer.
  @param  size	buffer size.
*/
__GURU__ U32
_resize(mrbc_printf *pf)
{
    U32 sz  = _size(pf);
	if (sz > 0) return 1;

    U32 off = pf->p - pf->buf;					        // current offset
    U32 inc = BUF_STEP_SIZE;
	while ((sz + inc) < pf->fmt.width) {
		inc += BUF_STEP_SIZE;
	}

	U32 nsz = (pf->end - pf->buf + 1) + inc;	        // new size
	U8  *nbuf = (U8 *)mrbc_realloc(pf->buf, nsz);	// deep copy
	if (!nbuf) return 0;

    pf->buf = nbuf;
    pf->end = nbuf + nsz - 1;                           // last byte for '\0'
    pf->p   = pf->buf + off;				            // reset pointer

    return 1;
}

//================================================================
/*! sprintf subcontract function for U8 '%c'

  @param  pf	pointer to mrbc_printf
  @param  ch	output character (ASCII)
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ S32
__char(mrbc_printf *pf, U8 ch)
{
    if (pf->fmt.minus) {
        if (_size(pf)) *pf->p++ = ch;
        else return -1;
    }

    U32 width = pf->fmt.width;
    while (--width > 0) {
        if (_size(pf)) *pf->p++ = ' ';
        else return -1;
    }
    if (!pf->fmt.minus) {
        if (_size(pf)) *pf->p++ = ch;
        else return -1;
    }

    return 0;
}

//================================================================
/*! sprintf subcontract function for byte array.

  @param  pf	pointer to mrbc_printf.
  @param  str	pointer to byte array.
  @param  len	byte length.
  @param  pad	padding character.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ S32
__bstr(mrbc_printf *pf, U8P str, U32 len, U8 pad)
{
    S32 ret = 0;

    if (str == NULL) {
        str = (U8P)"(null)";
        len = 6;
    }
    if (pf->fmt.prec && len > pf->fmt.prec) len = pf->fmt.prec;

    S32 tw = len;
    if (pf->fmt.width > len) tw = pf->fmt.width;

    U32 remain = _size(pf);
    if (len > remain) {
        len = remain;
        ret = -1;
    }
    if (tw > remain) {
        tw = remain;
        ret = -1;
    }

    S32 n_pad = tw - len;

    if (!pf->fmt.minus) {
        while (n_pad-- > 0) {
            *pf->p++ = pad;
        }
    }
    while (len-- > 0) {
        *pf->p++ = *str++;
    }
    while (n_pad-- > 0) {
        *pf->p++ = pad;
    }
    return ret;
}

#if GURU_USE_FLOAT
//================================================================
/*! sprintf subcontract function for float(double) '%f'

  @param  pf	pointer to mrbc_printf.
  @param  value	output value.
  @retval 0	done.
  @retval -1	buffer full.
*/
__GURU__ int
__float(mrbc_printf *pf, double value)
{
    U8 fstr[16];
    const U8 *p0 = pf->fstr;
    U8       *p1 = fstr + sizeof(fstr) - 1;

    *p1 = '\0';
    while ((*--p1 = *--p0) != '%');

    // TODO: 20181025 format print float
    //snprintf(pf->p, (pf->buf_end - pf->p + 1), p1, value);

    while (*pf->p != '\0')
        pf->p++;

    return _size(pf);
}
#endif

//================================================================
/*! sprintf subcontract function for U8 '%s'

  @param  pf	pointer to mrbc_printf.
  @param  str	output string.
  @param  pad	padding character.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ S32
__str(mrbc_printf *pf, const U8P str, U8 pad)
{
    return __bstr(pf, str, STRLEN(str), pad);
}

//================================================================
/*! sprintf subcontract function for integer '%d' '%x' '%b'

  @param  pf	pointer to mrbc_printf.
  @param  value	output value.
  @param  base	n base.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GURU__ S32
__int(mrbc_printf *pf, mrbc_int value, U32 base)
{
    U32 sign = 0;
    U32 v = value;			// (note) Change this when supporting 64 bit.

    if (pf->fmt.type == 'd' || pf->fmt.type == 'i') {	// signed.
        if (value < 0) {
            sign = '-';
            v = -value;
        } else if (pf->fmt.plus) {
            sign = '+';
        } else if (pf->fmt.space) {
            sign = ' ';
        }
    }
    if (pf->fmt.minus || pf->fmt.width == 0) {
        pf->fmt.zero = 0; 	// disable zero padding if left align or width zero.
    }
    pf->fmt.prec = 0;

    U32 bias_a = (pf->fmt.type == 'X') ? 'A' - 10 : 'a' - 10;

    // create string to local buffer
    U8  buf[64+2];				// int64 + terminate + 1
    U8P p = buf + sizeof(buf) - 1;
    *p = '\0';
    do {
        U32 i = v % base;
        *--p = (i < 10)? i + '0' : i + bias_a;
        v /= base;
    } while (v != 0);

    // decide pad character and output sign character
    U8 pad;
    if (pf->fmt.zero) {
        pad = '0';
        if (sign) {
            *pf->p++ = sign;
            if (!_size(pf)) return -1;
            pf->fmt.width--;
        }
    } else {
        pad = ' ';
        if (sign) *--p = sign;
    }
    return __str(pf, (U8P)p, pad);
}

//================================================================
/*! output formatted string

  @param  fstr		format string.
*/
__GURU__ U8P
guru_sprintf(U8P buf, const U8 *fstr, ...)
{
    va_list ap;
    va_start(ap, fstr);

    U32 ret = 0;
    mrbc_printf pf;

    _init(&pf, buf, BUF_STEP_SIZE, (U8P)fstr);
    while (ret==0 && _next(&pf)) {
    	switch(pf.fmt.type) {
        case 'c': ret = __char(&pf, va_arg(ap, int));        	 break;
        case 's': ret = __str(&pf, va_arg(ap, U8 *), ' '); 	 break;
        case 'd':
        case 'i':
        case 'u': ret = __int(&pf, va_arg(ap, unsigned int), 10); break;
        case 'b':
        case 'B': ret = __int(&pf, va_arg(ap, unsigned int), 2);  break;
        case 'x':
        case 'X': ret = __int(&pf, va_arg(ap, unsigned int), 16); break;
#if GURU_USE_FLOAT
        case 'f':
        case 'e':
        case 'E':
        case 'g':
        case 'G': ret = __float(&pf, va_arg(ap, double)); 		 break;
#endif
        default:
            console_str("?format: ");
            console_char(pf.fmt.type);
            console_str("\n");
            break;
        }
    }
    va_end(ap);
    _end(&pf);

    return pf.buf;
}

__GURU__ U8P
guru_vprintf(U8P buf, const U8 *fstr, mrbc_value v[], U32 argc)		// << from c_string.cu
{
    U32 i   = 0;
    U32 ret = 0;
    mrbc_printf pf;

    _init(&pf, buf, BUF_STEP_SIZE, (U8P)fstr);

    while (ret==0 && _next(&pf)) {
        if (i > argc) {
        	console_str("ArgumentError\n");
        	return NULL;
        }

        switch(pf.fmt.type) {
        case 'c':
            if (v[i].tt==GURU_TT_FIXNUM) {
                ret = __char(&pf, v[i].i);
            }
            break;
        case 's':
            if (v[i].tt==GURU_TT_STRING) {
                ret = __str(&pf, VSTR(&v[i]), ' ');
            }
            else if (v[i].tt==GURU_TT_SYMBOL) {
                ret = __str(&pf, VSYM(&v[i]), ' ');
            }
            break;
        case 'd':
        case 'i':
        case 'u':
            if (v[i].tt==GURU_TT_FIXNUM) {
                ret = __int(&pf, v[i].i, 10);
#if GURU_USE_FLOAT
            } else if (v[i].tt==GURU_TT_FLOAT) {
                ret = __int(&pf, (mrbc_int)v[i].f, 10);
#endif
            } else if (v[i].tt==GURU_TT_STRING) {
                mrbc_int ival = ATOI(VSTR(&v[i]));
                ret = __int(&pf, ival, 10);
            }
            break;
        case 'b':
        case 'B':
            if (v[i].tt==GURU_TT_FIXNUM) {
                ret = __int(&pf, v[i].i, 2);
            }
            break;
        case 'x':
        case 'X':
            if (v[i].tt==GURU_TT_FIXNUM) {
                ret = __int(&pf, v[i].i, 16);
            }
            break;
#if GURU_USE_FLOAT
        case 'f':
        case 'e':
        case 'E':
        case 'g':
        case 'G':
            if (v[i].tt==GURU_TT_FLOAT) {
                ret = __float(&pf, v[i].f);
            }
            else if (v[i].tt==GURU_TT_FIXNUM) {
            	ret = __float(&pf, v[i].i);
            }
            break;
#endif
        default: break;
        }
        i++;
    }
    _end(&pf);

    return pf.buf;		// local variable, deep copy only
}


